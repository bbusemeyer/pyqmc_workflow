import pyscf
import pyqmc
import numpy as np
import h5py
import pyqmc.dasktools


def _make_wf_sj(mol, mf, jastrow, jastrow_kws, slater_kws=None, mc=None):
    """
    mol and mf are pyscf objects

    jastrow may be either a function that returns wf, to_opt, or 
    a list of such functions.

    jastrow_kws is a dictionary of keyword arguments for the jastrow function, or
    a list of those functions.
    """
    if jastrow_kws is None:
        jastrow_kws = {}

    if slater_kws is None:
        slater_kws = {}

    if not isinstance(jastrow, list):
        jastrow = [jastrow]
        jastrow_kws = [jastrow_kws]

    if mc is None:
        wf1, to_opt1 = pyqmc.default_slater(mol, mf, **slater_kws)
    else:
        wf1, to_opt1 = pyqmc.default_multislater(mol, mf, mc, **slater_kws)

    pack = [jast(mol, **kw) for jast, kw in zip(jastrow, jastrow_kws)]
    wfs = [p[0] for p in pack]
    to_opts = [p[1] for p in pack]
    wf = pyqmc.MultiplyWF(wf1, *wfs)
    to_opt = {"wf1" + k: v for k, v in to_opt1.items()}
    for i, to_opt2 in enumerate(to_opts):
        to_opt.update({f"wf{i+2}" + k: v for k, v in to_opt2.items()})
    return wf, to_opt


def _recover_pyscf(chkfile, casfile=None, root=0):
    mol = pyscf.lib.chkfile.load_mol(chkfile)
    mol.output = None
    mol.stdout = None
    mf = pyscf.scf.RHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))
    mc = None
    if casfile is not None:
        with h5py.File(casfile, "r") as f:
            mc = pyscf.mcscf.CASCI(mf, ncas=int(f["ncas"][...]), nelecas=f["nelecas"][...])
            mc.ci = f["ci"][root, ...]
    return mol, mf, mc


class QMCManager:
    def __init__(
        self,
        chkfile,
        client=None,
        npartitions=1,
        jastrow=pyqmc.default_jastrow,
        jastrow_kws=None,
        slater_kws=None,
        casfile=None,
        recover_pyscf = _recover_pyscf,
        make_wf_sj = _make_wf_sj
    ):
        self.mol, self.mf, self.mc = recover_pyscf(chkfile, casfile)
        self.wf, self.to_opt = make_wf_sj(
            self.mol, self.mf, jastrow, jastrow_kws, slater_kws=slater_kws, mc=self.mc
        )
        self.client = client
        self.npartitions = npartitions

    def read_wf(self, wf_file):
        with h5py.File(wf_file, "r") as hdf:
            if "wf" in hdf.keys():
                grp = hdf["wf"]
                for k in grp.keys():
                    self.wf.parameters[k] = np.array(grp[k])

    def optimize(self, nconfig=1000, **kwargs):
        configs = pyqmc.initial_guess(self.mol, nconfig)
        acc = pyqmc.gradient_generator(
            self.mol, self.wf, to_opt=self.to_opt
        )
        if self.client is None:
            pyqmc.line_minimization(self.wf, configs, acc, **kwargs)
        else:
            pyqmc.dasktools.line_minimization(
                self.wf,
                configs,
                acc,
                **kwargs,
                client=self.client,
                lmoptions={"npartitions": self.npartitions},
                vmcoptions={"npartitions": self.npartitions, 'nblocks':5,'nsteps_per_block':20},
            )

    def dmc(self, nconfig=1000, **kwargs):
        configs = pyqmc.initial_guess(self.mol, nconfig)
        if self.client is None:
            pyqmc.vmc(self.wf, configs, nsteps=10)
            pyqmc.rundmc(
                self.wf,
                configs,
                accumulators={"energy": pyqmc.EnergyAccumulator(self.mol)},
                **kwargs,
            )
        else:
            pyqmc.dasktools.distvmc(
                self.wf,
                configs,
                nsteps=10,
                client=self.client,
                npartitions=self.npartitions,
            )
            pyqmc.rundmc(
                self.wf,
                configs,
                accumulators={"energy": pyqmc.EnergyAccumulator(self.mol)},
                propagate=pyqmc.dasktools.distdmc_propagate,
                **kwargs,
                client=self.client,
                npartitions=self.npartitions,
            )
