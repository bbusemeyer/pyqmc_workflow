import pyscf
import pyqmc
import numpy as np
import h5py
import pyqmc.dasktools


class QMCManager:
    def __init__(self, chkfile, client=None, npartitions=1):
        self.mol = pyscf.lib.chkfile.load_mol(chkfile)
        self.mol.output = None
        self.mol.stdout = None
        self.mf = pyscf.scf.RHF(self.mol)
        self.mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))
        self.wf, self.to_opt, self.freeze = pyqmc.default_sj(self.mol, self.mf, ion_cusp=True)
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
            self.mol, self.wf, to_opt=self.to_opt, freeze=self.freeze
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
                vmcoptions={"npartitions": self.npartitions}
            )

    def dmc(self, nconfig=1000, **kwargs):
        configs = pyqmc.initial_guess(self.mol, nconfig)
        if self.client is None:
            pyqmc.vmc(self.wf, configs, nsteps=10)
            pyqmc.rundmc(
                self.wf,
                configs,
                accumulators={"energy": pyqmc.EnergyAccumulator(self.mol)},
                **kwargs
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
                npartitions=self.npartitions
            )
