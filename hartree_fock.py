import pyscf
import h5py
import numpy as np 
import pyscf.mcscf

def hf(txt, chkfile='hf.chk'):
    mol = pyscf.gto.Mole(atom = txt, basis = 'cc-pvdz', unit='bohr')
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()



import pyscf
def casci(chkfile='hf.chk', casfile = 'casci.chk', nroots=1,**kwargs):

    mol = pyscf.lib.chkfile.load_mol(chkfile)
    mf = pyscf.scf.RHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))

    mycas = pyscf.mcscf.CASCI(mf,**kwargs)
    mycas.fcisolver.nroots=nroots
    mycas.kernel()
    with h5py.File(casfile,'w') as f:
        f['e_tot'] = mycas.e_tot
        f['ci'] = np.asarray(mycas.ci)
        f['nroots']= nroots
        for k,v in kwargs.items():
            f[k]=v
