import pyscf
def hf(txt, chkfile='hf.chk'):
    mol = pyscf.gto.Mole(atom = txt, basis = 'cc-pvdz', unit='bohr')
    mol.build()
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()
