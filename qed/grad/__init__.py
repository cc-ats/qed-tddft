from pyscf.scf.hf  import RHF
from pyscf.dft.rks import KohnShamDFT

from qed.grad import tdrhf, tdrks

def Gradients(td):
    scf_obj = td._scf
    assert isinstance(scf_obj, RHF)

    if isinstance(scf_obj, KohnShamDFT):
        return tdrks.Gradients(td)
    else:
        return tdrhf.Gradients(td)