from pyscf.scf.hf  import RHF
from pyscf.dft.rks import KohnShamDFT

from qed.grad import tdrhf, tdrks

# TODO: use `_is_dft_object(mf) = getattr(mf, 'xc', None) is not None and hasattr(mf, '_numint')`
def Gradients(td): 
    scf_obj = td._scf
    assert isinstance(scf_obj, RHF)

    if isinstance(scf_obj, KohnShamDFT):
        return tdrks.Gradients(td)
    else:
        return tdrhf.Gradients(td)