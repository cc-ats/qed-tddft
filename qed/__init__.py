import pyscf
from   pyscf import scf, dft

import qed
from   qed.tdscf.ghf  import TDASym, TDANoSym
from   qed.cavity.ghf import RotatingWaveApproximation

from   qed.grad       import Gradients

def JC(mf, key):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.ghf.RestrictedJaynesCummings(mf_obj=mf, key=key)

def Rabi(mf, key):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.ghf.RestrictedRabi(mf_obj=mf, key=key)

def RWA(mf, key):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.ghf.RestrictedRotatingWaveApproximation(mf_obj=mf, key=key)

def PF(mf, key):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.ghf.RestrictedPauliFierz(mf_obj=mf, key=key)

def TDA(mf_obj, td_obj, cav_obj, key):
    if isinstance(mf_obj, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf_obj = scf.addons.convert_to_rhf(mf_obj)
        #td_obj = pyscf.tdscf.TDA(mf_obj)
        if isinstance(cav_obj, RotatingWaveApproximation):
            return TDASym(td_obj, cav_obj, key)
        else:
            return TDANoSym(td_obj, cav_obj, key)

def RPA(mf_obj, td_obj, cav_obj, key):
    if isinstance(mf_obj, scf.uhf.UHF):
        raise NotImplementedError
    else:
        #mf_obj = scf.addons.convert_to_rhf(mf_obj)
        #td_obj = None
        #if isinstance(mf_obj, dft.rks.KohnShamDFT):
        #    td_obj = pyscf.tdscf.rks.TDDFT(mf_obj)
        #else:
        #    mf_obj = scf.addons.convert_to_rhf(mf_obj)
        #    td_obj = pyscf.tdscf.RPA(mf_obj)

        return qed.tdscf.ghf.RPA(td_obj, cav_obj, key)

TDDFT = RPA
