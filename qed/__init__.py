import pyscf
from   pyscf import scf, dft

import qed
from   qed.tdscf.rhf  import TDASym, TDANoSym
from   qed.cavity.rhf import RotatingWaveApproximation

from   qed.grad       import Gradients

def JC(mf, cavity_mode=None, cavity_freq=None):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.rhf.RestrictedJaynesCummings(mf_obj=mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)

def Rabi(mf, cavity_mode=None, cavity_freq=None):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.rhf.RestrictedRabi(mf_obj=mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)

def RWA(mf, cavity_mode=None, cavity_freq=None):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.rhf.RestrictedRotatingWaveApproximation(mf_obj=mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)

def PF(mf, cavity_mode=None, cavity_freq=None):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return qed.cavity.rhf.RestrictedPauliFierz(mf_obj=mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)

def TDA(mf_obj, cav_obj=None):
    if isinstance(mf_obj, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf_obj = scf.addons.convert_to_rhf(mf_obj)
        td_obj = pyscf.tdscf.TDA(mf_obj)
        if isinstance(cav_obj, RotatingWaveApproximation):
            return TDASym(td_obj, cav_obj)
        else:
            return TDANoSym(td_obj, cav_obj)

def RPA(mf_obj, cav_obj=None):
    if isinstance(mf_obj, scf.uhf.UHF):
        raise NotImplementedError
    else:
        mf_obj = scf.addons.convert_to_rhf(mf_obj)
        td_obj = None
        if isinstance(mf_obj, dft.rks.KohnShamDFT):
            td_obj = pyscf.tdscf.rks.TDDFT(mf_obj)
        else:
            mf_obj = scf.addons.convert_to_rhf(mf_obj)
            td_obj = pyscf.tdscf.RPA(mf_obj)
        
        return qed.tdscf.rhf.RPA(td_obj, cav_obj)

TDDFT = RPA
