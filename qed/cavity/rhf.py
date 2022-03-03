from   functools import reduce
import numpy

from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf.hf import RHF
from pyscf.scf import _response_functions  # noqa
from pyscf.data import nist
from pyscf import __config__

class CavityModel(lib.StreamObject):
    def __init__(self, mf_obj=None, cavity_mode=None, cavity_freq=None):
        self._scf        = mf_obj
        self.cavity_freq = cavity_freq
        self.cavity_mode = cavity_mode
        self.cavity_num  = None
        self._mol        = None
        self.dip_ov      = None
        self.amp_size    = None
        self.mns         = None

    def setup_cavity(self, cavity_mode=None, cavity_freq=None):
        if cavity_mode is None:
            cavity_mode = self.cavity_mode
        if cavity_freq is None:
            cavity_freq = self.cavity_freq

        if (cavity_mode is not None) and (cavity_freq is not None):
            self.cavity_freq = numpy.asarray(cavity_freq)
            self.cavity_num  = self.cavity_freq.size
            self.cavity_mode = numpy.asarray(cavity_mode).reshape(3, self.cavity_num)
        else:
            self.cavity_freq = None
            self.cavity_mode = None
            self.cavity_num  = 0

    def reset(self, mol=None, mo_coeff = None):
        if mol is not None:
            self._mol = mol

        if mo_coeff is not None:
            self._scf.mo_coeff = mo_coeff

        self._scf.reset(mol)
        self.dip_ov = None

        return self

    def check_sanity(self):
        assert self._scf.converged
        assert isinstance(self.cavity_num, int)
        assert self.cavity_num > 0

    def build(self, mf_obj): # implemented in RHF or UHF subclasses
        raise NotImplementedError

    def get_amps(self, amps): # implemented in cavity model subclasses
        raise NotImplementedError

    def get_mns(self, amps): # implemented in cavity model subclasses
        raise NotImplementedError

    def init_guess(self):
        raise NotImplementedError

    def get_hdiag(self):      # implemented in cavity model subclasses
        raise NotImplementedError

    def get_norms2(self, mns):  # implemented in cavity model subclasses
        raise NotImplementedError

    def gen_ph_resp(self):    # implemented in cavity models
        raise NotImplementedError

    def gen_dse_resp(self):   # implemented in cavity models
        raise NotImplementedError

class RestrictedCavityModel(CavityModel):
    def build(self):
        self.setup_cavity()
        mol       = self._scf.mol
        dip_ao    = mol.intor("int1e_r", comp=3)
        self._mol = mol
        mf_obj    = self._scf
        self.check_sanity()
        assert isinstance(mf_obj, RHF)

        mo_coeff = mf_obj.mo_coeff
        print("mo_coeff = ", mo_coeff)
        assert mo_coeff.dtype == numpy.double

        mo_occ    = mf_obj.mo_occ
        occidx    = numpy.where(mo_occ>0)[0]
        viridx    = numpy.where(mo_occ==0)[0]
        
        nocc      = len(occidx)
        nvir      = len(viridx)
        orbo      = mo_coeff[:,occidx]
        orbv      = mo_coeff[:,viridx]

        dip_ov      = numpy.einsum('xmn,mi,na->xia', dip_ao, orbo.conj(), orbv)
        self.dip_ov = dip_ov.reshape(3, nocc, nvir)
        

class PauliFierz(CavityModel):
    def get_amps(self, amps):
        cavity_num = self.cavity_num
        amp_size   = 2*cavity_num
        mns        = amps[:, (-amp_size):]
        amp_num    = mns.shape[0]
        mns        = mns.reshape(amp_num, 2, cavity_num)
        ls         = mns[:, 0, :] + mns[:, 1, :]
        return ls, mns

    def get_mns(self, amps):
        ls, mns = self.get_amps(amps)
        return [(m, n) for m,n in mns]

    def init_guess(self, nstates=None):
        if nstates is None:
            nstates = self.cavity_num
            
        e_cav0      = self.cavity_freq
        e_cav0_max  = e_cav0.max()
        num_cav     = self.cavity_num
        
        nstates      = min(nstates, num_cav)
        e_threshold  = min(e_cav0_max, e_cav0[numpy.argsort(e_cav0)[nstates-1]])
        e_threshold += 1e-6

        idx          = numpy.where(e_cav0 <= e_threshold)[0]
        x0           = numpy.zeros((idx.size, 2*num_cav))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return x0

    def get_hdiag(self):
        hdiag = numpy.hstack((self.cavity_freq.ravel(), self.cavity_freq.ravel()))
        return hdiag

    def get_norms2(self, mns):
        amp_size = 2 * self.cavity_num
        amp_num  = mns.shape[0]
        ms, ns   = mns.transpose(1,0,2)
        norms2   = numpy.einsum("la,la->l", ms.conj(), ms) - numpy.einsum("la,la->l", ns.conj(), ns)
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

PF = PauliFierz

class Rabi(PauliFierz):
    def gen_dse_resp(self):
        def vind(zs):
            return None
        return vind

class RotatingWaveApproximation(PauliFierz):
    def get_amps(self, amps):
        cavity_num = self.cavity_num
        amp_size   = cavity_num
        mns        = amps[:, (-amp_size):]
        amp_num    = mns.shape[0]
        mns        = mns.reshape(amp_num, cavity_num)
        ls         = mns
        return ls, mns

    def get_mns(self, amps):
        ms, ms = self.get_amps(amps)
        return [(m, 0) for m in ms]

    def init_guess(self, nstates=None):
        if nstates is None:
            nstates = self.cavity_num
            
        e_cav0      = self.cavity_freq
        e_cav0_max  = e_cav0.max()
        num_cav     = self.cavity_num
        
        nstates      = min(nstates, num_cav)
        e_threshold  = min(e_cav0_max, e_cav0[numpy.argsort(e_cav0)[nstates-1]])
        e_threshold += 1e-6

        idx          = numpy.where(e_cav0 <= e_threshold)[0]
        x0           = numpy.zeros((idx.size, num_cav))
        for i, j in enumerate(idx):
            x0[i, j] = 1
        return x0

    def get_hdiag(self):
        hdiag = self.cavity_freq.ravel()
        return hdiag

    def get_norms2(self, mns):
        amp_size = self.cavity_num
        mns      = mns.reshape(-1, amp_size)
        amp_num  = mns.shape[0]
        ms       = mns
        norms2   = numpy.einsum("la,la->l", ms.conj(), ms)
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

RWA = RotatingWaveApproximation

class JaynesCummings(RotatingWaveApproximation, Rabi):
    pass

JC = JaynesCummings

class RestrictedPauliFierz(RestrictedCavityModel, PauliFierz):
    def gen_ph_resp(self):
        dip_ov      = self.dip_ov
        nocc        = self.dip_ov.shape[1]
        nvir        = self.dip_ov.shape[2]
        
        amp_size    = 2*self.cavity_num
        cavity_mode = self.cavity_mode
        cavity_freq = self.cavity_freq

        def vind(zs, mns): # do we need zs*2 for double occupancy ?
            amp_num  =  zs.shape[0]
            ms, ns   =  mns.transpose(1,0,2)
            zs       =  zs.reshape(amp_num, nocc, nvir)
            tmp1     =  numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     =  numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            gzs      =  numpy.einsum("p,lp->lp", numpy.sqrt(cavity_freq), tmp2)
            omega_ms =  numpy.einsum("p,lp->lp", cavity_freq, ms) 
            omega_ns = -numpy.einsum("p,lp->lp", cavity_freq, ns) 
            return numpy.hstack([omega_ms + gzs, omega_ns - gzs]).reshape(amp_num, amp_size)
        
        return vind, self.get_hdiag()

    def gen_dse_resp(self):
        dip_ov  = self.dip_ov
        nocc    = self.dip_ov.shape[1]
        nvir    = self.dip_ov.shape[2]
        cavity_mode = self.cavity_mode

        def vind(zs): # do we need zs*2 for double occupancy ?
            amp_num  =  zs.shape[0]
            zs       =  zs.reshape(amp_num, nocc, nvir)
            tmp1     =  numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     =  numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            tmp3     =  numpy.einsum("xp,lp->lx", cavity_mode, tmp2)
            delta_zs =  numpy.einsum("xia,lx->lia", dip_ov,    tmp3)
            return 2*delta_zs.reshape(amp_num, nocc*nvir)

        return vind

RPF = RestrictedPauliFierz

class RestrictedRabi(RestrictedCavityModel, Rabi):
    def gen_ph_resp(self):
        dip_ov      = self.dip_ov
        nocc        = self.dip_ov.shape[1]
        nvir        = self.dip_ov.shape[2]
        
        amp_size    = 2 * self.cavity_num
        cavity_mode = self.cavity_mode
        cavity_freq = self.cavity_freq

        def vind(zs, mns): # do we need zs*2 for double occupancy ?
            amp_num  =  zs.shape[0]
            ms, ns   =  mns.transpose(1,0,2)
            zs       =  zs.reshape(amp_num, nocc, nvir)
            tmp1     =  numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     =  numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            gzs      =  numpy.einsum("p,lp->lp", numpy.sqrt(cavity_freq), tmp2)
            omega_ms =  numpy.einsum("p,lp->lp", cavity_freq, ms) 
            omega_ns = -numpy.einsum("p,lp->lp", cavity_freq, ns) 
            return numpy.hstack([omega_ms + gzs, omega_ns - gzs]).reshape(amp_num, amp_size)
        
        return vind, self.get_hdiag()

class RestrictedRotatingWaveApproximation(RestrictedCavityModel, RotatingWaveApproximation):
    def gen_ph_resp(self):
        dip_ov  = self.dip_ov
        nocc    = self.dip_ov.shape[1]
        nvir    = self.dip_ov.shape[2]

        amp_size    = self.cavity_num
        cavity_mode = self.cavity_mode
        cavity_freq = self.cavity_freq

        def vind(zs, mns): # do we need zs*2 for double occupancy ?
            amp_num  = zs.shape[0]
            ms       = mns
            zs       = zs.reshape(amp_num, nocc, nvir)
            tmp1     = numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     = numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            gzs      = numpy.einsum("p,lp->lp", numpy.sqrt(cavity_freq), tmp2)
            omega_ms = numpy.einsum("p,lp->lp", cavity_freq, ms) 
            return (omega_ms + gzs).reshape(amp_num, amp_size)
        
        return vind, self.get_hdiag()

    def gen_dse_resp(self):
        dip_ov  = self.dip_ov
        nocc    = self.dip_ov.shape[1]
        nvir    = self.dip_ov.shape[2]
        cavity_mode = self.cavity_mode

        def vind(zs): # do we need zs*2 for double occupancy ?
            amp_num  = zs.shape[0]
            zs       = zs.reshape(amp_num, nocc, nvir)

            tmp1     =  numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     =  numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            tmp3     =  numpy.einsum("xp,lp->lx", cavity_mode, tmp2)
            delta_zs =  numpy.einsum("xia,lx->lia", dip_ov,    tmp3)
            return 2*delta_zs.reshape(amp_num, nocc, nvir)

        return vind

RestrictedRWA = RestrictedRotatingWaveApproximation
RRWA          = RestrictedRotatingWaveApproximation

class RestrictedJaynesCummings(RestrictedCavityModel, JaynesCummings):
    def gen_ph_resp(self):
        dip_ov      = self.dip_ov
        nocc        = self.dip_ov.shape[1]
        nvir        = self.dip_ov.shape[2]

        amp_size    = self.cavity_num
        cavity_mode = self.cavity_mode
        cavity_freq = self.cavity_freq

        def vind(zs, mns): # do we need zs*2 for double occupancy ?
            amp_num  = zs.shape[0]
            zs       = zs.reshape(amp_num, nocc, nvir)
            ms       = mns

            tmp1     = numpy.einsum("lia,xia->lx", zs, dip_ov)
            tmp2     = numpy.einsum("xp,lx->lp", cavity_mode, tmp1)
            gzs      = numpy.einsum("p,lp->lp", numpy.sqrt(cavity_freq), tmp2)
            omega_ms = numpy.einsum("p,lp->lp", cavity_freq, ms) 
            return (omega_ms + gzs).reshape(amp_num, amp_size)
        
        return vind, self.get_hdiag()

RJC = RestrictedJaynesCummings