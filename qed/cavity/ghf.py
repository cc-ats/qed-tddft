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

from qed.tdscf.dipole_field_coupling import dipole_dot_efield, magnetic_dipole_mat

def init_guess(diag, nstates=None, factor=1, resonance_state=None,
               e_threshold=None, e_threshold_l=None):
    e_max = diag.max()
    space = len(diag)
    nstates = min(nstates, space)

    if resonance_state:
        e_threshold_l = numpy.partition(diag, resonance_state-1)[resonance_state-1]
    if e_threshold_l:
        idx = numpy.where(diag >= e_threshold_l)[0]
        n = min(nstates, len(diag[idx])) - 1
        e_threshold = numpy.partition(diag[idx], n)[n]
    else:
        if e_threshold is None:
            e_threshold = numpy.partition(diag, nstates-1)[nstates-1]

    e_threshold = min(e_max, e_threshold)
    e_threshold += 1e-3 # deg_eia_thresh

    idx = numpy.where(diag <= e_threshold)[0]
    #print('idx:\n', idx)

    x0  = numpy.zeros((idx.size, factor*space))
    for i, j in enumerate(idx):
        x0[i, j] = 1.
    return x0

class CavityModel(lib.StreamObject):
    #def __init__(self, mf_obj, cavity_freq=None, cavity_mode=None, key):
    def __init__(self, mf_obj, key):
        if not isinstance(mf_obj, list):
            mf_obj = [mf_obj]

        cavity_freq = key.get('cavity_freq')
        cavity_mode = key.get('cavity_mode')
        if isinstance(cavity_freq, float):
            cavity_freq = [cavity_freq]
        self.cavity_freq = numpy.array(cavity_freq)
        self.cavity_mode = numpy.array(cavity_mode).reshape(3, -1)
        self.cavity_num  = self.cavity_freq.size

        self.build(mf_obj, key) # get dipole and nov dimension

    def check_sanity(self):
        assert isinstance(self.cavity_num, int)
        assert self.cavity_num > 0

    def build(self, mf_obj, key): # implemented in RHF or UHF subclasses
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
    def build(self, mf_obj, key):
        uniform_field = key.get('uniform_field', True)
        efield_file = key.get('efield_file', None)

        self.has_k = key.get('has_k', True) # k-term in dse contribution

        nfrag = len(mf_obj)
        self.elec_occupation = numpy.zeros(nfrag)
        #self.mo_coeff = [None]*nfrag
        self.orbo = [None]*nfrag
        self.orbv = [None]*nfrag
        self.nbas = [None]*nfrag
        self.nocc = [None]*nfrag

        nov, e_ia = [], []
        # electronic dipole integral * efield/photon coupling
        self.dip_scaled_oo = [None]*nfrag
        self.dip_scaled_vv = [None]*nfrag
        self.dip_scaled_ov = numpy.array([]).reshape(self.cavity_num, 0)
        # electronic dipole integral for transition dipole calculation
        self.dip_ov = numpy.array([]).reshape(3, 0)
        # magnetic dipole in ov
        self.mag_dip_ov = numpy.array([]).reshape(3, 0)

        for n in range(nfrag):
            mf        = mf_obj[n]
            #assert isinstance(mf, scf_type)
            scf_type = type(mf).__name__
            print('scf_type:', scf_type)
            if scf_type in {'RHF', 'RKS'}:
                occupation = 2
            elif scf_type in {'GHF', 'GKS'}:
                occupation = 1
            print('occupation:', occupation)
            self.elec_occupation[n] = occupation

            mo_occ    = mf.mo_occ
            occidx    = numpy.where(mo_occ==occupation)[0]
            viridx    = numpy.where(mo_occ==0)[0]

            mo_coeff  = mf.mo_coeff
            orbo      = mo_coeff[:,occidx]
            orbv      = mo_coeff[:,viridx]

            nocc      = len(occidx)
            nvir      = len(viridx)

            #self.mo_coeff[n] = numpy.copy(mo_coeff)
            self.orbo[n] = numpy.copy(orbo)
            self.orbv[n] = numpy.copy(orbv)
            self.nbas[n]     = mo_coeff.shape[0]
            self.nocc[n]     = nocc


            mo_energy = mf.mo_energy
            #e = mo_energy[viridx] - mo_energy[occidx,None]
            e = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
            nov.append(e.size)
            e_ia.append(e.ravel())

            dip_ao, dip_scaled = dipole_dot_efield(mf.mol, mf.grids,
                                                   self.cavity_mode,
                                                   uniform_field, efield_file,
                                                   scf_type)

            mag_dip = magnetic_dipole_mat(mf.mol, scf_type)

            # dipole without photon field for transition dipole moment calculation at the end
            # note PySCF uses matrix transpose instead
            dip_ov = numpy.einsum('xmn,mi,na->xia', dip_ao, orbo.conj(), orbv).reshape(3, -1)
            self.dip_ov = numpy.hstack((self.dip_ov, dip_ov))

            mag_dip_ov = numpy.einsum('xmn,mi,na->xia', mag_dip, orbo.conj(), orbv).reshape(3, -1)
            self.mag_dip_ov = numpy.hstack((self.mag_dip_ov, mag_dip_ov))

            dip_scaled = numpy.einsum('kmn,mp,nq->kpq', dip_scaled, mo_coeff.conj(), mo_coeff)
            dip_scaled *= numpy.sqrt(occupation)
            self.dip_scaled_oo[n] = dip_scaled[:, :nocc, :nocc]
            self.dip_scaled_vv[n] = dip_scaled[:, nocc:, nocc:]
            self.dip_scaled_ov = numpy.hstack((self.dip_scaled_ov, dip_scaled[:, :nocc, nocc:].reshape(self.cavity_num, -1)))

        # dimensions used in the polariton calculations
        self.accum_nov = numpy.insert(numpy.cumsum(nov), 0, 0)
        self.e_ia = numpy.reshape(e_ia, -1)


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

    def get_mns_weight(self, mns):
        weight = []
        for n in range(len(mns)):
            m, n = mns[n]
            w = numpy.einsum('a,a->', m.conj(), m) - numpy.einsum('a,a->', n.conj(), n)
            weight.append(w)
        return numpy.array(weight)

    def init_guess(self, nstates=None):
        if nstates is None:
            nstates = self.cavity_num

        self.amp_size = 2 * self.cavity_num
        return init_guess(self.cavity_freq, nstates, 2)

    def get_hdiag(self):
        hdiag = numpy.hstack((self.cavity_freq.ravel(), self.cavity_freq.ravel()))
        return hdiag

    def get_norms2(self, mns):
        amp_num  = mns.shape[0]
        ms, ns   = mns.transpose(1,0,2)
        norms2   = numpy.einsum('la,la->l', ms.conj(), ms) - numpy.einsum('la,la->l', ns.conj(), ns)
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

    def get_mns_weight(self, mns):
        weight = []
        for n in range(len(mns)):
            m, _ = mns[n]
            w = numpy.einsum('a,a->', m.conj(), m)
            weight.append(w)
        return numpy.array(weight)

    def init_guess(self, nstates=None):
        if nstates is None:
            nstates = self.cavity_num

        self.amp_size = self.cavity_num
        return init_guess(self.cavity_freq, nstates, 1)

    def get_hdiag(self):
        hdiag = self.cavity_freq.ravel()
        return hdiag

    def get_norms2(self, mns):
        amp_size = self.cavity_num
        mns      = mns.reshape(-1, amp_size)
        amp_num  = mns.shape[0]
        ms       = mns
        norms2   = numpy.einsum('la,la->l', ms.conj(), ms)
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
        cavity_freq = self.cavity_freq
        accum_nov   = self.accum_nov
        occupation  = self.elec_occupation

        dip_ov      = numpy.copy(self.dip_scaled_ov)
        #for n in numpy.where(occupation==2)[0]:
        #    n0, n1 = accum_nov[n], accum_nov[n+1]
        #    dip_ov[:,n0:n1] *= numpy.sqrt(occupation[n])

        def vind(zs, mns): # do we need zs*2 for double occupancy ?
            ms, ns   =  mns.transpose(1,0,2)
            tmp1     =  numpy.einsum('ln,pn->lp', zs, dip_ov)
            gzs      =  numpy.einsum('p,lp->lp', numpy.sqrt(cavity_freq/2.), tmp1)
            omega_ms =  numpy.einsum('p,lp->lp', cavity_freq, ms)
            omega_ns = -numpy.einsum('p,lp->lp', cavity_freq, ns)
            return numpy.hstack([omega_ms + gzs, omega_ns - gzs])

        return vind, self.get_hdiag()

    def gen_dse_resp(self):
        dip_ov  = self.dip_scaled_ov # has been scaled by sqrt(2.) for RHF
        dip_oo, dip_vv = self.dip_scaled_oo, self.dip_scaled_vv
        accum_nov = self.accum_nov
        occupation = self.elec_occupation
        nfrag = len(dip_oo)

        def vind(zs): # do we need zs*2 for double occupancy ?
            tmp1     =  numpy.einsum('ln,pn->lp', zs, dip_ov)
            delta_zs =  numpy.einsum('pn,lp->ln', dip_ov.conj(), tmp1) # j-type

            if self.has_k is False:
                return numpy.array([delta_zs, delta_zs])

            delta_z2 = numpy.copy(delta_zs)
            for n in range(nfrag):
                n0, n1 = accum_nov[n], accum_nov[n+1]
                doo, dvv = dip_oo[n], dip_vv[n]

                nocc, nvir = doo.shape[1], dvv.shape[1]
                dov = dip_ov[:,n0:n1].reshape(-1, nocc, nvir)

                pov = zs[:,n0:n1].reshape(-1, nocc, nvir)

                # for B matrix
                # same j-type, already has 2 from above
                #tmp2 = delta_zs[:,n0:n1].reshape(-1, nocc, nvir).transpose(0,2,1)
                tmp2 = numpy.einsum('pja,pib,ljb->lia', dov, dov, pov) # k-type
                delta_z2[:,n0:n1] -= tmp2.reshape(-1, n1-n0) / occupation[n]

                # k-type for A matrix
                tmp3 = numpy.einsum('pab,pji,ljb->lia', dvv, doo, pov)
                delta_zs[:,n0:n1] -= tmp3.reshape(-1, n1-n0) / occupation[n]

            return numpy.array([delta_zs, delta_z2])

        return vind

RPF = RestrictedPauliFierz


class RestrictedRabi(RestrictedCavityModel, Rabi):
    gen_ph_resp = RestrictedPauliFierz.gen_ph_resp


class RestrictedRotatingWaveApproximation(RestrictedCavityModel, RotatingWaveApproximation):
    def gen_ph_resp(self):
        cavity_freq = self.cavity_freq
        accum_nov   = self.accum_nov
        occupation  = self.elec_occupation

        dip_ov      = numpy.copy(self.dip_scaled_ov)
        #for n in numpy.where(occupation==2)[0]:
        #    n0, n1 = accum_nov[n], accum_nov[n+1]
        #    dip_ov[:,n0:n1] *= numpy.sqrt(occupation[n])

        def vind(zs, mns):
            ms       = mns
            tmp1     = numpy.einsum('ln,pn->lp', zs, dip_ov)
            gzs      = numpy.einsum('p,lp->lp', numpy.sqrt(cavity_freq/2.), tmp1)
            omega_ms = numpy.einsum('p,lp->lp', cavity_freq, ms)
            return (omega_ms + gzs)

        return vind, self.get_hdiag()

    gen_dse_resp = RestrictedPauliFierz.gen_dse_resp

RestrictedRWA = RestrictedRotatingWaveApproximation
RRWA          = RestrictedRotatingWaveApproximation


class RestrictedJaynesCummings(RestrictedCavityModel, JaynesCummings):
    gen_ph_resp = RestrictedRotatingWaveApproximation.gen_ph_resp

RJC = RestrictedJaynesCummings
