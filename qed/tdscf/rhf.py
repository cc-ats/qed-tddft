# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
# Recent Advances in Density Functional Methods, Chapter 5, M. E. Casida
#
import numpy

import pyscf
from pyscf        import lib
from pyscf.scf.hf import RHF
from pyscf.lib    import logger
from pyscf.data   import nist
from pyscf        import __config__

from pyscf.lib import davidson1, davidson_nosym1

from pyscf.tdscf.rhf import get_ab
from pyscf.tdscf.rhf import get_nto, oscillator_strength
from pyscf.tdscf.rhf import _contract_multipole
from pyscf.tdscf.rhf import transition_dipole
from qed.tdscf.dipole_field_coupling import dipole_dot_efield_on_grid

OUTPUT_THRESHOLD      = getattr(__config__, 'tdscf_rhf_get_nto_threshold',             0.3)
REAL_EIG_THRESHOLD    = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold',     1e-4)
MO_BASE               = getattr(__config__, 'MO_BASE',                                   1)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)

# What I need? td_obj.get_norms2(xys), cav_obj.get_norms2(mns)
# In the eigen part

def get_qed_tdscf_operation(td_obj, cav_obj):
    get_elec_resp, elec_hdiag = td_obj.gen_elec_resp()
    get_ph_resp, ph_hdiag     = cav_obj.gen_ph_resp()
    get_dse_resp              = cav_obj.gen_dse_resp()

    hdiag    = numpy.hstack((elec_hdiag.ravel(), ph_hdiag.ravel()))
    amp_size = hdiag.size

    get_elec_amps = td_obj.get_amps
    get_ph_amps   = cav_obj.get_amps

    def vind(amps):
        amps     = numpy.asarray(amps).reshape(-1, amp_size)
        num_amps = amps.shape[0]
        zs, xys  = get_elec_amps(amps)  # TDA: zs = xs; RPA: zs = xs + ys
        ls, mns  = get_ph_amps(amps)    # RWA: ls = ms; PF:  ls = ms + ns

        dse_resp    = get_dse_resp(zs) # give dse_resp = None for JC and Rabi
        elec_resp   = get_elec_resp(xys, ls, dse_resp=dse_resp)
        ph_resp     = get_ph_resp(zs, mns)

        tot_resp    = numpy.hstack((elec_resp.reshape(num_amps,-1), ph_resp.reshape(num_amps,-1)))
        return tot_resp.reshape(num_amps, amp_size)

    return vind, hdiag

def get_g_block(td_obj, cav_obj):
    mol       = td_obj.mol
    mf_obj    = td_obj._scf

    dip_ao    = mol.intor("int1e_r", comp=3)
    mo_coeff  = mf_obj.mo_coeff
    mo_occ    = mf_obj.mo_occ
    occidx    = numpy.where(mo_occ>0)[0]
    viridx    = numpy.where(mo_occ==0)[0]
    nocc      = len(occidx)
    nvir      = len(viridx)
    orbo      = mo_coeff[:,occidx]
    orbv      = mo_coeff[:,viridx]

    cav_omegas = cav_obj.cavity_freq
    cav_lams   = cav_obj.cavity_mode
    cav_nums   = cav_obj.cavity_num

    dip_dot_efield = dipole_dot_efield_on_grid(mol, mf_obj, cav_obj)
    g_block   = lib.einsum('pmn,mi,na->pia', dip_dot_efield, orbo.conj(), orbv)
    g_block   = lib.einsum('p,pia->pia', numpy.sqrt(cav_omegas), g_block)

    return g_block.reshape(cav_nums, nocc, nvir)

def get_dse_block(td_obj, cav_obj):
    mol       = td_obj.mol
    mf_obj    = td_obj._scf

    dip_ao    = mol.intor("int1e_r", comp=3)
    mo_coeff  = mf_obj.mo_coeff
    mo_occ    = mf_obj.mo_occ
    occidx    = numpy.where(mo_occ>0)[0]
    viridx    = numpy.where(mo_occ==0)[0]
    nocc      = len(occidx)
    nvir      = len(viridx)
    orbo      = mo_coeff[:,occidx]
    orbv      = mo_coeff[:,viridx]

    cav_lams   = cav_obj.cavity_mode

    dip_dot_efield = dipole_dot_efield_on_grid(mol, mf_obj, cav_obj)
    dse_block = lib.einsum('pmn,mi,na->pia', dip_dot_efield, orbo.conj(), orbv)
    dse_block = lib.einsum('pia,pjb->iajb', dse_block, dse_block)

    return 2*dse_block.reshape(nocc, nvir, nocc, nvir)

def get_ab_block(td_obj):
    return get_ab(td_obj._scf)

def get_omega_block(cav_obj):
    return numpy.diag(cav_obj.cavity_freq).reshape(cav_obj.cavity_num, cav_obj.cavity_num)

class TDMixin(lib.StreamObject):
    conv_tol    = getattr(__config__, 'tdscf_rhf_TDA_conv_tol',  1e-9)
    nstates     = getattr(__config__, 'tdscf_rhf_TDA_nstates',      3)
    singlet     = getattr(__config__, 'tdscf_rhf_TDA_singlet',   True)
    lindep      = getattr(__config__, 'tdscf_rhf_TDA_lindep',   1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift',  0)
    max_space   = getattr(__config__, 'tdscf_rhf_TDA_max_space',   50)
    max_cycle   = getattr(__config__, 'tdscf_rhf_TDA_max_cycle',  100)

    def __init__(self, tdobj, cav_obj):
        self.td_obj     = tdobj
        self._nov       = None
        self.cav_obj    = cav_obj

        self.verbose    = tdobj.verbose
        self.stdout     = tdobj.stdout
        self.mol        = tdobj.mol
        self._scf       = tdobj._scf
        self.max_memory = tdobj.max_memory
        self.chkfile    = tdobj.chkfile
        self.wfnsym     = None

        # xy[i] = (X_I,Y_I), In TDA, Y_I = 0
        # mn[i] = (M_I,N_I), In RWA, N_I = 0
        # Normalized to 1:
        # For restricted case:
        # 2(X_I X_I - Y_I Y_I) + (M_I M_I - N_I N_I) = 1
        # For unrestricted case:
        # (Xa_I Xa_I - Ya_I Ya_I) + (Xb_I Xb_I - Yb_I Yb_I) + (M_I M_I - N_I N_I) = 1
        self.converged = None
        self.e         = None
        self.xy        = None
        self.mn        = None

        keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift', 'max_space', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def nroots(self):
        return self.nstates
    @nroots.setter
    def nroots(self, x):
        self.nstates = x

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.e

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s from %s-%s-%s ********', self.__class__, self.td_obj.__class__, self.cav_obj.__class__, self._scf.__class__)
        log.info("cQED-TDDFT:          %s", "Yang, et. al. J. Chem. Phys. 155, 064107 (2021) https://doi.org/10.1063/5.0057542")
        log.info("Analytical Gradient: %s", "Yang, et. al. J. Chem. Phys. 156, 124104 (2022) https://doi.org/10.1063/5.0082386")

        mf_obj    = self._scf
        assert isinstance(mf_obj, RHF)
        mo_occ    = mf_obj.mo_occ
        occidx    = numpy.where(mo_occ>0)[0]
        viridx    = numpy.where(mo_occ==0)[0]
        nocc      = len(occidx)
        nvir      = len(viridx)
        self._nov = nocc * nvir

        if self.singlet:
            log.info('nov              = %d', self._nov)
            log.info('nstates          = %d singlet', self.nstates)
        else:
            log.info('nov              = %d', self._nov)
            log.info('nstates          = %d triplet', self.nstates)

        if self.cav_obj.cavity_num == 0:
            log.info("No cavity mode in the calculation.")
        else:
            log.info("%d cavity mode in the calculation:", self.cav_obj.cavity_num)
            for alpha in range(self.cav_obj.cavity_num):
                log.info("Cavity %4d: freq = % 6.4f, mode = (% 6.4f, % 6.4f, % 6.4f)",
                alpha+1, self.cav_obj.cavity_freq[alpha],
                self.cav_obj.cavity_mode[0, alpha], self.cav_obj.cavity_mode[1, alpha], self.cav_obj.cavity_mode[2, alpha]
                )

        log.info('wfnsym           = %s', self.wfnsym)
        log.info('conv_tol         = %g', self.conv_tol)
        log.info('eigh lindep      = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_space   = %d', self.max_space)
        log.info('eigh max_cycle   = %d', self.max_cycle)
        log.info('chkfile          = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    def check_sanity(self):
        if self._scf.mo_coeff is None:
            raise RuntimeError('SCF object is not initialized')
        lib.StreamObject.check_sanity(self)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def get_ab_block(self, td_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        return get_ab_block(td_obj)

    def get_g_block(self, td_obj=None, cav_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_g_block(td_obj, cav_obj)

    def get_dse_block(self, td_obj=None, cav_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_dse_block(td_obj, cav_obj)

    def get_omega_block(self, cav_obj=None):
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_omega_block(cav_obj)

    def gen_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if not all(self.converged):
            logger.note(self, 'TD-SCF states %s not converged.',
                        [i for i, x in enumerate(self.converged) if not x])
        logger.note(self, 'Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self

    get_nto             = get_nto
    oscillator_strength = oscillator_strength
    _contract_multipole = _contract_multipole  # needed by following methods
    transition_dipole   = transition_dipole

    def gen_vind(self, cav_obj=None):
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_qed_tdscf_operation(self, cav_obj)

    def gen_eigen_solver(self):
        raise NotImplementedError

    def gen_elec_resp(self, td_obj=None):
        raise NotImplementedError

    def get_init_guess(self, nstates=None):
        if nstates is None: nstates = self.nstates
        td_x0  = self.init_guess(self._scf, nstates=nstates)
        cav_x0 = self.cav_obj.init_guess(nstates=nstates)

        x0 = numpy.block(
        [[td_x0,   numpy.zeros([td_x0.shape[0], cav_x0.shape[1]])],
         [numpy.zeros([cav_x0.shape[0], td_x0.shape[1]]),  cav_x0]]
        )
        return x0

    def nuc_grad_method(self):
        from qed import grad
        return grad.Gradients(self)

    def init_guess(self, mf_obj, nstates=None):
        raise NotImplementedError

    def get_amps(self, amps):
        raise NotImplementedError

    def get_norms2(self, xys):
        raise NotImplementedError

    def get_xys(amps):
        raise NotImplementedError

    def kernel(self, amp0=None, nstates=None):
        raise NotImplementedError

class TDASym(TDMixin):
    def gen_eigen_solver(self):
        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:,idx], idx
        return davidson1, pickeig

    def gen_elec_resp(self, td_obj=None, cav_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        assert isinstance(td_obj, pyscf.tdscf.rhf.TDA)

        if cav_obj is None:
            cav_obj = self.cav_obj
        td_obj.singlet = self.singlet
        td_obj.wfnsym  = self.wfnsym
        amp_size       = self._nov

        ge_ov       = cav_obj.ge_ov
        cavity_mode = cav_obj.cavity_mode
        cavity_freq = cav_obj.cavity_freq

        vind0, hdiag   = td_obj.gen_vind(td_obj._scf)

        def vind(xs, ls, dse_resp=None):
            amp_num = xs.shape[0]
            axs     = vind0(xs)
            tmp1    = numpy.einsum("lp,p->lp", ls, numpy.sqrt(cavity_freq))
            gls     = numpy.einsum("pia,lp->lia", ge_ov, tmp1).reshape(amp_num, amp_size)
            if dse_resp is None:
                return (axs + gls).reshape(amp_num, amp_size)
            else:
                dse_resp = dse_resp.reshape(amp_num, amp_size)
                return (axs + gls + dse_resp).reshape(amp_num, amp_size)
        return vind, hdiag

    def get_amps(self, amps):
        xs = amps[:, :self._nov]
        return xs, xs

    def get_xys(self, amps, fac=None):
        xs, xs = self.get_amps(amps)
        if fac is None:
            return [(x, 0) for x in xs]
        else:
            return [(x * fac, 0) for x in xs]

    def get_norms2(self, xys):
        amp_size = self._nov
        xys      = xys.reshape(-1, amp_size)
        amp_num  = xys.shape[0]
        xs       = xys
        norms2   = numpy.einsum("li,li->l", xs.conj(), xs)
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

    def init_guess(self, mf_obj, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf_obj.mo_energy
        mo_occ    = mf_obj.mo_occ
        occidx    = numpy.where(mo_occ==2)[0]
        viridx    = numpy.where(mo_occ==0)[0]
        e_ia      = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia_max  = e_ia.max()

        nov         = e_ia.size
        nstates     = min(nstates, nov)
        e_ia        = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0  = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1          # Koopmans' excitations
        return x0

    def kernel(self, amp0=None, nstates=None):
        assert self.singlet
        td_obj  = self.td_obj
        cav_obj = self.cav_obj
        cav_obj.build()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates      = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag              = self.gen_vind(cav_obj=cav_obj)
        precond                  = self.gen_precond(hdiag)
        davidson_solver, pickeig = self.gen_eigen_solver()

        if amp0 is None:
            amp0 = self.get_init_guess(nstates=nstates)

        converged, e, amps  = davidson_solver(
                              vind, amp0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log
                              )

        # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.converged = converged[:nstates]
        self.e         = e[:nstates]
        amps           = amps[:nstates]
        nstates = self.e.size
        amps    = numpy.asarray(amps).reshape(nstates, -1)

        zs, xys     = self.get_amps(amps)
        ls, mns     = self.cav_obj.get_amps(amps)
        norms2_elec = self.get_norms2(xys)
        norms2_ph   = self.cav_obj.get_norms2(mns)
        norms2      = (norms2_elec + norms2_ph).reshape(nstates)
        amps        = numpy.einsum("li,l->li", amps, 1/numpy.sqrt(norms2))

        if self.verbose > 3:
            for istate, xy in enumerate(xys):
                log.info("istate = %4d, norms2_elec = % 6.4f, norms2_ph = % 6.4f", istate, norms2_elec[istate], norms2_ph[istate])

        mo_occ  = self._scf.mo_occ
        nocc    = sum(mo_occ==2)
        nvir    = sum(mo_occ==0)

        xys = self.get_xys(amps, fac=1.0/numpy.sqrt(2)) # For alpha beta spin

        self.xy = []
        for x, y in xys:
            if isinstance(y, int):
                assert y == 0
                self.xy.append((x.reshape(nocc, nvir), 0))
            else:
                self.xy.append((x.reshape(nocc, nvir), y.reshape(nocc, nvir)))

        self.mn = self.cav_obj.get_mns(amps)

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('QED-TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy, self.mn

class TDANoSym(TDASym):
    def gen_eigen_solver(self):
        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > POSTIVE_EIG_THRESHOLD))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=True)
        return davidson_nosym1, pickeig

class RPA(TDANoSym):
    def gen_elec_resp(self, td_obj=None, cav_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        assert isinstance(td_obj, pyscf.tdscf.rhf.RPA)

        if cav_obj is None:
            cav_obj = self.cav_obj
        td_obj.singlet = self.singlet
        td_obj.wfnsym  = self.wfnsym
        amp_size       = 2 * self._nov

        ge_ov  = cav_obj.ge_ov
        nocc   = ge_ov.shape[1]
        nvir   = ge_ov.shape[2]
        cavity_mode = cav_obj.cavity_mode
        cavity_freq = cav_obj.cavity_freq

        vind0, hdiag   = td_obj.gen_vind(td_obj._scf)
        def vind(xys, ls, dse_resp=None):
            xys     = numpy.asarray(xys).reshape(-1, 2, nocc, nvir)
            amp_num = xys.shape[0]
            abxys   = vind0(xys)

            abxys   = abxys.reshape(amp_num, 2, nocc, nvir)
            abxys1, abxys2 = abxys.transpose(1,0,2,3)
            abxys1  = abxys1.reshape(amp_num, nocc, nvir)
            abxys2  = abxys2.reshape(amp_num, nocc, nvir)

            tmp1 = numpy.einsum("lp,p->lp", ls, numpy.sqrt(cavity_freq))
            gls  = numpy.einsum("pia,px->lia", ge_ov, tmp2).reshape(amp_num, nocc, nvir)

            abxys1 = (abxys1 + gls)
            abxys2 = (abxys2 - gls)

            if dse_resp is None:
                return numpy.hstack([abxys1, abxys2]).reshape(amp_num, amp_size)
            else:
                dse_resp = dse_resp.reshape(amp_num, nocc, nvir)
                return numpy.hstack([abxys1+dse_resp, abxys2-dse_resp]).reshape(amp_num, amp_size)
        return vind, hdiag

    def get_amps(self, amps):
        xys = amps[:, :2*self._nov]
        xys = numpy.asarray(xys).reshape(-1,2,self._nov)
        xs, ys = xys.transpose(1,0,2)
        return xs + ys, xys

    def get_xys(self, amps, fac=None):
        zs, xys = self.get_amps(amps)

        if fac is None:
            return [(x, y) for x, y in xys]
        else:
            return [(x * fac, y * fac) for x, y in xys]


    def get_norms2(self, xys):
        amp_size = 2 * self._nov
        xys      = xys.reshape(-1, amp_size)
        amp_num  = xys.shape[0]
        xs       = xys[:, :self._nov]
        ys       = xys[:, self._nov:]
        norms2   = (numpy.einsum("li,li->l", xs.conj(), xs) - numpy.einsum("li,li->l",ys.conj(), ys))
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

    def init_guess(self, mf_obj, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf_obj.mo_energy
        mo_occ    = mf_obj.mo_occ
        occidx    = numpy.where(mo_occ==2)[0]
        viridx    = numpy.where(mo_occ==0)[0]
        e_ia      = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia_max  = e_ia.max()

        nov         = e_ia.size
        nstates     = min(nstates, nov)
        e_ia        = e_ia.ravel()
        e_threshold = min(e_ia_max, e_ia[numpy.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0  = numpy.zeros((idx.size, 2*nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1          # Koopmans' excitations
        return x0
