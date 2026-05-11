# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
# Recent Advances in Density Functional Methods, Chapter 5, M. E. Casida
#
import numpy
import itertools

import pyscf
from pyscf        import lib, gto
from pyscf.scf    import RHF
from pyscf.lib    import logger
from pyscf.data   import nist
from pyscf        import __config__

from pyscf.lib import davidson1, davidson_nosym1
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig

from pyscf.tdscf.rhf import get_nto#, _contract_multipole, transition_magnetic_dipole

from qed.cavity.ghf import init_guess

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

    hdiag    = numpy.hstack((numpy.reshape(elec_hdiag, -1), ph_hdiag.ravel()))
    amp_size = hdiag.size

    get_elec_amps = td_obj.get_elec_amps
    get_ph_amps   = cav_obj.get_amps

    def vind(amps):
        amps     = numpy.asarray(amps).reshape(-1, amp_size)
        #num_amps = amps.shape[0]
        zs, xys  = get_elec_amps(amps)  # TDA: zs = xs; RPA: zs = xs + ys
        ls, mns  = get_ph_amps(amps)    # RWA: ls = ms; PF:  ls = ms + ns

        dse_resp    = get_dse_resp(zs) # give dse_resp = None for JC and Rabi
        elec_resp   = get_elec_resp(xys, ls, dse_resp=dse_resp)
        ph_resp     = get_ph_resp(zs, mns)

        #tot_resp    = numpy.hstack((elec_resp.reshape(num_amps,-1), ph_resp.reshape(num_amps,-1)))
        #return tot_resp.reshape(num_amps, amp_size)
        tot_resp  = numpy.hstack((elec_resp, ph_resp))
        return tot_resp

    return vind, hdiag

def get_g_block(cav_obj):
    cavity_freq = cav_obj.cavity_freq
    dip_ov  = cav_obj.dip_scaled_ov

    g_block = numpy.einsum('p,pn->pn', numpy.sqrt(cavity_freq/2.), dip_ov)
    return g_block

def get_dse_block(cav_obj, rpa=False, has_k=True):
    dip_ov = cav_obj.dip_scaled_ov
    dip_oo = cav_obj.dip_scaled_oo
    dip_vv = cav_obj.dip_scaled_vv
    dse_block = numpy.einsum('pm,pn->mn', dip_ov.conj(), dip_ov)

    if not has_k:
        if rpa:
            return numpy.array([dse_block, dse_block])
        else:
            return dse_block

    if rpa:
        dse_2 = numpy.copy(dse_block)

    accum_nov = cav_obj.accum_nov
    occupation = cav_obj.elec_occupation
    dip_oo, dip_vv = cav_obj.dip_scaled_oo, cav_obj.dip_scaled_vv

    for n in range(len(dip_oo)):
        n0, n1 = accum_nov[n], accum_nov[n+1]
        doo, dvv = dip_oo[n], dip_vv[n]

        # k-type
        tmp2 = numpy.einsum('pab,pji->iajb', dvv, doo)
        dse_block -= tmp2.reshape(-1, n1-n0) / occupation[n]

        if rpa:
            nocc, nvir = doo.shape[1], dvv.shape[1]
            dov = dip_ov[:,n0:n1].reshape(-1, nocc, nvir)
            tmp3 = numpy.einsum('pja,pib->iajb', dov, dov)
            dse_2 -= tmp3.reshape(-1, n1-n0) / occupation[n]

    if rpa:
        return numpy.array([dse_block, dse_2])
    else:
        return dse_block

def get_dse_block2(xs, dip_ov, dip_oo, dip_vv, occupation, rpa=False, has_k=True):
    # memory efficient on single fragment
    dip = numpy.einsum('sm,pm->ps', xs, dip_ov)
    dse_block = numpy.einsum('ps,pt->st', dip.conj(), dip)

    if not has_k:
        if rpa:
            return numpy.array([dse_block, dse_block])
        else:
            return dse_block

    nocc, nvir = dip_oo.shape[1], dip_vv.shape[1]
    xs = xs.reshape(-1, nocc, nvir)

    if rpa:
        dse_2 = numpy.copy(dse_block)

    # k-type of A
    tmp2 = numpy.einsum('pab,pji,tjb->tia', dip_vv, dip_oo, xs)
    #tmp2 = numpy.einsum('pij,pba,tjb->tia', dip_oo, dip_vv, xs)
    dse_block -= numpy.einsum('sia,tia->st', xs.conj(), tmp2) / occupation

    if rpa:
        dov = dip_ov.reshape(-1, nocc, nvir)
        tmp3 = numpy.einsum('pja,pib,tjb->tia', dov, dov, xs)
        dse_2 -= numpy.einsum('sia,tia->st', xs.conj(), tmp3) / occupation
        return numpy.array([dse_block, dse_2])
    else:
        return dse_block

def get_dse_block_off(xs, dip_ov, accum_nov, rpa=False):
    nfrag = len(dip_ov)
    dip = [None]*nfrag
    for n in range(nfrag):
        n0, n1 = accum_nov[n], accum_nov[n+1]
        dip[n] = numpy.einsum('sm,pm->ps', xs[n], dip_ov[n, n0:n1])

    nstates = len(dip[0])
    dse_block = numpy.zeros((nfrag*nstates, nfrag*nstates))
    for m, n in itertools.combinations(range(nfrag), 2):
        dse = numpy.einsum('ps,pt->st', dip[m].conj(), dip[n])
        dse_block[m*nstates:(m+1)*nstates, n*nstates:(n+1)*nstates] = dse
        dse_block[n*nstates:(n+1)*nstates, m*nstates:(m+1)*nstates] = dse.conj().T

    return dse_block

def get_ab_block(td_obj, cav_obj, has_offdiag=False):
    accum_nov = cav_obj.accum_nov
    a_block = numpy.zeros((accum_nov[-1], accum_nov[-1]))
    b_block = numpy.zeros((accum_nov[-1], accum_nov[-1]))

    nfrag = len(accum_nov)-1
    for n in range(nfrag):
        n0, n1 = accum_nov[n], accum_nov[n+1]
        a, b = td_obj[n].get_ab(td_obj[n]._scf)
        a_block[n0:n1, n0:n1] += a.reshape(n1-n0, n1-n0)
        b_block[n0:n1, n0:n1] += b.reshape(n1-n0, n1-n0)

    if has_offdiag:
        #mo_coeff = cav_obj.mo_coeff
        orbo, orbv = cav_obj.orbo, cav_obj.orbv
        #nbas, nocc = cav_obj.nbas, cav_obj.nocc
        elec_occupation = cav_obj.elec_occupation
        for m, n in itertools.combinations(range(nfrag), 2): # upper triangular
            a, b = get_inter_coupling_coulomb([td_obj[m].mol, td_obj[n].mol],
                                              [orbo[m], orbo[n]],
                                              [orbv[m], orbv[n]],
                                              elec_occupation[m])
            m0, m1 = accum_nov[m], accum_nov[m+1]
            n0, n1 = accum_nov[n], accum_nov[n+1]

            a_block[m0:m1, n0:n1] += a
            b_block[m0:m1, n0:n1] += b
            a_block[n0:n1, m0:m1] += a.conj().T
            b_block[n0:n1, m0:m1] += b.conj().T

    return a_block, b_block

def get_omega_block(cav_obj):
    return numpy.diag(cav_obj.cavity_freq).reshape(cav_obj.cavity_num, cav_obj.cavity_num)

def get_inter_coupling_coulomb(mols, orbo, orbv, occupation=2.):
    conc_mol = gto.mole.conc_mol
    mol = conc_mol(mols[0], mols[1])
    nbas = orbo[0].shape[0] # for first fragment

    #intor_cross = gto.mole.intor_cross
    #v = intor_cross('int2e_sph', td_obj[m].mol, td_obj[n].mol)
    # 2e integral between 1 and 2 fragments
    v2 = mol.intor('int2e')[:nbas, :nbas, nbas:, nbas:]

    mo_coeff = numpy.block([orbo[1], orbv[1]]) # for second fragment
    eri_mo = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', v2,
                        orbo[0].conj(), orbv[0], mo_coeff.conj(), mo_coeff)

    nbas, nocc = orbo[1].shape # for second fragment
    nov = nocc * (nbas-nocc) # for second fragment
    a = numpy.einsum('iabj->iajb', eri_mo[:,:,nocc:,:nocc]) * occupation
    b = numpy.einsum('iajb->iajb', eri_mo[:,:,:nocc,nocc:]) * occupation

    return a.reshape(-1, nov), b.reshape(-1, nov)

def get_inter_coupling_dipole(coms, dips):
    dr = coms[1] - coms[0]
    d1, d2 = dips

    d1dr, d2dr = numpy.einsum('nx,x->n', d1, dr), numpy.einsum('nx,x->n', d2, dr)

    dr = numpy.linalg.norm(dr)
    r3, r5 = dr**3, dr**5

    v = numpy.einsum('mx,nx->mn', d1, d2) / r3 - numpy.einsum('m,n->mn', d1dr,  d2dr)* (3. / r5)
    return v


class TDMixin(lib.StreamObject):
    conv_tol    = getattr(__config__, 'tdscf_rhf_TDA_conv_tol',  1e-9)
    nstates     = getattr(__config__, 'tdscf_rhf_TDA_nstates',      3)
    singlet     = getattr(__config__, 'tdscf_rhf_TDA_singlet',   None)
    lindep      = getattr(__config__, 'tdscf_rhf_TDA_lindep',   1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift',  0)
    max_space   = getattr(__config__, 'tdscf_rhf_TDA_max_space',  200)
    max_cycle   = getattr(__config__, 'tdscf_rhf_TDA_max_cycle',  500)

    def __init__(self, td_obj, cav_obj, key):
        if not isinstance(td_obj, list):
            td_obj = [td_obj]
        self.td_obj     = td_obj
        self.nfrag      = len(td_obj)

        self.cav_obj    = cav_obj

        self.verbose    = td_obj[0].verbose
        self.stdout     = td_obj[0].stdout
        self.max_memory = td_obj[0].max_memory
        self.chkfile    = td_obj[0].chkfile
        self.wfnsym     = None


        # xy[i] = (X_I,Y_I), In TDA, Y_I = 0
        # mn[i] = (M_I,N_I), In RWA, N_I = 0
        # Normalized to 1:
        # For restricted case:
        # 2(X_I X_I - Y_I Y_I) + (M_I M_I - N_I N_I) = 1
        # For unrestricted case:
        # (Xa_I Xa_I - Ya_I Ya_I) + (Xb_I Xb_I - Yb_I Yb_I) + (M_I M_I - N_I N_I) = 1

        for name, value in key.items(): # put all the variables in the class
            setattr(self, name, value)

        if getattr(self, 'has_offdiag', None) is None or self.nfrag == 1:
            self.has_offdiag = False
        print('has_offdiag inter-fragment coupling?', self.has_offdiag)

        if getattr(self, 'resonance_state', None) is None:
            energy = []
            for n in range(self.nfrag):
                energy.append(td_obj[n].e)
            energy = numpy.array(energy) - cav_obj.cavity_freq[0]
            self.resonance_state = numpy.argsort(numpy.sum(numpy.abs(energy), axis=0))[0] + 1
            print('resonance_state:', self.resonance_state)

        self.converged = None
        self.e         = None
        self.xy        = None
        self.mn        = None

        keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift', 'max_cycle'))
        #keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift', 'max_space', 'max_cycle'))
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
        e = list(map(lambda n: self.td_obj[n]._scf.e_tot, range(self.nfrag)))
        return numpy.sum(e) + self.e

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s from %s-%s-%s ********', self.__class__, self.td_obj.__class__, self.cav_obj.__class__, self._scf.__class__)
        log.info("cQED-TDDFT:          %s", "Yang, et. al. J. Chem. Phys. 155, 064107 (2021) https://doi.org/10.1063/5.0057542")
        log.info("Analytical Gradient: %s", "Yang, et. al. J. Chem. Phys. 156, 124104 (2022) https://doi.org/10.1063/5.0082386")

        self._nov = self.cav_obj.accum_nov[-1]

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
        for n in range(self.nfrag):
            if self.td_obj[n]._scf.mo_coeff is None:
                raise RuntimeError('SCF object is not initialized')
        lib.StreamObject.check_sanity(self)

    def reset(self, mol, index=0):
        if isinstance(mol, list):
            if index == -1:
                index = range(len(mol))
            for n in index:
                self.td_obj[n]._scf.reset(mol[n])
        else:
            self.td_obj[index]._scf.reset(mol)
        return self

    def get_ab_block(self, td_obj=None):
        if td_obj is None:
            td_obj = self.td_obj
        return get_ab_block(td_obj, self.cav_obj, self.has_offdiag)

    def get_g_block(self, cav_obj=None):
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_g_block(cav_obj)

    def get_dse_block(self, cav_obj=None, rpa=True, has_k=True):
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_dse_block(cav_obj, rpa, has_k)

    def get_omega_block(self, cav_obj=None):
        if cav_obj is None:
            cav_obj = self.cav_obj
        return get_omega_block(cav_obj)

    def gen_precond(self, hdiag):
        def precond(x, e, x0):
        #def precond(x, e, *args):
        #    if isinstance(e, numpy.ndarray):
        #        e = e[0]
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if not all(self.converged):
            logger.note(self, 'QED-TD-SCF states %s not converged.',
                        [i for i, x in enumerate(self.converged) if not x])
        logger.note(self, type(self.cav_obj).__name__+' Polariton State energies (au)\n%s', self.e)
        return self

    get_nto             = get_nto
    #_contract_multipole = _contract_multipole  # needed by following methods

    def transition_dipole(self, amps):
        xys, _ = self.get_elec_amps(numpy.copy(amps))
        # scale RHF amplitudes
        occupation = self.cav_obj.elec_occupation
        accum_nov  = self.cav_obj.accum_nov
        for n in numpy.where(occupation==2)[0]:
            n0, n1 = accum_nov[n], accum_nov[n+1]
            xys[:,n0:n1] *= numpy.sqrt(2.)

        dip_ov = self.cav_obj.dip_ov
        trans_dip = numpy.einsum('xl,pl->px', dip_ov, xys)
        #return trans_dip

        ipr = numpy.einsum('pl,pl,pl,pl->p',xys,xys,xys,xys)
        return trans_dip, ipr

    def transition_magnetic_dipole(self, amps):
        xys, _ = self.get_elec_amps(numpy.copy(amps), hermi=-1.)
        # scale RHF amplitudes
        occupation = self.cav_obj.elec_occupation
        accum_nov  = self.cav_obj.accum_nov
        for n in numpy.where(occupation==2)[0]:
            n0, n1 = accum_nov[n], accum_nov[n+1]
            xys[:,n0:n1] *= numpy.sqrt(2.)

        mag_dip_ov = self.cav_obj.mag_dip_ov
        # pure imagnary part
        m_dip = numpy.einsum('xl,pl->px', mag_dip_ov, xys)
        return -m_dip

    def oscillator_strength(self, e=None, trans_dip=None, gauge='length', order=0):
        if e is None: e = self.e
        if trans_dip is None: trans_dip = self.trans_dip

        if gauge == 'length':
            f = 2./3. * numpy.einsum('s,sx,sx->s', e, trans_dip.conj(), trans_dip)
            return f

    def rotation_strength(self, trans_dip=None, trans_mag_dip=None):
        if trans_dip is None: trans_dip = self.trans_dip
        if trans_mag_dip is None: trans_mag_dip = self.trans_mag_dip

        f = numpy.einsum('sx,sx->s', trans_dip.conj(), trans_mag_dip)
        return f

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

        # get default guess
        cav_x0 = self.cav_obj.init_guess(nstates=nstates)
        td_x0  = self.init_elec_guess(nstates=nstates-cav_x0.shape[1])

        x0 = numpy.block(
        [[td_x0,   numpy.zeros([td_x0.shape[0], cav_x0.shape[1]])],
         [numpy.zeros([cav_x0.shape[0], td_x0.shape[1]]),  cav_x0]]
        )

        if getattr(self, 'qed_obj0', None): # append polariton state
            s = self.resonance_state-1
            accum_nov = self.cav_obj.accum_nov
            nf = accum_nov[-1]
            x = numpy.zeros((1, nf+self.cav_obj.cavity_num))
            for n in range(self.nfrag):
                n0, n1 = accum_nov[n], accum_nov[n+1]
                xy = self.qed_obj0[n].xy[s][0]
                mn = self.qed_obj0[n].mn[s][0]
                if numpy.sign(xy[numpy.argmax(numpy.abs(xy))]) < 0.:
                    xy *= -1.
                if numpy.sign(mn[numpy.argmax(numpy.abs(mn))]) < 0.:
                    mn *= -1.
                x[0, n0:n1] = xy
                x[0, nf:] += mn
            x[0] = x[0] / numpy.linalg.norm(x[0])
            numpy.fill_diagonal(x0[td_x0.shape[0]:, nf:], 0.)
            x0 = numpy.concatenate((x, x0), axis=0)

        return x0

    def init_elec_guess(self, mf_obj, nstates=None):
        raise NotImplementedError

    def get_elec_amps(self, amps):
        raise NotImplementedError

    def get_norms2(self, xys):
        raise NotImplementedError

    def get_xys(self, amps):
        raise NotImplementedError

    def get_xys_weight(self, xys):
        raise NotImplementedError

    def kernel(self, amp0=None, nstates=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        from qed import grad
        return grad.Gradients(self)

class TDASym(TDMixin):
    def gen_eigen_solver(self):
        positive_eig_threshold = getattr(self, 'level_shift', POSTIVE_EIG_THRESHOLD**2)
        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx
        return davidson1, pickeig
        #return lr_eigh, pickeig

    def gen_elec_resp(self, td_obj=None, cav_obj=None):
        if td_obj is None: td_obj = self.td_obj
        if cav_obj is None: cav_obj = self.cav_obj

        nfrag        = self.nfrag
        cavity_freq2 = numpy.sqrt(cav_obj.cavity_freq/2.)
        accum_nov    = cav_obj.accum_nov
        dip_ov       = cav_obj.dip_scaled_ov.conj()

        amp_size  = accum_nov[-1]

        vind0, hdiag = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            #print('td_obj:', td_obj[n].singlet, td_obj[n].wfnsym)
            #td_obj[n].singlet = self.singlet
            #td_obj[n].wfnsym  = self.wfnsym
            vind0[n], hdiag[n] = td_obj[n].gen_vind()

        def vind(xs, ls, dse_resp=None):
            amp_num = xs.shape[0]
            axs     = numpy.zeros((amp_num, amp_size))
            for n in range(nfrag):
                n0, n1 = accum_nov[n], accum_nov[n+1]
                axs[:,n0:n1] += vind0[n](xs[:,n0:n1].reshape(amp_num, -1))
            tmp1    = numpy.einsum('lp,p->lp', ls, cavity_freq2)
            gls     = numpy.einsum('pn,lp->ln', dip_ov, tmp1)

            resp = axs + gls
            if isinstance(dse_resp, numpy.ndarray):
                resp += dse_resp[0]
            return resp

        return vind, numpy.asarray(hdiag)

    def get_elec_amps(self, amps, hermi=1.):
        xs = amps[:, :self.cav_obj.accum_nov[-1]] # pointer
        return xs, xs

    def get_xys(self, amps):
        _, xs2 = self.get_elec_amps(amps)
        xs = numpy.copy(xs2) # prevent changing amps

        # scale RHF amplitudes
        occupation = self.cav_obj.elec_occupation
        accum_nov  = self.cav_obj.accum_nov
        for n in numpy.where(occupation==2)[0]:
            n0, n1 = accum_nov[n], accum_nov[n+1]
            xs[:,n0:n1] *= numpy.sqrt(.5)
        return [(x, 0) for x in xs]

    def get_xys_weight(self, xys=None):
        if xys == None: xys = self.xy

        occupation = self.cav_obj.elec_occupation
        accum_nov = self.cav_obj.accum_nov

        weight = []
        for j in range(len(xys)):
            x, _ = xys[j]
            for n in range(self.nfrag):
                n0, n1 = accum_nov[n], accum_nov[n+1]
                weight.append(numpy.einsum('i,i->', x[n0:n1].conj(), x[n0:n1])*occupation[n])
        return numpy.reshape(weight, (len(xys), -1)).T

    def get_norms2(self, xys):
        amp_size = self.cav_obj.accum_nov[-1]
        xys      = xys.reshape(-1, amp_size)
        amp_num  = xys.shape[0]
        xs       = xys
        norms2   = numpy.einsum('li,li->l', xs.conj(), xs)
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

    def init_elec_guess(self, nstates=None):
        if nstates is None: nstates = self.nstates

        nfrag = self.nfrag
        s = 1 if (nstates>=nfrag and nfrag>1) else 0

        if self.td_obj[0].xy and s > 0: # normal td_obj has excited state eigenvectors
            s = self.resonance_state-1
            accum_nov = self.cav_obj.accum_nov
            td_x0 = numpy.zeros((nfrag, accum_nov[-1]))
            for n in range(nfrag):
                x0 = self.td_obj[n].xy[s][0]
                n0, n1 = accum_nov[n], accum_nov[n+1]
                td_x0[n,n0:n1] += x0.ravel()

            return td_x0

        else:
            return init_guess(self.cav_obj.e_ia, nstates=nstates,
                              resonance_state=(self.resonance_state-1)*nfrag)

    def kernel(self, amp0=None, nstates=None):
        td_obj  = self.td_obj
        cav_obj = self.cav_obj
        cpu0 = (logger.process_clock(), logger.perf_counter())
        #self.check_sanity()
        #self.dump_flags()
        if nstates is None:
            nstates      = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        if amp0 is None:
            amp0 = self.get_init_guess(nstates=nstates)

        vind, hdiag              = self.gen_vind(cav_obj=cav_obj)
        precond                  = self.gen_precond(hdiag)
        davidson_solver, pickeig = self.gen_eigen_solver()

        if getattr(self, 'target_states', 'electronic') == 'polariton':
            pickeig = None # temporally write in linalg.helper.py
            log.target_states = 'polariton'
            log.amp_size = cav_obj.amp_size

        self.converged, e, amps = davidson_solver(
                              vind, amp0, precond,
                              tol=self.conv_tol,
                              #tol_residual=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space,
                              max_memory=self.max_memory,
                              pick=pickeig, verbose=log)

        # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.e  = e[:nstates]
        amps    = amps[:nstates]
        nstates = self.e.size
        amps    = numpy.asarray(amps).reshape(nstates, -1)

        zs, xys     = self.get_elec_amps(amps)
        ls, mns     = cav_obj.get_amps(amps)
        norms2_elec = self.get_norms2(xys)
        norms2_ph   = cav_obj.get_norms2(mns)
        norms2      = (norms2_elec + norms2_ph).reshape(nstates)
        amps        = numpy.einsum('li,l->li', amps, 1./numpy.sqrt(norms2))

        if self.verbose > 3:
            for istate, xy in enumerate(xys):
                log.info("istate = %4d, norms2_elec = % 6.4f, norms2_ph = % 6.4f", istate, norms2_elec[istate], norms2_ph[istate])

        self.xy = self.get_xys(amps) # For alpha beta spin
        self.mn = cav_obj.get_mns(amps)

        self.trans_dip, self.ipr = self.transition_dipole(amps)
        self.trans_mag_dip = self.transition_magnetic_dipole(amps)

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('QED-TDDFT', *cpu0)
        self._finalize()
        #self.kernel0()
        return self.e, self.xy, self.mn

    def kernel0(self): # direct diagonalization
        a, b = self.get_ab_block()
        eigval, eigvec = numpy.linalg.eigh(a)
        print('tda eigval:\n', eigval)
        g = self.get_g_block()
        omega = self.get_omega_block()
        H = numpy.block([[a, g.conj().T], [g, omega]])
        eigval, eigvec = numpy.linalg.eigh(H)
        print('JC eigval:\n', eigval)
        #for i in range(len(eigval)):
        #    print(str(i+1)+' eigvec:\n', eigvec[:,i])
        weight = numpy.einsum('j,j->j', eigvec[-1:].conj(), eigvec[-1:])
        print('weight:\n', weight)

        z = numpy.zeros(omega.shape)
        H = numpy.block([[a, g.conj().T, g.conj().T], [g, omega, z], [-g, z, -omega]])
        eigval, eigvec = numpy.linalg.eig(H)
        idx = eigval.argsort()[(eigval<0).sum():]
        eigval, eigvec = eigval[idx], eigvec[:,idx]
        print('Rabi eigval:\n', eigval)
        #for i in range(len(eigval)):
        #    print(str(i+1)+' eigvec:\n', eigvec[:,i])
        weight = numpy.einsum('ij,ij->ij', eigvec[-2:].conj(), eigvec[-2:])
        weight = weight[0] - weight[1]
        print('weight:\n', weight)

class TDANoSym(TDASym):
    def gen_eigen_solver(self):
        # We only need positive eigenvalues
        positive_eig_threshold = getattr(self, 'level_shift', POSTIVE_EIG_THRESHOLD)
        ensure_real = self.td_obj[0]._scf.mo_coeff.dtype == numpy.double
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > positive_eig_threshold))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, ensure_real)
        return davidson_nosym1, pickeig
        #return lr_eig, pickeig

class RPA(TDANoSym):
    def gen_elec_resp(self, td_obj=None, cav_obj=None):
        if td_obj is None: td_obj = self.td_obj
        #assert isinstance(td_obj, pyscf.tdscf.rhf.RPA)
        if cav_obj is None: cav_obj = self.cav_obj

        nfrag        = self.nfrag
        cavity_freq2 = numpy.sqrt(cav_obj.cavity_freq/2.)
        accum_nov    = cav_obj.accum_nov
        dip_ov       = cav_obj.dip_scaled_ov.conj()

        amp_size  = accum_nov[-1]

        vind0, hdiag = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            #td_obj[n].singlet = self.singlet
            #td_obj[n].wfnsym  = self.wfnsym
            vind0[n], hdiag[n] = td_obj[n].gen_vind()

        def vind(xys, ls, dse_resp=None):
            #xys     = numpy.asarray(xys).reshape(-1, 2, nocc, nvir)
            amp_num = xys.shape[0]
            abxys   = numpy.zeros((amp_num, 2, amp_size))
            for n in range(nfrag):
                n0, n1 = accum_nov[n], accum_nov[n+1]
                _abxy = vind0[n](xys[:,:,n0:n1])
                abxys[:, :, n0:n1] += _abxy.reshape(amp_num, 2, -1)

            abxys1, abxys2 = abxys.transpose(1,0,2)
            tmp1 = numpy.einsum('lp,p->lp', ls, cavity_freq2)
            gls  = numpy.einsum('pn,lp->ln', dip_ov, tmp1)

            abxys1 = (abxys1 + gls)
            abxys2 = (abxys2 - gls)

            if isinstance(dse_resp, numpy.ndarray):
                abxys1 += dse_resp[0]
                abxys2 -= dse_resp[1]
            return numpy.hstack([abxys1, abxys2])#.reshape(amp_num, -1)

        return vind, numpy.asarray(hdiag)

    def get_elec_amps(self, amps, hermi=1.):
        amp_size = self.cav_obj.accum_nov[-1]
        xys = amps[:, :2*amp_size].reshape(-1,2,amp_size) # pointer
        xs, ys = xys.transpose(1,0,2)
        return xs + hermi* ys, xys

    def get_xys(self, amps):
        _, xys2 = self.get_elec_amps(amps)
        xys = numpy.copy(xys2) # prevent changing amps

        # scale RHF amplitudes
        occupation = self.cav_obj.elec_occupation
        accum_nov  = self.cav_obj.accum_nov
        for n in numpy.where(occupation==2)[0]:
            n0, n1 = accum_nov[n], accum_nov[n+1]
            xys[:,:,n0:n1] *= numpy.sqrt(.5)

        return [(x, y) for x, y in xys]

    def get_xys_weight(self, xys=None):#, fac=1.0):
        if xys == None: xys = self.xy
        xys = numpy.asarray(xys)

        occupation = self.cav_obj.elec_occupation
        accum_nov = self.cav_obj.accum_nov

        weight = numpy.zeros((self.nfrag, xys.shape[0]))
        for n in range(self.nfrag):
            n0, n1 = accum_nov[n], accum_nov[n+1]
            x, y = xys[:,0,n0:n1], xys[:,1,n0:n1]
            w = numpy.einsum('ki,ki->k', x.conj(), x) - numpy.einsum('ki,ki->k', y.conj(), y)
            weight[n] = w*occupation[n]
        #return numpy.reshape(weight, (len(xys), -1)).T
        return weight

    def get_norms2(self, xys):
        amp_size = self.cav_obj.accum_nov[-1]
        xys      = xys.reshape(-1, 2*amp_size)
        amp_num  = xys.shape[0]
        xs       = xys[:, :amp_size]
        ys       = xys[:, amp_size:]
        norms2   = (numpy.einsum('li,li->l', xs.conj(), xs) - numpy.einsum('li,li->l', ys.conj(), ys))
        if amp_num == 1:
            return norms2[0]
        else:
            return norms2

    def init_elec_guess(self, nstates=None):
        if nstates is None: nstates = self.nstates

        nfrag = self.nfrag
        s = 1 if (nstates>=nfrag and nfrag>1) else 0

        if self.td_obj[0].xy and s > 0: # normal td_obj has excited state eigenvectors
            s = self.resonance_state-1
            accum_nov = self.cav_obj.accum_nov
            td_x0 = numpy.zeros((nfrag, 2, accum_nov[-1]))
            for n in range(nfrag):
                x0, y0 = self.td_obj[n].xy[s]
                n0, n1 = accum_nov[n], accum_nov[n+1]
                td_x0[n,0,n0:n1] = x0.ravel()
                td_x0[n,1,n0:n1] = y0.ravel()

            return td_x0.reshape(nfrag, -1)

        else:
            x0 = init_guess(self.cav_obj.e_ia, nstates=nstates,
                            resonance_state=(self.resonance_state-1)*nfrag)
            y0 = numpy.zeros_like(x0)
            return numpy.hstack((x0, y0))


def few_level_matrix(td_obj, cav_obj, has_dse, has_k=True, nstates=None,
                     save_amplitude=False, has_offdiag=False):
    nfrag = len(td_obj)
    if nstates is None or nstates > len(td_obj[0].e):
        nstates = len(td_obj[0].e)

    accum_nov = cav_obj.accum_nov
    cav_obj.init_guess() # get amp_size dimension

    nd1  = nfrag * nstates
    nd2  = nd1 + cav_obj.cavity_num
    ndim = nd1 + cav_obj.amp_size

    matrix = numpy.zeros((ndim, ndim))
    trans_dip = numpy.zeros((nd1, 3)) # for transition dipole calculations
    mag_dip = numpy.zeros((nd1, 3)) # for transition magnetic dipole calculations

    e = []
    for n in range(nfrag):
        e.append(td_obj[n].e[:nstates])
    e_ph = cav_obj.cavity_freq.ravel()
    e = numpy.concatenate((numpy.reshape(e, -1), e_ph))
    if ndim > nd2:
        e = numpy.concatenate((e, -e_ph))
    numpy.fill_diagonal(matrix, e)

    g_block = get_g_block(cav_obj)
    if has_dse:
        #dse_block = get_dse_block(cav_obj)
        dip_ov = cav_obj.dip_scaled_ov
        dip_oo, dip_vv = cav_obj.dip_scaled_oo, cav_obj.dip_scaled_vv

    amplitude = [] #(nframe, nstate, nov)
    if has_offdiag: save_amplitude = True

    occupation = cav_obj.elec_occupation
    for n in range(nfrag):
        xs, xs2 = [], []
        for i in range(nstates):
            x, y = td_obj[n].xy[i]
            xs.append((x+y)*numpy.sqrt(occupation[n])) # alpha and beta electrons
            xs2.append((x-y)*numpy.sqrt(occupation[n])) # alpha and beta electrons

        n0, n1 = accum_nov[n], accum_nov[n+1]
        xs = numpy.reshape(xs, (-1, n1-n0))
        xs2 = numpy.reshape(xs2, (-1, n1-n0))
        if save_amplitude: amplitude.append(xs)

        trans_dip[n*nstates:(n+1)*nstates] += numpy.einsum('sn,xn->sx', xs, cav_obj.dip_ov[:,n0:n1])*numpy.sqrt(occupation[n])
        mag_dip[n*nstates:(n+1)*nstates] -= numpy.einsum('sn,xn->sx', xs2, cav_obj.mag_dip_ov[:,n0:n1])*numpy.sqrt(occupation[n])

        g = lib.einsum('sn,pn->sp', xs, g_block[:,n0:n1])
        matrix[n*nstates:(n+1)*nstates, nd1:nd2] += g.conj()
        matrix[nd1:nd2, n*nstates:(n+1)*nstates] += g.T
        if ndim > nd2:
            matrix[n*nstates:(n+1)*nstates, nd2:] += g.conj()
            matrix[nd2:, n*nstates:(n+1)*nstates] -= g.T

        if has_dse:
            #d = lib.einsum('sm,mn,tn->st', xs, dse_block[n0:n1,n0:n1], xs)
            d = get_dse_block2(xs, dip_ov[:,n0:n1], dip_oo[n], dip_vv[n],
                               occupation[n], has_k=has_k)
            matrix[n*nstates:(n+1)*nstates, n*nstates:(n+1)*nstates] += d

    amplitude = numpy.asarray(amplitude) # convert list to array for safety

    if has_offdiag:
        orbo, orbv = cav_obj.orbo, cav_obj.orbv
        elec_occupation = cav_obj.elec_occupation

        coords, mass = [], []
        for n in range(nfrag):
            mol = td_obj[n]._scf.mol
            coords.append(mol.atom_coords())
            mass.append(mol.atom_mass_list())
        coms = numpy.einsum('ni,nix,n->nx', mass, coords, 1./numpy.sum(mass, axis=1))

        for m, n in itertools.combinations(range(nfrag), 2): # upper triangular
            a = get_inter_coupling_dipole([coms[m], coms[n]],
                                          [trans_dip[m*nstates:(m+1)*nstates],
                                           trans_dip[n*nstates:(n+1)*nstates]])
            matrix[m*nstates:(m+1)*nstates, n*nstates:(n+1)*nstates] += a
            matrix[n*nstates:(n+1)*nstates, m*nstates:(m+1)*nstates] += a.conj().T

        if has_dse:
            matrix[:nd1, :nd1] += get_dse_block_off(amplitude, dip_ov, accum_nov)

    return matrix, trans_dip, mag_dip, amplitude


class FewLevel(TDMixin):
    def kernel(self, td_obj=None, cav_obj=None, nstates=None):
        if td_obj is None:
            td_obj = self.td_obj
        if cav_obj is None:
            cav_obj = self.cav_obj

        cavity_model = self.cavity_model.upper()
        self.ng = 1 if (cavity_model == 'JC' or cavity_model == 'RWA') else 2
        has_dse = True if (cavity_model == 'RWA' or cavity_model == 'PF') else False
        has_k = cav_obj.has_k
        save_amplitude = getattr(self, 'save_amplitude', False)
        has_offdiag = getattr(self, 'has_offdiag', None)
        print('has_offdiag inter-fragment coupling?', has_offdiag)

        # this nstates is provided exciton state numbers for model hamiltonian
        matrix, trans_dip, mag_dip, amplitude = few_level_matrix(td_obj, cav_obj, has_dse, has_k, nstates, save_amplitude, has_offdiag)
        #print_matrix('matrix', matrix, 10)

        if self.nstates > matrix.shape[0]:
            self.nstates = matrix.shape[0] - (self.ng-1)

        e, v = eigen_solver(matrix, self.nstates, self.target_states,
                            self.solver_conv_prop, self.level_shift, self.max_cycle,
                            self.tolerance, self.ng, self.solver_algorithm)

        #print_matrix(cavity_model+' polariton energy', e, 10)
        #print_matrix('v', v[:-self.ng], 10)
        #print_matrix('trans_dip:', trans_dip, 10)
        trans_dip = numpy.einsum('ni,nx->ix', v[:-self.ng], trans_dip)
        mag_dip = numpy.einsum('ni,nx->ix', v[:-self.ng], mag_dip)

        if save_amplitude:
            v0 = numpy.copy(v[-self.ng:]).T
            v = numpy.reshape(v[:-self.ng], (self.nfrag, -1, self.nstates))
            amplitude = numpy.einsum('nmi,nmk->kni', amplitude, v).reshape(self.nstates, -1)
            v = numpy.concatenate((amplitude, v0), axis=1)

        return e, v, trans_dip, mag_dip

    def get_weights(self, vector):
        ng = self.ng
        weight_p = vector[-ng:]
        weight_p = numpy.einsum('ij,ij->ij', weight_p.conj(), weight_p)
        if ng == 1: weight_p = weight_p[0]
        else: weight_p = weight_p[0] - weight_p[1]
        #print('photon character total:', numpy.sum(weight_p))

        weight_e = numpy.reshape(vector[:-ng], (self.nfrag, -1, vector.shape[1]))
        weight_e = numpy.einsum('nik,nik->nk', weight_e.conj(), weight_e)

        return weight_p, weight_e


def eigen_solver(matrix, nroots, target_states, conv_prop, level_shift, max_cycle, tol,
                 method, solver_algorithm):
    if 'davidson' in solver_algorithm:
        e, v = diagonalize_matrix_davidson(matrix, nroots, target_states,
                                           conv_prop,
                                           level_shift, max_cycle, tol, method,
                                           solver_algorithm[-2:])
    else:
        e, v = diagonalize_matrix(matrix, method)
        e, v = e[:nroots], v[:, :nroots] # get the required numbers of states

    return e, v

def diagonalize_matrix(mat, imethod=1):
    # eigenvectors in Fortran's order
    if imethod == 1:
        w, v = numpy.linalg.eigh(mat) # symmetric matrix
    else:
        w, v = numpy.linalg.eig(mat) # general matrix
        ind = w.argsort()
        ind = ind[(w<0).sum():] # sort engenvalues and pick positive ones

        w, v = w[ind], v[:, ind]
        #print_matrix('w', w, 10)

    return w, v

def diagonalize_matrix_davidson(mat, nroots, target_states, conv_prop,
                                level_shift=1e-2, max_cycle=50, tol=1e-8, method=2,
                                solver='qr'):
    D = numpy.diag(mat)

    def initial_guess(D, n, level_shift=1e-2, offset=0):
        t = numpy.zeros((D.shape[0], n*2)) # double initial space

        offset = len(numpy.where(D<level_shift)[0])
        print('initial_guess offset:', offset)
        arg = numpy.argsort(D)[offset:n+offset]
        print('initial_guess arg:', arg)
        for i, j in enumerate(arg):
            t[j,i] = 1.0

        return t

    t = initial_guess(D, nroots, level_shift)
    n, k = t.shape

    theta_old = numpy.zeros(nroots)
    left = nroots
    V = t

    for m in range(max_cycle):
        if solver == 'qr':
            V, R = numpy.linalg.qr(V) # QR decomposition where R is not used

        T = numpy.einsum('mn,nj->mj', mat, V)
        T = numpy.einsum('mi,mj->ij', V, T) # Rayleigh matrix
        theta, vector = numpy.linalg.eigh(T)
        #print_matrix('T', T, 10)

        if target_states == 'polariton':
            vector = numpy.einsum('mi,ij->mj', V.conj(), vector) # Ritz vector
            arg = numpy.argsort(-numpy.abs(vector[-1]))[:nroots]
            theta, vector = theta[arg], vector[:, arg]
        else:
            arg = numpy.where(theta > level_shift)[0]
            arg = arg[:nroots]
            theta, vector = theta[arg], vector[:, arg]
            vector = numpy.einsum('mi,ij->mj', V.conj(), vector) # Ritz vector

        R = numpy.einsum('mn,nj->mj', mat, vector) - numpy.einsum('j,mj->mj', theta, vector) # residuals
        norm = numpy.linalg.norm(R, axis=0)
        index = numpy.where(norm >= tol)[0]
        left = index.size

        if solver == 'qr':
            for i, j in enumerate(index):
                C = numpy.nan_to_num(1.0/(theta[j] - D)) # Jacobi preconditioner
                R[:, j] *= C
            V = numpy.concatenate((V, R), axis=1)
        else: # GS
            for i, j in enumerate(index):
                C = numpy.nan_to_num(1.0/(theta[j] - D)) # Jacobi preconditioner
                v = R[:, j] * C
                q = numpy.einsum('mi,m->i', V, v) # Gram-Schmidt orthogonalization
                v -= numpy.einsum('i,mi->m', q, V)
                dot = numpy.linalg.norm(v)
                if dot > 1e-8: # 1e-4 is sufficient
                    V = numpy.concatenate((V, (v/dot)[:,None]), axis=1)
                else:
                    left -= 1

        #check energy convergence, if using energy as the criteria
        if conv_prop == 'energy':
           diff = theta - theta_old
           index = numpy.where(abs(diff) > 1e-6)[0]
           left_e = index.size
           if left_e == 0: left = 0
           theta_old = numpy.copy(theta)

        print('m:', m+1, 'left:', left, 'norm:', numpy.sum(norm))
        if left == 0:
            break

    return theta, vector

