from functools import reduce
import numpy
from pyscf      import lib
from pyscf.lib  import logger
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.scf  import cphf
from pyscf      import __config__

import qed
from qed.cavity.rhf import Rabi

def _qed_tdrhf_elec_grad(qed_td, x_y, m_n, singlet=True, atmlst=None,
                                 max_memory=2000,        verbose=logger.INFO):
    cav_obj = qed_td.cav_obj
    td_obj  = qed_td.td_obj
    td_grad = td_obj.nuc_grad_method()

    with_dse = not isinstance(cav_obj, qed.cavity.rhf.Rabi)
    assert singlet

    log   = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol       = td_grad.mol
    offsetdic = mol.offset_nr_by_atom()

    mf        = td_grad.base._scf
    mo_coeff  = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ    = mf.mo_occ
    occ_idx   = (mo_occ == 2)
    vir_idx   = (mo_occ == 0)

    if atmlst is None:
        atmlst = range(mol.natm)
    natm      = len(atmlst)
    offsetdic = mol.offset_nr_by_atom()

    nao, nmo  = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc

    orbv = mo_coeff[:,vir_idx]
    orbo = mo_coeff[:,occ_idx]

    ncav      = cav_obj.cavity_num
    cav_mode  = cav_obj.cavity_mode
    cav_mode  = cav_mode.reshape(3, ncav)
    cav_freq  = cav_obj.cavity_freq
    cav_freq  = cav_freq.reshape(ncav, )

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad     = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    
    dip_ao_grad = numpy.zeros([natm, 3, 3, nao, nao])
    irp_ao      = mol.intor("int1e_irp", comp=9, hermi=0).reshape(3,3,nao,nao)
    
    # this should have been done in a smart way
    for iatm in range(0, natm):
        ishl0, ishl1, ip0, ip1 = offsetdic[iatm]
        for katm in range(0, natm):
            kshl0, kshl1, kp0, kp1 = offsetdic[katm]
            for q in range(3):
                for y in range(3):
                    dip_ao_grad[iatm, q, y, kp0:kp1, ip0:ip1] -= irp_ao[y, q, kp0:kp1, ip0:ip1]
                    dip_ao_grad[iatm, q, y, ip0:ip1, kp0:kp1] -= irp_ao[y, q, kp0:kp1, ip0:ip1].T

    dip_ao      = mol.intor("int1e_r", comp=3)
    dip_ao      = dip_ao.reshape(3, nao, nao) # have the indices of `xmn`

    # This should be done in a smart way
    ovlp_deriv = numpy.zeros([natm, 3, nao, nao])
    ipovlp     = mol.intor("int1e_ipovlp", comp=3)
    ipovlp     = ipovlp.reshape(3, nao, nao)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ovlp_deriv[k, :, p0:p1, :] += ipovlp[:, p0:p1, :]  
        ovlp_deriv[k, :, :, p0:p1] += ipovlp[:, p0:p1, :].transpose(0,2,1)

    fock_ao = mf.get_fock()
    dm0     = reduce(numpy.dot, (orbo, orbo.T))
    cct     = reduce(numpy.dot, (mo_coeff, mo_coeff.T))

    x, y = x_y

    xpy = (x + y).reshape(nocc,nvir).T
    xmy = (x - y).reshape(nocc,nvir).T

    x = (xpy + xmy)/2
    y = (xpy - xmy)/2

    m, n = m_n

    mpn = (m + n).reshape(ncav,)
    mmn = (m - n).reshape(ncav,)

    m = (mpn + mmn)/2
    n = (mpn - mmn)/2

    assert abs(2*numpy.einsum("ai,ai->", x, x) - 2*numpy.einsum("ai,ai->", y, y) + numpy.einsum("c,c->", m, m) - numpy.einsum("c,c->", n, n) - 1) < 1e-8

    tmp        = numpy.einsum("xc,c->xc", cav_mode, numpy.sqrt(cav_freq/2))
    g_ao_grad  = numpy.einsum("kqxmn,xc->kqcmn", dip_ao_grad, tmp)
    gm_ao_grad = numpy.einsum("kqcmn,c->kqmn", g_ao_grad,     mpn)

    g_ao       = numpy.einsum("xmn,xc->cmn", dip_ao, tmp)
    gm_ao      = numpy.einsum("cmn,c->mn",   g_ao,   mpn)

    tmp         = numpy.einsum('xmn,xc->cmn', dip_ao, cav_mode)
    dse_ao      = numpy.einsum('cmn,clu->mnlu', tmp, tmp)
    dse_ao_grad = numpy.einsum('kqxmn,xc->kqcmn', dip_ao_grad, cav_mode)
    dse_ao_grad = 2 * numpy.einsum('kqcmn,clu->kqmnlu', dse_ao_grad, tmp)

    diff_den_oo     = - numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    diff_den_vv     =   numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)

    tran_den_xpy_ao  = reduce(numpy.dot, (orbv, xpy, orbo.T))
    tran_den_xmy_ao  = reduce(numpy.dot, (orbv, xmy, orbo.T))
    diff_den_ao  = reduce(numpy.dot, (orbo, diff_den_oo, orbo.T))
    diff_den_ao += reduce(numpy.dot, (orbv, diff_den_vv, orbv.T))

    dms = (diff_den_ao, tran_den_xpy_ao+tran_den_xpy_ao.T, tran_den_xmy_ao-tran_den_xmy_ao.T)

    vj, vk = mf.get_jk(mol, dms, hermi=0)
    veff_diff_den_ao  = vj[0] * 2 - vk[0]
    veff_trans_xpy_ao = vj[1] * 2 - vk[1]
    veff_trans_xmy_ao = - vk[2]

    assert singlet
    
    # Build RHS of CPHF equation
    lag_vo            = reduce(numpy.dot, (orbv.T, veff_diff_den_ao, orbo)) * 2

    veff_trans_xpy_mo = reduce(numpy.dot, (mo_coeff.T, veff_trans_xpy_ao, mo_coeff))
    veff_trans_xpy_oo = veff_trans_xpy_mo[occ_idx, :][:, occ_idx]
    veff_trans_xpy_vv = veff_trans_xpy_mo[vir_idx, :][:, vir_idx]
    lag_vo -= numpy.einsum('ki,ai->ak', veff_trans_xpy_oo, xpy) * 2
    lag_vo += numpy.einsum('ac,ai->ci', veff_trans_xpy_vv, xpy) * 2

    veff_trans_xmy_mo = reduce(numpy.dot, (mo_coeff.T, veff_trans_xmy_ao, mo_coeff))
    veff_trans_xmy_oo = veff_trans_xmy_mo[occ_idx, :][:, occ_idx]
    veff_trans_xmy_vv = veff_trans_xmy_mo[vir_idx, :][:, vir_idx]
    lag_vo -= numpy.einsum('ki,ai->ak', veff_trans_xmy_oo, xmy) * 2
    lag_vo += numpy.einsum('ac,ai->ci', veff_trans_xmy_vv, xmy) * 2

    gm_mo   = reduce(numpy.dot, (mo_coeff.T, gm_ao, mo_coeff))
    lag_vo -= 4 * numpy.einsum('ij,aj->ai', gm_mo[occ_idx, :][:, occ_idx], xpy)  # oo
    lag_vo += 4 * numpy.einsum('ba,ai->bi', gm_mo[vir_idx, :][:, vir_idx], xpy)  # vv

    if with_dse:
        dse_xpy_ao = numpy.einsum('mnlu,ul->mn', dse_ao, tran_den_xpy_ao+tran_den_xpy_ao.T)
        dse_xpy_mo = reduce(numpy.dot, (mo_coeff.T, dse_xpy_ao, mo_coeff))
        dse_xpy_oo = dse_xpy_mo[occ_idx, :][:, occ_idx]
        dse_xpy_vv = dse_xpy_mo[vir_idx, :][:, vir_idx]

        lag_vo -= numpy.einsum('ki,ai->ak', dse_xpy_oo, xpy) * 2
        lag_vo += numpy.einsum('ac,ai->ci', dse_xpy_vv, xpy) * 2

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm   = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()

    zvo = cphf.solve(fvind, mo_energy, mo_occ, lag_vo,
                     max_cycle=td_grad.cphf_max_cycle,
                     tol=td_grad.cphf_conv_tol)[0]
    zvo = zvo.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    dm_z_ao  = reduce(numpy.dot, (orbv, zvo, orbo.T))
    diff_den_relaxed  = dm_z_ao + diff_den_ao

    veff_z_ao = vresp(dm_z_ao+dm_z_ao.T)

    fock_dot_diff_den      = numpy.dot(fock_ao, (diff_den_relaxed+diff_den_relaxed.T)/4+dm0)
    veff_diff_den_relaxed      = veff_z_ao + veff_diff_den_ao
    dm0_dot_veff_diff_ao       = numpy.dot(veff_diff_den_relaxed, dm0)/2

    lam_ao  = numpy.dot(veff_trans_xpy_ao, tran_den_xpy_ao+tran_den_xpy_ao.T)/2
    lam_ao -= numpy.dot(veff_trans_xmy_ao, tran_den_xmy_ao-tran_den_xmy_ao.T)/2
    lam_ao += dm0_dot_veff_diff_ao
    lam_ao += fock_dot_diff_den

    wao1    = reduce(numpy.dot, [cct, lam_ao])
    
    gm_dot_x_ao = numpy.dot(gm_ao, tran_den_xpy_ao + tran_den_xpy_ao.T)
    wao2        = reduce(numpy.dot, [cct, gm_dot_x_ao])

    if with_dse:
        x_dse_x_ao  = numpy.dot(dse_xpy_ao, tran_den_xpy_ao+tran_den_xpy_ao.T)/2
        wao3        = reduce(numpy.dot, [cct, x_dse_x_ao])

    dms = (dm0, diff_den_relaxed+diff_den_relaxed.T, tran_den_xpy_ao+tran_den_xpy_ao.T, tran_den_xmy_ao-tran_den_xmy_ao.T)

    vj, vk = td_grad.get_jk(mol, dms)
    vj = vj.reshape(-1,3,nao,nao)
    vk = vk.reshape(-1,3,nao,nao)
    vhf1 = vj * 2 - vk

    time1     = log.timer('2e AO integral derivatives', *time1)    
    de        = numpy.zeros((natm,3)) # the total gradient
    de_elec   = numpy.zeros((natm,3)) # contains the additional z-vector contribution
    de_dipx   = numpy.zeros((natm,3)) # the direct dipole derivative contribution
    de_dsex   = numpy.zeros((natm,3))
    de_sx     = numpy.zeros((natm,3)) # the crossing term overlap derivative contribution

    if with_dse:
        dse_grad_xpy = numpy.einsum("kqmnlu,ul->kqmn", dse_ao_grad, tran_den_xpy_ao+tran_den_xpy_ao.T)
        de_dsex     += numpy.einsum("kqmn,nm->kq", dse_grad_xpy, tran_den_xpy_ao+tran_den_xpy_ao.T)

    de_dipx  += 4 * numpy.einsum("mn,kqmn->kq", tran_den_xpy_ao, gm_ao_grad)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += vhf1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += vhf1[0,:,p0:p1].transpose(0,2,1)
        
        # oo0*2 for doubly occupied orbitals
        de_elec[k]  = numpy.einsum('xpq,pq->x', h1ao, dm0) * 2
        de_elec[k] += numpy.einsum('xpq,pq->x', h1ao, diff_den_relaxed)

        de_elec[k] += numpy.einsum('xpq,pq->x', ovlp_deriv[ia, :, :, :], wao1)*2
        de_sx[k]   += numpy.einsum('xpq,pq->x', ovlp_deriv[ia, :, :, :], wao2)*2
        if with_dse:
            de_sx[k]   += numpy.einsum('xpq,pq->x', ovlp_deriv[ia, :, :, :], wao3)*2
        
        de_elec[k] += numpy.einsum('xij,ij->x', vhf1[1,:,p0:p1], dm0[p0:p1])
        de_elec[k] += numpy.einsum('xij,ij->x', vhf1[2,:,p0:p1], tran_den_xpy_ao[p0:p1,:]) * 2
        de_elec[k] += numpy.einsum('xij,ij->x', vhf1[3,:,p0:p1], tran_den_xmy_ao[p0:p1,:]) * 2
        de_elec[k] += numpy.einsum('xji,ij->x', vhf1[2,:,p0:p1], tran_den_xpy_ao[:,p0:p1]) * 2
        de_elec[k] -= numpy.einsum('xji,ij->x', vhf1[3,:,p0:p1], tran_den_xmy_ao[:,p0:p1]) * 2
        # print_rec(de[k, :].reshape(1,3), title="de[k] +179")

    de = de_elec + de_sx + de_dipx
    return de

def grad_elec(td_grad, xy, mn, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of QED-CIS, QED-TDHF nuclear gradients
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    td      = td_grad.base
    td_obj  = td.td_obj
    cav_obj = td.cav_obj
    td_grad = td_obj.nuc_grad_method()

    with_dse = not isinstance(cav_obj, qed.cavity.rhf.Rabi)
    assert singlet

    log.debug("cav_obj = %s", cav_obj.__class__)
    if with_dse:
        log.debug("QED-TDRHF Gradient With DSE\n")
    else:
        log.debug("QED-TDRHF Gradient No DSE\n")
    
    tmp = _qed_tdrhf_elec_grad(td, xy, mn, singlet=singlet, atmlst=atmlst, max_memory=max_memory, verbose=verbose)
    
    log.timer('TDHF nuclear gradients', *time0)
    return tmp

class Gradients(tdrhf_grad.Gradients):

    cphf_max_cycle = getattr(__config__, 'grad_tdrhf_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol  = getattr(__config__, 'grad_tdrhf_Gradients_cphf_conv_tol', 1e-8)

    def __init__(self, td):
        assert isinstance(td, qed.tdscf.rhf.TDMixin)
        self.verbose    = td.verbose
        self.stdout     = td.stdout
        self.mol        = td.mol
        self.base       = td
        self.chkfile    = td.chkfile
        self.max_memory = td.max_memory
        self.state      = 1  # of which the gradients to be computed.
        self.atmlst     = None
        self.de         = None
        keys = set(('cphf_max_cycle', 'cphf_conv_tol'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s-%s ********', self.base.td_obj.__class__, self.base.cav_obj.__class__)
        log.info('cphf_conv_tol  = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('chkfile        = %s', self.chkfile)
        log.info('State ID       = %d', self.state)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('\n')
        return self

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, mn, singlet, atmlst=None):
        return grad_elec(self, xy, mn, singlet, atmlst, self.max_memory, self.verbose)

    def kernel(self, xy=None, state=None, singlet=None, atmlst=None):
        '''
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        '''
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(self, 'state=0 found in the input. '
                            'Gradients of ground state is computed.')
                return self.base._scf.nuc_grad_method().kernel(atmlst=atmlst)

            xy = self.base.xy[state-1]
            mn = self.base.mn[state-1]

        if singlet is None: singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, mn, singlet, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    # Calling the underlying SCF nuclear gradients because it may be modified
    # by external modules (e.g. QM/MM, solvent)
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '------------ QED-TDHF gradients for state %d ------------', self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '---------------------------------------------------------')
