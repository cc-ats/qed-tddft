from functools import reduce
import numpy
from pyscf import lib, scf, dft
from pyscf.lib import logger
from pyscf.grad.tdrks import _contract_xc_kernel
from pyscf.grad import tdrks as tdrks_grad
from pyscf.scf import cphf
from pyscf import __config__

import qed
from qed.grad.tdrhf import as_scanner

def _qed_tdrks_elec_grad(qed_td, x_y, m_n, singlet=True, atmlst=None,
                         with_dse=True, max_memory=2000, verbose=logger.INFO):
    td_obj  = qed_td.td_obj
    cav_obj = qed_td.cav_obj
    td_grad = td_obj.nuc_grad_method()

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
    cav_mode  = cav_obj.cavity_mode.reshape(3, ncav)
    cav_freq  = cav_obj.cavity_freq.reshape(ncav, )
 
    # dipole and dipole derivatives AO matrices
    dip_ao = mol.intor("int1e_r", comp=3).reshape(3, nao, nao)
    irp_ao      = mol.intor("int1e_irp", comp=9, hermi=0).reshape(3,3,nao,nao)
    

    x, y = x_y
    xpy = (x + y).reshape(nocc,nvir).T
    xmy = (x - y).reshape(nocc,nvir).T

    tran_den_xpy_ao  = reduce(numpy.dot, (orbv, xpy, orbo.T))
    tran_den_xpy_ao += tran_den_xpy_ao.T
    tran_den_xmy_ao  = reduce(numpy.dot, (orbv, xmy, orbo.T))
    tran_den_xmy_ao -= tran_den_xmy_ao.T

    diff_den_oo      = - numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    diff_den_vv      =   numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    diff_den_ao      = reduce(numpy.dot, (orbo, diff_den_oo, orbo.T))
    diff_den_ao     += reduce(numpy.dot, (orbv, diff_den_vv, orbv.T))

    m, n = m_n
    mpn = (m + n).reshape(ncav,)
    mmn = (m - n).reshape(ncav,)
    mpn_scaled = numpy.einsum("xc,c,c->x", cav_mode, mpn, numpy.sqrt(cav_freq/2))

    if with_dse:
        dip_amp = numpy.einsum('ylu,ul->y', dip_ao, tran_den_xpy_ao)
        dip_amp = numpy.einsum('y,yc,xc->x', dip_amp, cav_mode, cav_mode)
        mpn_scaled += dip_amp # here add dse contribution to the off-diagnoal dipole


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)

    # start to form Lagrangian

    # figure out DFT type
    dft_type = dft.libxc.parse_xc(mf.xc)
    #ni = mf._numint
    #ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    has_xc = True if dft_type[1]!=[] else False
    log.info('hyb: ' + str(hyb) + ' dft: ' + str(has_xc))

    # two-electron terms
    if abs(hyb) > 1e-10:
        dms    = (diff_den_ao, tran_den_xpy_ao, tran_den_xmy_ao)
        vj, vk = mf.get_jk(mol, dms, hermi=0)
        vk    *= hyb
        if abs(omega) > 1e-10:
            vk += mf.get_k(mol, dms, hermi=0, omega=omega) * (alpha-hyb)

        veff_diff_den_ao  = vj[0] * 2 - vk[0]
        # only has singlet here
        veff_trans_xpy_ao = vj[1] * 2 - vk[1]
        veff_trans_xmy_ao = - vk[2]
    
    else:
        vj = mf.get_j(mol, (diff_den_ao, tran_den_xpy_ao), hermi=1)

        veff_diff_den_ao  = vj[0] * 2
        veff_trans_xpy_ao = vj[1] * 2

    if has_xc:
        # dm0 = mf.make_rdm1(mo_coeff, mo_occ), but it is not used when computing
        # fxc since rho0 is passed to fxc function.
        rho0, vxc, fxc = mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
        # in the orginial code, only xpy is used
        # since its transpose is added before, I scale it by 1/2 here
        f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, mf.xc, tran_den_xpy_ao*0.5, diff_den_ao, True, True, singlet, max_memory)

        veff_diff_den_ao += f1oo[0] + k1ao[0] * 2
        veff_trans_xpy_ao += f1vo[0] * 2

    # dipole contributions 
    gm_ao = numpy.einsum("xmn,x->mn",   dip_ao, mpn_scaled)
    veff_trans_xpy_ao += gm_ao * 2

    # Build RHS of CPHF equation
    lag_vo  = reduce(numpy.dot, (orbv.T, veff_diff_den_ao, orbo))
    veff_trans_xpy_mo = reduce(numpy.dot, (mo_coeff.T, veff_trans_xpy_ao, mo_coeff))
    lag_vo -= numpy.einsum('ki,ai->ak', veff_trans_xpy_mo[:nocc,:nocc], xpy)
    lag_vo += numpy.einsum('ac,ai->ci', veff_trans_xpy_mo[nocc:,nocc:], xpy)

    if abs(hyb) > 1e-10:
        veff_trans_xmy_mo = reduce(numpy.dot, (mo_coeff.T, veff_trans_xmy_ao, mo_coeff))
        lag_vo -= numpy.einsum('ki,ai->ak', veff_trans_xmy_mo[:nocc,:nocc], xmy)
        lag_vo += numpy.einsum('ac,ai->ci', veff_trans_xmy_mo[nocc:,nocc:], xmy)

    lag_vo *= 2
    # finished Lagrangian, solve z-vector

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm   = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dm+dm.T)
        return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()

    zvo   = cphf.solve(fvind, mo_energy, mo_occ, lag_vo,
                              max_cycle=td_grad.cphf_max_cycle,
                              tol=td_grad.cphf_conv_tol)[0]
    zvo   = zvo.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)


    # form gradients

    dm_z_ao  = reduce(numpy.dot, (orbv, zvo, orbo.T))
    veff_z_ao = vresp(dm_z_ao+dm_z_ao.T)

    diff_den_relaxed  = dm_z_ao + diff_den_ao
    diff_den_relaxed += diff_den_relaxed.T

    dm0 = reduce(numpy.dot, (orbo, orbo.T))  # ground-state density
    dm1 = dm0 + diff_den_relaxed/4  # excited-state density 

    fock_ao = mf.get_fock()
    lam_ao  = numpy.dot(fock_ao, dm1)
    veff_diff_den_relaxed  = veff_z_ao + veff_diff_den_ao
    lam_ao += numpy.dot(veff_diff_den_relaxed, dm0)/2

    lam_ao += numpy.dot(veff_trans_xpy_ao, tran_den_xpy_ao)/2
    if abs(hyb) > 1e-10:
        lam_ao -= numpy.dot(veff_trans_xmy_ao, tran_den_xmy_ao)/2

    cct     = reduce(numpy.dot, (mo_coeff, mo_coeff.T))
    wao1    = reduce(numpy.dot, [cct, lam_ao])
    wao1   += wao1.T


    if abs(hyb) > 1e-10:
        dms    = (dm0, diff_den_relaxed, tran_den_xpy_ao, tran_den_xmy_ao)
        vj, vk = td_grad.get_jk(mol, dms)
        vk    *= hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dms) * (alpha-hyb)

        vj     = vj.reshape(-1,3,nao,nao)
        vk     = vk.reshape(-1,3,nao,nao)
        veff1  = vj * 2 - vk
    else:
        vj    = td_grad.get_j(mol,  (dm0, diff_den_relaxed, tran_den_xpy_ao))
        vj    = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        veff1[:3] = vj * 2

    if has_xc:
        fxcz1 = _contract_xc_kernel(td_grad, mf.xc, dm_z_ao, None,
                                False, False, True, max_memory)[0]

        veff1[0] += vxc1[1:]
        veff1[1] +=(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
        veff1[2] += f1vo[1:] * 2

    # dipole derivative explicit contributions
    # note here m and n have to change places to be consistent with veff1 
    # the irp_ao integral also has an extra minus sign
    dm_ao_grad = numpy.einsum("xqmn,x->qnm", irp_ao, mpn_scaled) 
    veff1[2] -= dm_ao_grad*2

    time1     = log.timer('2e AO integral derivatives', *time1)    


    # the total gradient
    de = numpy.zeros((natm,3))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad     = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1          = mf_grad.get_ovlp(mol)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += veff1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += veff1[0,:,p0:p1].transpose(0,2,1)
        
        e1  = numpy.einsum('xpq,pq->x', h1ao, dm1) * 2
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], wao1[p0:p1])*2
        
        e1 += numpy.einsum('xij,ij->x', veff1[1,:,p0:p1], dm0[p0:p1])
        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], tran_den_xpy_ao[p0:p1]) * 2
        if abs(hyb) > 1e-10:
            e1 += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], tran_den_xmy_ao[p0:p1]) * 2
        
        de[k] = e1

    log.timer('TDHF nuclear gradients', *time0)

    return de


def grad_elec(td_grad, xy, mn, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Electronic part of TDA, TDHF nuclear gradients
    Args:
        td_grad : grad.tdrks.Gradients or grad.tdrks.Gradients object.
        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    td      = td_grad.base
    td_obj  = td.td_obj
    cav_obj = td.cav_obj
    td_grad = td_obj.nuc_grad_method()

    with_dse = not isinstance(cav_obj, qed.cavity.rhf.Rabi)
    assert singlet

    if with_dse:
        log.info("QED-TDRKS Gradient With DSE\n")
    else:
        log.info("QED-TDRKS Gradient No DSE\n")
    tmp = _qed_tdrks_elec_grad(td, xy, mn, singlet=singlet, atmlst=atmlst, 
                                with_dse=with_dse, max_memory=max_memory, verbose=verbose)
    
    log.timer('TDKS nuclear gradients', *time0)
    return tmp

class Gradients(tdrks_grad.Gradients):

    cphf_max_cycle = getattr(__config__, 'grad_tdrks_Gradients_cphf_max_cycle', 20)
    cphf_conv_tol  = getattr(__config__, 'grad_tdrks_Gradients_cphf_conv_tol', 1e-8)

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

    as_scanner = as_scanner

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
            logger.note(self, '------------ QED-TDDFT gradients for state %d ------------', self.state)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------------------')
