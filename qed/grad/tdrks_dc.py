import warnings
import numpy
from pyscf import lib, scf, dft
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad.rks import grids_response_cc
from pyscf.grad.tdrks import _contract_xc_kernel
from pyscf.scf import cphf
from pyscf.dft import gen_grid, numint
from pyscf import __config__


def print_matrix(keyword, matrix, nwidth=0):
    if '\n' in keyword[-3:]: keyword = keyword[:-2]
    print(keyword)

    if len(matrix.shape)==1: # 1d array
        if nwidth==0: nwidth = 6
        for n in range(len(matrix)):
            print('%13.8f ' % matrix[n], end='')
            if (n+1)%nwidth==0: print('')
        print('\n')

    elif len(matrix.shape)==2: # 2d array
        nrow, ncol = matrix.shape
        if nwidth==0:
            nloop = 1
        else:
            nloop = ncol//nwidth
            if nloop*nwidth<ncol: nloop += 1

        for n in range(nloop):
            s0, s1 = n*nwidth, (n+1)*nwidth
            if s1>ncol or nwidth==0: s1 = ncol

            for r in range(nrow):
                for c in range(s0, s1):
                    print('%13.8f ' % matrix[r,c], end='')
                print('')
            print('')

    elif len(matrix.shape)==3: # 3d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+str(i+1), matrix[i], nwidth)
    else:
        warnings.warn('the matrix has higher dimension than this funciton can handle.')


def _get_grid_response(td_grad, mf):
    grid_response = getattr(td_grad, 'grid_response', None)
    if grid_response is None:
        grid_response = getattr(mf, 'grid_response', False)
    return grid_response


def _is_many_qed(qed_td):
    return (hasattr(qed_td, 'cav_obj') and
            isinstance(getattr(qed_td, 'td_obj', None), list) and
            len(qed_td.td_obj) > 1)


def _split_many_xy(qed_td, x_y):
    x, y = x_y
    x = numpy.asarray(x)
    has_y = isinstance(y, numpy.ndarray)
    if has_y:
        y = numpy.asarray(y)

    cav_obj = qed_td.cav_obj
    xys = []
    for ifrag, td_obj in enumerate(qed_td.td_obj):
        mf = td_obj._scf
        mo_occ = mf.mo_occ
        nocc = (mo_occ > 0).sum()
        nvir = mo_occ.size - nocc
        n0, n1 = cav_obj.accum_nov[ifrag], cav_obj.accum_nov[ifrag+1]
        if n1 - n0 != nocc * nvir:
            raise ValueError('QED many-fragment amplitude size does not match '
                             f'fragment {ifrag} orbital dimensions')
        x_frag = x[n0:n1].reshape(nocc, nvir)
        if has_y:
            y_frag = y[n0:n1].reshape(nocc, nvir)
        else:
            y_frag = 0
        xys.append((x_frag, y_frag))
    return xys


def _split_many_atmlst(qed_td, atmlst):
    td_objs = qed_td.td_obj
    if atmlst is None:
        return [None] * len(td_objs)

    if (isinstance(atmlst, (list, tuple)) and len(atmlst) == len(td_objs) and
            any(x is None or isinstance(x, (list, tuple, range, numpy.ndarray))
                for x in atmlst)):
        return list(atmlst)

    natm = [td.mol.natm for td in td_objs]
    offsets = numpy.insert(numpy.cumsum(natm), 0, 0)
    frag_atmlst = [[] for _ in td_objs]
    for ia in atmlst:
        if ia < 0 or ia >= offsets[-1]:
            raise IndexError(f'atom index {ia} out of range for QED fragments')
        ifrag = numpy.searchsorted(offsets[1:], ia, side='right')
        frag_atmlst[ifrag].append(ia - offsets[ifrag])
    return frag_atmlst


def _many_fragment_ptrans(td_obj, xy_m, xy_n):
    mf = td_obj._scf
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nocc = (mo_occ > 0).sum()
    nvir = mo_occ.size - nocc
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]

    x1, y1 = xy_m
    x2, y2 = xy_n
    x_vo = numpy.array([x1.reshape(nocc, nvir).T,
                        x2.reshape(nocc, nvir).T])
    ptrans = numpy.einsum('pa,xai,qi->xpq', orbv, x_vo, orbo,
                          optimize=True)
    if isinstance(y1, numpy.ndarray):
        y_vo = numpy.array([y1.reshape(nocc, nvir).T,
                            y2.reshape(nocc, nvir).T])
        ptrans += numpy.einsum('pi,xai,qa->xpq', orbo, y_vo, orbv,
                               optimize=True)
    return ptrans


def _qed_m_plus_n(m_n, ncav):
    m, n = m_n
    if isinstance(n, numpy.ndarray):
        return (m + n).reshape(ncav)
    return numpy.asarray(m).reshape(ncav)


def _many_mpn_scaled(qed_td, frag_xys, m_n, with_dse):
    cav_obj = qed_td.cav_obj
    ncav = cav_obj.cavity_num
    cav_mode = cav_obj.cavity_mode.reshape(3, ncav)
    cav_freq = cav_obj.cavity_freq.reshape(ncav)

    mpn = numpy.array([_qed_m_plus_n(mn, ncav) for mn in m_n])
    mpn_scaled = numpy.einsum('xc,tc,c->tx', cav_mode, mpn,
                              numpy.sqrt(cav_freq/2))
    if not with_dse:
        return mpn_scaled

    dip_amp = numpy.zeros((2, 3))
    for ifrag, td_obj in enumerate(qed_td.td_obj):
        mol = td_obj.mol
        ptrans = _many_fragment_ptrans(td_obj, frag_xys[0][ifrag],
                                       frag_xys[1][ifrag])
        nao = ptrans.shape[-1]
        dip_ao = mol.intor("int1e_r", comp=3).reshape(3, nao, nao)
        dip_amp += numpy.einsum('ylu,tul->ty', dip_ao, ptrans) * 2

    dip_amp = numpy.einsum('ty,yc,xc->tx', dip_amp, cav_mode, cav_mode)
    return mpn_scaled + dip_amp


def _fragment_qed_obj(td_obj, cav_obj, mpn_scaled=None):
    obj = type('FragmentQEDView', (), {})()
    obj.td_obj = td_obj
    obj.cav_obj = cav_obj
    if mpn_scaled is not None:
        obj._tdrks_dc_mpn_scaled = mpn_scaled
    return obj


def tdrks_deriv_coupling_ge(qed_td, x_y, m_n, energy, singlet=True, atmlst=None,
                        with_dse=True, max_memory=2000, verbose=logger.INFO,
                        Theta=None, iprint=None):

    if (hasattr(qed_td, 'cav_obj') and
            isinstance(getattr(qed_td, 'td_obj', None), list) and
            len(qed_td.td_obj) == 1):
        qed_td = _fragment_qed_obj(qed_td.td_obj[0], qed_td.cav_obj)

    if _is_many_qed(qed_td):
        frag_xys = _split_many_xy(qed_td, x_y)
        frag_atmlst = _split_many_atmlst(qed_td, atmlst)
        de = []
        for td_obj, xy_frag, atmlst_frag in zip(qed_td.td_obj, frag_xys,
                                                frag_atmlst):
            if atmlst_frag == []:
                continue
            de.append(tdrks_deriv_coupling_ge(
                td_obj, xy_frag, None, energy, singlet=singlet,
                atmlst=atmlst_frag, with_dse=with_dse,
                max_memory=max_memory, verbose=verbose, Theta=Theta,
                iprint=iprint))
        return numpy.vstack(de) if de else numpy.zeros((0, 3))

    qed = False
    if hasattr(qed_td, 'cav_obj'): # qed
        td_obj  = qed_td.td_obj
        cav_obj = qed_td.cav_obj
        qed = True
    else: # normal tddft
        td_obj = qed_td
    td_grad = td_obj.nuc_grad_method()

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    nbas = mo_coeff.shape[0]

    x, y = x_y
    xmy = (x-y).reshape(nocc,nvir).T
    if iprint:
        print_matrix('xmy:\n', xmy)

    if Theta is not None:
        print('X*Theta: ', numpy.einsum('ai,xai->x', xmy, Theta)*2)

    tran_den_xmy_ao  = orbv @ xmy @ orbo.T
    if iprint:
        print_matrix('tran_den_xmy_ao:\n', tran_den_xmy_ao)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)

    # start to form Lagrangian
    lag_vo = xmy
    if iprint:
        print_matrix('lag_vo:\n', lag_vo)
    # finished Lagrangian, solve z-vector

    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm   = orbv @ (x.reshape(nvir,nocc)*2) @ orbo.T
        v1ao = vresp(dm+dm.T)
        return (orbv.T @ v1ao @ orbo).ravel()

    zvo   = cphf.solve(fvind, mo_energy, mo_occ, lag_vo,
                              max_cycle=td_grad.cphf_max_cycle,
                              tol=td_grad.cphf_conv_tol)[0]
    zvo   = zvo.reshape(nvir,nocc)
    if iprint:
        print_matrix('zvo:\n', zvo)
    time1 = log.timer('Z-vector using CPHF solver', *time0)


    # form gradients
    dm_z_ao  = orbv @ zvo @ orbo.T
    dm_z_ao *= 2
    dm_z_ao_sym = dm_z_ao + dm_z_ao.T
    if iprint:
        print_matrix('dm_z_ao_sym:\n', dm_z_ao_sym)
    veff_z_ao = vresp(dm_z_ao_sym)

    dm0     = orbo @ orbo.T * 2  # ground-state density
    cct     = mo_coeff @ mo_coeff.T
    fock_ao = mf.get_fock()

    # W combines with S^x
    wao1  = veff_z_ao @ dm0
    wao1 += fock_ao @ dm_z_ao_sym
    wao1 = cct @ wao1
    wao1 += wao1.T
    if iprint:
        print_matrix('wao1:\n', wao1)


    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()

    de   = numpy.zeros((len(atmlst),3))
    de_etf   = numpy.zeros((len(atmlst),3))
    de_force = numpy.zeros((len(atmlst),3))

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad     = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1          = mf_grad.get_ovlp(mol)

    # figure out DFT type
    dft_type = dft.libxc.parse_xc(mf.xc)
    #ni = mf._numint
    #ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    has_xc = len(dft_type[1]) > 0
    grid_response = _get_grid_response(td_grad, mf)
    #log.info('hyb: ', hyb, ' dft: ', has_xc)

    if has_xc:
        mf_grad.grid_response = grid_response
    vhf1         = mf_grad.get_veff(mol, dm0)
    vhf2         = mf_grad.get_veff(mol, dm_z_ao_sym)

    dms = (dm0, dm_z_ao_sym)
    if abs(hyb) > 1e-10:
        vj, vk = td_grad.get_jk(mol, dms)
        vk    *= hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dms) * (alpha-hyb)

        vj     = vj.reshape(-1,3,nao,nao)
        vk     = vk.reshape(-1,3,nao,nao)
        veff1  = vj * 2 - vk
    else:
        vj    = td_grad.get_j(mol, dms)
        vj    = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((4,3,nao,nao))
        veff1 = vj * 2

    if has_xc:
        fxcz1 = _contract_xc_kernel(td_grad, mf.xc, dm_z_ao_sym, None,
                False, False, True, max_memory)[0]

        veff1[1] = fxcz1[1:]*2
    time1       = log.timer('2e AO integral derivatives', *time1)

    if iprint:
        print_matrix('s1:\n', s1[2])
        print_matrix('vhf1:\n', vhf1[2])
        print_matrix('veff1:\n', veff1[0,2])


    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h1ao = hcore_deriv(ia)

        # orbital rotation derivative contribution
        e1  = numpy.einsum('xpq,pq->x', h1ao, dm_z_ao_sym)
        e1 += numpy.einsum('xpq,pq->x', vhf1[:,p0:p1], dm_z_ao_sym[p0:p1])*2
        e1 += numpy.einsum('xpq,pq->x', vhf2[:,p0:p1], dm0[p0:p1])*2
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], wao1[p0:p1])
        e1 *= 0.5

        f1  = e1 * energy
        de_force[k] = f1
        de_etf[k]   = e1

        # asymmetric overlap derivative contribution
        e1 += numpy.einsum('xpq,pq->x', s1[:,p0:p1],
                           tran_den_xmy_ao[:,p0:p1].T - tran_den_xmy_ao[p0:p1])

        de[k] = e1

    log.timer('TDDFT derivative coupling', *time0)


    print_matrix('Derivative coupling without ETF:\n', de)
    print_matrix('Force elements:\n', de_force)
    print_matrix('Derivative coupling with ETF:\n', de_etf)

    return de


def tdrks_deriv_coupling_ee(qed_td, x_y, m_n, energy, singlet=True, atmlst=None,
                        with_dse=True, max_memory=2000, verbose=logger.INFO,
                        iprint=None):

    if (hasattr(qed_td, 'cav_obj') and
            isinstance(getattr(qed_td, 'td_obj', None), list) and
            len(qed_td.td_obj) == 1):
        mpn_scaled = getattr(qed_td, '_tdrks_dc_mpn_scaled', None)
        qed_td = _fragment_qed_obj(qed_td.td_obj[0], qed_td.cav_obj,
                                   mpn_scaled)

    if _is_many_qed(qed_td):
        frag_xys = [_split_many_xy(qed_td, xy) for xy in x_y]
        frag_atmlst = _split_many_atmlst(qed_td, atmlst)
        mpn_scaled = _many_mpn_scaled(qed_td, frag_xys, m_n, with_dse)

        de = []
        for ifrag, td_obj in enumerate(qed_td.td_obj):
            if frag_atmlst[ifrag] == []:
                continue
            frag_qed = _fragment_qed_obj(td_obj, qed_td.cav_obj, mpn_scaled)
            de.append(tdrks_deriv_coupling_ee(
                frag_qed, [frag_xys[0][ifrag], frag_xys[1][ifrag]],
                m_n, energy, singlet=singlet, atmlst=frag_atmlst[ifrag],
                with_dse=with_dse, max_memory=max_memory, verbose=verbose,
                iprint=iprint))
        return numpy.vstack(de) if de else numpy.zeros((0, 3))

    qed = False
    if hasattr(qed_td, 'cav_obj'): # qed
        td_obj  = qed_td.td_obj
        cav_obj = qed_td.cav_obj
        qed = True
    else: # normal tddft
        td_obj = qed_td
    td_grad = td_obj.nuc_grad_method()

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]
    nbas = mo_coeff.shape[0]

    energy_i, energy_j = energy
    energy_diff = energy_j - energy_i
    #print('energy: ', energy_i, energy_j, energy_diff)
    [x1, y1], [x2, y2] = x_y

    rpa = True if isinstance(y1, numpy.ndarray) else False

    # different from the gradient code
    # do not add the transpose
    x_vo = numpy.array([x1.reshape(nocc,nvir).T, x2.reshape(nocc,nvir).T])
    pvv = x_vo[0] @ x_vo[1].T
    poo = x_vo[1].T @ x_vo[0]
    Ptrans = numpy.einsum('pa,xai,qi->xpq', orbv, x_vo, orbo,
                          optimize=True)
    if rpa:
        y_vo = numpy.array([y1.reshape(nocc,nvir).T, y2.reshape(nocc,nvir).T])
        pvv += y_vo[1] @ y_vo[0].T
        poo += y_vo[0].T @ y_vo[1]
        Ptrans += numpy.einsum('pi,xai,qa->xpq', orbo, y_vo, orbv,
                               optimize=True)
    Pdiff = orbv @ pvv @ orbv.T
    Pdiff -= orbo @ poo @ orbo.T
    diff_den_ao = (Pdiff+Pdiff.T)

    if qed:
        ncav      = cav_obj.cavity_num
        cav_mode  = cav_obj.cavity_mode.reshape(3, ncav)
        cav_freq  = cav_obj.cavity_freq.reshape(ncav, )

        # dipole and dipole derivatives AO matrices
        dip_ao = mol.intor("int1e_r", comp=3).reshape(3, nao, nao)
        irp_ao = mol.intor("int1e_irp", comp=9, hermi=0).reshape(3,3,nao,nao)

        mpn_scaled = getattr(qed_td, '_tdrks_dc_mpn_scaled', None)
        if mpn_scaled is None:
            [m1, n1], [m2, n2] = m_n

            # photon amplitudes (t is two states)
            mpn = numpy.array([_qed_m_plus_n((m1, n1), ncav),
                               _qed_m_plus_n((m2, n2), ncav)])
            mpn_scaled = numpy.einsum('xc,tc,c->tx', cav_mode, mpn, numpy.sqrt(cav_freq/2))

            if with_dse:
                # 2 because the Ptrans without its transpose
                dip_amp = numpy.einsum('ylu,tul->ty', dip_ao, Ptrans)*2
                dip_amp = numpy.einsum('ty,yc,xc->tx', dip_amp, cav_mode, cav_mode)
                mpn_scaled += dip_amp # here add dse contribution to the off-diagnoal dipole


    mem_now = lib.current_memory()[0]
    max_memory = max(2000, td_grad.max_memory*.9-mem_now)


    # figure out DFT type
    dft_type = dft.libxc.parse_xc(mf.xc)
    #ni = mf._numint
    #ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    has_xc = len(dft_type[1]) > 0
    grid_response = _get_grid_response(td_grad, mf)
    #log.info('hyb: ', hyb, ' dft: ', has_xc)

    # two-electron terms
    # no transpose for transition densities
    dms = (diff_den_ao, Ptrans[0], Ptrans[1])
    if abs(hyb) > 1e-10:
        vj, vk = mf.get_jk(mol, dms, hermi=0)
        vk *= hyb
        if abs(omega) > 1e-10:
            vk += mf.get_k(mol, dms, hermi=0, omega=omega) * (alpha-hyb)

        veff_diff_den_ao  = vj[0] * 2 - vk[0]
        if singlet:
            veff_trans_ao = vj[1:] * 2 - vk[1:]
        else:
            veff_trans_ao = -vk[1:]
    else:
        vj = mf.get_j(mol, dms, hermi=0)

        veff_diff_den_ao  = vj[0] * 2
        if singlet:
            veff_trans_ao = vj[1:] * 2
        else:
            veff_trans_ao = None

    if has_xc:
        # the function will symmetrize the transition density matrix
        f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel_dc(td_grad, mf.xc,
                                            Ptrans, diff_den_ao, True, True,
                                            singlet, max_memory)

        veff_diff_den_ao += f1oo[0] + k1ao[0] * 2
        veff_trans_ao += f1vo[:,0]

    if qed:
        # dipole contributions
        gm_ao = numpy.einsum("xmn,tx->tmn", dip_ao, mpn_scaled)
        veff_trans_ao += gm_ao


    # Build RHS of CPHF equation
    lag_vo  = orbv.T @ veff_diff_den_ao @ orbo

    veff_trans_mo = mo_coeff.T @ veff_trans_ao[0] @ mo_coeff
    lag_vo += numpy.einsum('ac,ai->ci', veff_trans_mo[nocc:,nocc:], x_vo[1])
    lag_vo -= numpy.einsum('ij,cj->ci', veff_trans_mo[:nocc,:nocc], x_vo[1])
    if rpa:
        lag_vo += numpy.einsum('ca,ai->ci', veff_trans_mo[nocc:,nocc:], y_vo[1])
        lag_vo -= numpy.einsum('ji,cj->ci', veff_trans_mo[:nocc,:nocc], y_vo[1])

    veff_trans_mo = mo_coeff.T @ veff_trans_ao[1] @ mo_coeff
    lag_vo += numpy.einsum('ac,ai->ci', veff_trans_mo[nocc:,nocc:], x_vo[0])
    lag_vo -= numpy.einsum('ij,cj->ci', veff_trans_mo[:nocc,:nocc], x_vo[0])
    if rpa:
        lag_vo += numpy.einsum('ca,ai->ci', veff_trans_mo[nocc:,nocc:], y_vo[0])
        lag_vo -= numpy.einsum('ji,cj->ci', veff_trans_mo[:nocc,:nocc], y_vo[0])

    if iprint:
        print_matrix('lag_vo:\n', lag_vo)
    # finished Lagrangian, solve z-vector


    # set singlet=None, generate function for CPHF type response kernel
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind(x):  # For singlet, closed shell ground state
        dm   = orbv @ (x.reshape(nvir,nocc)*2) @ orbo.T
        v1ao = vresp(dm+dm.T)
        return (orbv.T @ v1ao @ orbo).ravel()

    zvo = cphf.solve(fvind, mo_energy, mo_occ, lag_vo,
                            max_cycle=td_grad.cphf_max_cycle,
                            tol=td_grad.cphf_conv_tol)[0]
    zvo = zvo.reshape(nvir,nocc)
    if iprint:
        print_matrix('zvo:\n', zvo)
    time1 = log.timer('Z-vector using CPHF solver', *time0)


    # form gradients

    dm_z_ao  = orbv @ zvo @ orbo.T
    dm_z_ao += dm_z_ao.T
    #print_matrix('dm_z_ao:\n', dm_z_ao)
    veff_z_ao = vresp(dm_z_ao)

    diff_den_relaxed  = dm_z_ao + diff_den_ao
    if iprint:
        print_matrix('diff_den_relaxed:\n', diff_den_relaxed)

    dm0 = orbo @ orbo.T  # ground-state density
    #dm1 = dm0 + diff_den_relaxed/4  # excited-state density

    fock_ao = mf.get_fock()
    lam_ao  = numpy.dot(fock_ao, diff_den_relaxed)
    veff_diff_den_relaxed = veff_z_ao*2 + veff_diff_den_ao
    lam_ao += numpy.dot(veff_diff_den_relaxed, dm0)

    lam_ao += numpy.dot(veff_trans_ao[0], Ptrans[1].T)
    lam_ao += numpy.dot(veff_trans_ao[1], Ptrans[0].T)
    lam_ao += numpy.dot(veff_trans_ao[0].T, Ptrans[1])
    lam_ao += numpy.dot(veff_trans_ao[1].T, Ptrans[0])
    #print_matrix('lam_ao:\n', lam_ao)

    cct  = mo_coeff @ mo_coeff.T
    wao1 = cct @ lam_ao
    wao1 += wao1.T
    if iprint:
        print_matrix('wao1:\n', wao1)

    # last term: X^T*X, X*X^T for SRx
    ewd_s = Pdiff - Pdiff.T
    # ETF term: refer Eq. (21) in JPCL. 2012, 3, 2039−2043.
    ewd_etf  = orbo @ poo @ orbo.T # note here is plus
    ewd_etf += orbv @ pvv @ orbv.T
    ewd_etf *= 0.5 # SRx of ETF term has an extra 1/2
    ewd_etf -= ewd_etf.T


    # two-electron terms
    dms = (dm0, diff_den_relaxed, Ptrans[0], Ptrans[1], Ptrans[0].T, Ptrans[1].T)
    if abs(hyb) > 1e-10:
        vj, vk = td_grad.get_jk(mol, dms)
        vk *= hyb
        if abs(omega) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk += td_grad.get_k(mol, dms) * (alpha-hyb)

        vj    = vj.reshape(-1,3,nao,nao)
        vk    = vk.reshape(-1,3,nao,nao)
        if singlet:
            veff1 = vj * 2 - vk
        else:
            veff1 = numpy.vstack((vj[:2]*2-vk[:2], -vk[2:]))
    else:
        vj    = td_grad.get_j(mol, dms[:4])
        vj    = vj.reshape(-1,3,nao,nao)
        veff1 = numpy.zeros((6,3,nao,nao))
        if singlet:
            veff1[:4] = vj * 2
            veff1[4:] = vj[2:] * 2
        else:
            veff1[:2] = vj[:2] * 2

    if has_xc:
        fxcz1 = _contract_xc_kernel(td_grad, mf.xc, dm_z_ao*0.5, None,
                                False, False, True, max_memory)[0]

        veff1[0] += vxc1[1:]
        veff1[1] += f1oo[1:] + (fxcz1[1:] + k1ao[1:]) * 2
        veff1[2:4] += f1vo[:,1:]
        veff1[4:] += f1vo[:,1:]

    if has_xc and grid_response:
        grid_sum = _contract_xc_kernel_dc_grid(
                td_grad, mf.xc, Ptrans, diff_den_relaxed/4,
                singlet, max_memory)

    if qed:
        # dipole derivative explicit contributions
        # note here m and n have to change places to be consistent with veff1
        # the irp_ao integral also has an extra minus sign
        dm_ao_grad = numpy.einsum("xqmn,tx->tqnm", irp_ao, mpn_scaled)
        veff1[2:4] -= dm_ao_grad
        veff1[4:] -= dm_ao_grad

    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()

    de = numpy.zeros((len(atmlst),3))
    de_force = numpy.zeros((len(atmlst),3))
    de_etf = numpy.zeros((len(atmlst),3))


    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad     = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1          = mf_grad.get_ovlp(mol)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        h1ao[:,p0:p1]   += veff1[0,:,p0:p1]
        h1ao[:,:,p0:p1] += veff1[0,:,p0:p1].transpose(0,2,1)

        e1  = numpy.einsum('xpq,pq->x', h1ao, diff_den_relaxed)
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], wao1[p0:p1])

        e1 += numpy.einsum('xij,ij->x', veff1[1,:,p0:p1], dm0[p0:p1])
        e1 += numpy.einsum('xji,ij->x', veff1[1,:,p0:p1], dm0[:,p0:p1])

        e1 += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], Ptrans[1][p0:p1])*2
        e1 += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], Ptrans[0][p0:p1])*2
        e1 += numpy.einsum('xji,ij->x', veff1[4,:,p0:p1], Ptrans[1][:,p0:p1])*2
        e1 += numpy.einsum('xji,ij->x', veff1[5,:,p0:p1], Ptrans[0][:,p0:p1])*2

        e1 /= (energy_diff)
        e1 -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], ewd_s[p0:p1]) # SRx
        de[k] = e1

        # ETF: asymmetric overlap derivative contribution
        e1 += numpy.einsum('xpq,pq->x', s1[:,p0:p1], ewd_etf[p0:p1])

        de_etf[k] = e1

    if has_xc and grid_response:
        grid_sum = grid_sum[list(atmlst)] / energy_diff
        de += grid_sum
        de_etf += grid_sum

    de_force = de_etf * energy_diff

    log.timer('TDHF nuclear gradients', *time0)

    print_matrix('Derivative coupling without ETF:\n', de)
    print_matrix('Force element:\n', de_force)
    print_matrix('Derivative coupling with ETF:\n', de_etf)

    return de


def _get_state_mn(qed_td, state):
    if hasattr(qed_td, 'mn'):
        return qed_td.mn[state]
    return None


def _default_with_dse(qed_td):
    if not hasattr(qed_td, 'cav_obj'):
        return False
    name = type(qed_td.cav_obj).__name__.upper()
    return 'ROTATINGWAVE' in name or 'PAULIFIERZ' in name


def tdrks_deriv_coupling(qed_td, states=None, singlet=True,
                         atmlst=None, with_dse=None, max_memory=2000,
                         verbose=logger.INFO, Theta=None, iprint=None):
    if states is None:
        states = [0, 1]
    states = list(states)
    if len(states) < 2:
        raise ValueError('states must contain at least two state indices')
    if min(states) < 0:
        raise ValueError('states must be non-negative 0-based state indices')
    if len(set(states)) != len(states):
        raise ValueError('states must not contain duplicate indices')
    if max(states) >= len(qed_td.e):
        raise ValueError('state index exceeds the available excited states')

    if with_dse is None:
        with_dse = _default_with_dse(qed_td)

    de_ge = {}
    for state in states:
        de_ge[state] = tdrks_deriv_coupling_ge(
            qed_td, qed_td.xy[state], _get_state_mn(qed_td, state),
            qed_td.e[state], singlet=singlet, atmlst=atmlst,
            with_dse=with_dse, max_memory=max_memory, verbose=verbose,
            Theta=Theta, iprint=iprint)

    de_ee = {}
    for i, state1 in enumerate(states):
        for state2 in states[i+1:]:
            de_ee[(state1, state2)] = tdrks_deriv_coupling_ee(
                qed_td, [qed_td.xy[state1], qed_td.xy[state2]],
                [_get_state_mn(qed_td, state1), _get_state_mn(qed_td, state2)],
                [qed_td.e[state1], qed_td.e[state2]], singlet=singlet,
                atmlst=atmlst, with_dse=with_dse, max_memory=max_memory,
                verbose=verbose, iprint=iprint)
    return de_ge, de_ee


def _print_dc_values(title, de_ge, de_ee):
    print(title)
    for state in sorted(de_ge):
        print_matrix(f'GE state {state} derivative coupling:\n', de_ge[state])
    for state_pair in sorted(de_ee):
        print_matrix(f'EE states {state_pair[0]}-{state_pair[1]} derivative coupling:\n',
                     de_ee[state_pair])


"""
adopted from grad/tdrks.py, with two sets of transition densities
for derivative couplings
"""
def _contract_xc_kernel_dc(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, singlet=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # dmvo ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmvo = (dmvo + dmvo.transpose(0,2,1)) * .5 # because K_{ia,jb} == K_{ia,jb}

    f1vo = numpy.zeros((2,4,nao,nao))  # 0th-order, d/dx, d/dy, d/dz
    deriv = 2
    if dmoo is not None:
        f1oo = numpy.zeros((4,nao,nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = numpy.zeros((4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = numpy.zeros((4,nao,nao))
        deriv = 3
    else:
        k1ao = None

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
    else:
        raise NotImplementedError(f'td-rks derivative coupling for {xc_code}')

    if not singlet:
        raise NotImplementedError(f'{xctype} triplet')
    if mf.do_nlc():
        raise NotImplementedError("TDDFT derivative coupling with NLC contribution "
                                  "is not supported yet.")

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype,
                           with_lapl=False)
        vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv,
                                       xctype=xctype)[1:]

        rho1 = []
        for x in range(2): # two excited-states
            rho1x = ni.eval_rho(mol, ao0, dmvo[x], mask, xctype,
                                hermi=1, with_lapl=False) * 2
            if xctype == 'LDA':
                rho1x = rho1x[numpy.newaxis]
            rho1.append(rho1x)
            wv = numpy.einsum('yg,xyg,g->xg', rho1x, fxc, weight)
            fmat_(mol, f1vo[x], ao, wv, mask, shls_slice, ao_loc)

        if dmoo is not None:
            rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype,
                               hermi=1, with_lapl=False) * 2
            if xctype == 'LDA':
                rho2 = rho2[numpy.newaxis]
            wv = numpy.einsum('yg,xyg,g->xg', rho2, fxc, weight)
            fmat_(mol, f1oo, ao, wv, mask, shls_slice, ao_loc)
        if with_vxc:
            fmat_(mol, v1ao, ao, vxc * weight, mask, shls_slice, ao_loc)
        if with_kxc:
            wv = numpy.einsum('yg,zg,xyzg,g->xg',
                              rho1[0], rho1[1], kxc, weight)
            fmat_(mol, k1ao, ao, wv, mask, shls_slice, ao_loc)

    f1vo[:,1:] *= -1
    if f1oo is not None: f1oo[1:] *= -1
    if v1ao is not None: v1ao[1:] *= -1
    if k1ao is not None: k1ao[1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


def _contract_xc_kernel_dc_grid(td_grad, xc_code, dmvo, dmoo,
                                singlet=True, max_memory=2000):
    """
    Compute the grid contribution to the derivative coupling for TDDFT.
    """
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    ao_loc = mol.ao_loc_nr()

    nocc = (mo_occ > 0).sum()
    dm0 = mo_coeff[:, :nocc] @ mo_coeff[:, :nocc].T

    dmvo = (dmvo + dmvo.transpose(0, 2, 1)) * .5
    dmoo = (dmoo + dmoo.T) * .5

    if not singlet:
        raise NotImplementedError(f'{xctype} triplet')
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(f'{xctype}')

    excsum = numpy.zeros((mol.natm, 3))

    def make_vtmp(ao, wv, weight, mask, vtmp):
        vtmp.fill(0)
        if xctype == 'LDA':
            aow = numint._scale_ao(ao[0], weight * wv[0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
        elif xctype == 'GGA':
            wv = wv * weight
            wv[0] *= .5
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
        else:
            wv = wv * weight
            wv[0] *= .5
            wv[4] *= .5
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[:4], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)

    ao_deriv = 1 if xctype == 'LDA' else 2
    vtmp = numpy.empty((3, nao, nao))
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                        cutoff=grids.cutoff)
        if xctype == 'LDA':
            ao0 = ao[0]
        elif xctype == 'GGA':
            ao0 = ao[:4]
        else:
            ao0 = ao[:10]

        rho = ni.eval_rho2(mol, ao0, mo_coeff, mo_occ, mask, xctype,
                           with_lapl=False)
        exc, vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, 3,
                                            xctype=xctype)

        rho1m = ni.eval_rho(mol, ao0, dmvo[0], mask, xctype, hermi=1,
                            with_lapl=False) * 2
        rho1n = ni.eval_rho(mol, ao0, dmvo[1], mask, xctype, hermi=1,
                            with_lapl=False) * 2
        rho2 = ni.eval_rho(mol, ao0, dmoo, mask, xctype, hermi=1,
                           with_lapl=False) * 2
        if xctype == 'LDA':
            rho1m = rho1m[numpy.newaxis]
            rho1n = rho1n[numpy.newaxis]
            rho2 = rho2[numpy.newaxis]

        # Off-diagonal derivative couplings do not include the pure
        # ground-state E_xc grid response.  Only the relaxed density and
        # transition-density bilinear terms enter this scalar.
        wv = numpy.einsum('xg,xg->g', vxc, rho2)
        wv += numpy.einsum('xg,yg,xyg->g', rho1m, rho1n, fxc)
        excsum += numpy.einsum('g,nxg->nx', wv, weight1)

        wv0 = numpy.einsum('yg,xyg->xg', rho2, fxc)
        wv0 += numpy.einsum('yg,zg,xyzg->xg', rho1m, rho1n, kxc)

        wv1m = numpy.einsum('yg,xyg->xg', rho1n, fxc)
        wv1n = numpy.einsum('yg,xyg->xg', rho1m, fxc)

        for wv_i, dm_i in ((wv0, dm0), (vxc, dmoo),
                           (wv1m, dmvo[0]), (wv1n, dmvo[1])):
            make_vtmp(ao, wv_i, weight, mask, vtmp)
            e1 = numpy.einsum('xij,ji->x', vtmp, dm_i) * 4
            excsum[atm_id] += e1

    return excsum


if __name__ == '__main__':
    import argparse
    from pyscf import gto, tdscf
    import qed

    parser = argparse.ArgumentParser(
            description='Run small normal and QED TDA derivative-coupling examples.')
    parser.add_argument('--xc', default='lda',
                        help='RKS functional. Use "hf" to run TDA/HF.')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--nroots', type=int, default=3)
    parser.add_argument('--states', type=int, nargs='+', default=[0, 1],
                        help='0-based excited-state indices for GE and EE derivative couplings.')
    parser.add_argument('--qed-cavity', default='JC',
                        choices=('JC', 'Rabi', 'RWA', 'PF'),
                        help='Cavity Hamiltonian for the QED-TDA example.')
    parser.add_argument('--cavity-scale', type=float, default=0.02,
                        help='Scale applied to the first transition dipole for the cavity mode.')
    parser.add_argument('--grid-response', action='store_true',
                        help='Enable DFT grid-weight response where supported.')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    atom = '''
    O   0.0000000  -0.1113511   0.0000000
    H   0.0000000   0.4454045   0.7830366
    H   0.0000000   0.4454045  -0.7830366
    '''

    mol = gto.M(
        atom=atom,
        basis=args.basis,
        spin=0,
        charge=0,
        verbose=args.verbose,
    )

    mf = scf.RKS(mol)
    mf.xc = args.xc
    mf.grid_response = args.grid_response
    mf.grids.prune = True
    mf.kernel()

    td = tdscf.TDA(mf)
    states = list(args.states)
    if len(states) < 2:
        raise ValueError('--states must contain at least two 0-based state indices')
    if min(states) < 0:
        raise ValueError('--states entries must be non-negative 0-based state indices')
    if len(set(states)) != len(states):
        raise ValueError('--states entries must be unique')
    nroots = max(args.nroots, max(states) + 1)

    td.nroots = nroots
    td.verbose = args.verbose
    td.kernel()

    print('Normal TDA derivative-coupling example')
    print('functional:', args.xc)
    print('basis:', args.basis)
    print('0-based states:', states)
    print('excitation energies / eV:')
    print(td.e * 27.211386245988)

    print('ground-state and selected excited-states')
    print('selected excited-state pairs')
    de_ge, de_ee = tdrks_deriv_coupling(td, states=states,
                                        verbose=args.verbose)
    _print_dc_values('Normal TDA returned derivative couplings', de_ge, de_ee)

    cavity_freq = numpy.asarray([td.e[0]])
    cavity_mode = numpy.asarray(td.transition_dipole()[0]).reshape(3, 1)
    cavity_mode *= args.cavity_scale

    key = {
        'cavity_freq': cavity_freq,
        'cavity_mode': cavity_mode,
        'nstates': nroots,
        'resonance_state': 1,
        'target_states': 'polariton',
        'uniform_field': True,
        'has_offdiag': False,
    }
    cav_model = getattr(qed, args.qed_cavity)
    with_dse = args.qed_cavity in ('RWA', 'PF')
    cav_obj = cav_model(mf, key)
    qed_td = qed.TDA(mf, td, cav_obj, key)
    qed_td.nroots = nroots
    qed_td.verbose = args.verbose
    qed_td.kernel()

    print('QED TDA derivative-coupling example')
    print('cavity model:', args.qed_cavity)
    print('cavity frequency / eV:', cavity_freq * 27.211386245988)
    print('cavity mode:')
    print(cavity_mode.reshape(3))
    print('QED excitation energies / eV:')
    print(qed_td.e * 27.211386245988)

    print('ground-state and selected QED excited-states')
    print('selected QED excited-state pairs')
    de_ge, de_ee = tdrks_deriv_coupling(qed_td, states=states,
                                        with_dse=with_dse,
                                        verbose=args.verbose)
    _print_dc_values('QED TDA returned derivative couplings', de_ge, de_ee)
