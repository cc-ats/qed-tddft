import sys
import copy
import numpy
from pyscf import __config__
from pyscf import lib
from pyscf.scf import hf
from pyscf.dft import rks

# from mqed.lib      import logger

TIGHT_GRAD_CONV_TOL = getattr(__config__, "scf_hf_kernel_tight_grad_conv_tol", True)


def get_veff(mf, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, vhfopt=None):
    """
    QEDHF Veff construction
    Coulomb + XC functional

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    """

    # HF veff
    # may implement a universal veff for QEDHF/RKS

    print("debug: qed.get_veff")
    """
    if dm_last is None:
        vj, vk = get_jk(mol, numpy.asarray(dm), hermi, vhfopt)
        return vj - vk * .5
    else:
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        vj, vk = get_jk(mol, ddm, hermi, vhfopt)
        return vj - vk * .5 + numpy.asarray(vhf_last)
    """

    raise NotImplementedError


# we don't need this, only need to define veff
# def get_fock(mf, cav=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
#             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
#    '''F = h^{core} + V^{HF}
#    cav: cavity object
#
#    '''


# may be better to implement QEDHF as a HF instance, need to run bare HF first and run QED-HF
# i.e., class HF(hf.SCF)


class RHF(hf.RHF):
    # class HF(lib.StreamObject):
    r"""
    QEDSCF base class.   non-relativistic RHF.

    """

    def __init__(self, mol, xc=None, **kwargs):
        hf.RHF.__init__(self, mol)
        # if xc is not None:
        #    rks.KohnShamDFT.__init__(self, xc)

        cavity = None
        if "cavity" in kwargs:
            cavity = kwargs["cavity"]
        if "cavity_mode" in kwargs:
            cavity_mode = kwargs["cavity_mode"]
        else:
            raise ValueError("The required keyword argument 'cavity_mode' is missing")

        if "cavity_freq" in kwargs:
            cavity_freq = kwargs["cavity_freq"]
        else:
            raise ValueError("The required keyword argument 'cavity_freq' is missing")

        print("cavity_freq=", cavity_freq)
        print("cavity_mode=", cavity_mode)

        self.cavity_freq = cavity_freq
        self.cavity_mode = cavity_mode
        self.nmode = len(cavity_freq)
        nao = self.mol.nao_nr()

        # make dipole matrix in AO
        self.make_dipolematrix()
        self.gmat = numpy.empty((self.nmode, nao, nao))
        self.gmat = lib.einsum("Jx,xuv->Juv", self.cavity_mode, self.mu_mat_ao)
        x_out_y = 0.5 * numpy.outer(cavity_mode, cavity_mode).reshape(-1)
        self.qd2 = lib.einsum("J, Juv->uv", x_out_y, self.qmat)

        # print(f"{cavity} cavity mode is used!")
        # self.verbose    = mf.verbose
        # self.stdout     = mf.stdout
        # self.mol        = mf.mol
        # self.max_memory = mf.max_memory
        # self.chkfile    = mf.chkfile
        # self.wfnsym     = None
        self.dip_ao = mol.intor("int1e_r", comp=3)

    def make_dipolematrix(self):
        """
        return dipole and quadrupole matrix in AO

        Quarupole:
        # | xx, xy, xz |
        # | yx, yy, yz |
        # | zx, zy, zz |
        # xx <-> rrmat[0], xy <-> rrmat[3], xz <-> rrmat[6]
        #                  yy <-> rrmat[4], yz <-> rrmat[7]
        #                                   zz <-> rrmat[8]
        """

        self.mu_mo = None
        charges = self.mol.atom_charges()
        coords = self.mol.atom_coords()
        charge_center = (0, 0, 0)  # np.einsum('i,ix->x', charges, coords)
        with self.mol.with_common_orig(charge_center):
            self.mu_mat_ao = self.mol.intor_symmetric("int1e_r", comp=3)
            self.qmat = -self.mol.intor("int1e_rr")

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if not omega and (
            self._eri is not None or mol.incore_anyway or self._is_mem_enough()
        ):
            if self._eri is None:
                self._eri = mol.intor("int2e", aosym="s8")
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = RHF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    # get_veff = get_veff
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        r"""QED Hartree-Fock potential matrix for the given density matrix

        .. math::
            V_{eff} = J - K/2 + \bra{i}\lambda\cdot\mu\ket{j}

        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * 0.5
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * 0.5
            vhf += numpy.asarray(vhf_last)

        # add photon contribution
        if self.mu_mat_ao is None:
            self.make_dipolematrix()

        mu_mo = lib.einsum("pq, Xpq ->X", dm, self.mu_mat_ao)
        scaled_mu = 0.0
        self.mu_lambda = 0.0
        for imode in range(self.nmode):
            self.mu_lambda -= numpy.dot(mu_mo, self.cavity_mode[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, self.gmat[imode])

        self.oei = self.gmat * self.mu_lambda
        self.oei -= self.qd2
        self.oei = numpy.sum(self.oei, axis=0)

        vhf += self.oei

        for imode in range(self.nmode):
            vhf += scaled_mu * self.gmat[imode]
        vhf -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", self.gmat, self.gmat, dm)

        return vhf

    def dump_flags(self, verbose=None):
        return hf.RHF.dump_flags(self, verbose)

    def dse(self, dm):
        r"""
        compute dipole self-energy
        """
        dip = self.dip_moment(dm=dm)
        # print("dipole_moment=", dip)
        e_dse = 0.0
        e_dse += 0.5 * self.mu_lambda * self.mu_lambda

        print("dipole self-energy=", e_dse)
        return e_dse

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        r"""Total QED Hartree-Fock energy, electronic part plus nuclear repulstion
        See :func:`scf.hf.energy_elec` for the electron part

        Note this function has side effects which cause mf.scf_summary updated.
        """

        nuc = self.energy_nuc()
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc
        e_tot += 0.5 * numpy.einsum("pq,pq->", self.oei, dm)
        dse = self.dse(dm)  # dipole sefl-energy
        e_tot += dse
        self.scf_summary["nuc"] = nuc.real
        return e_tot

    """
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        #print("debug-zy: qed get_jk")
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        # add photon contribution, not done yet!!!!!!! (todo)
        # update molecular-cavity couplings
        vp = numpy.zeros_like(vj)

        vj += vp
        vk += vp
        return vj, vk
    """


class RKS(rks.KohnShamDFT, RHF):
    def __init__(self, mol, xc="LDA,VWN", **kwargs):
        RHF.__init__(self, mol, **kwargs)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        RHF.dump_flags(self, verbose)
        return rks.KohnShamDFT.dump_flags(self, verbose)

    get_veff = rks.get_veff
    get_vsap = rks.get_vsap
    energy_elec = rks.energy_elec


"""
class RKS(rks.RKS):

    def __init__(self, mol, xc=None, **kwargs):
        print("debug- DFT driver is used!")
        print("xc=", xc)
        rks.RKS.__init__(self, mol, xc=xc)

        cavity = None
        if "cavity" in kwargs:
            cavity = kwargs['cavity']
            if "cavity_mode" in kwargs:
                cavity_mode = kwargs['cavity_mode']
            else:
                raise ValueError("The required keyword argument 'cavity_mode' is missing")

            if "cavity_freq" in kwargs:
                cavity_freq = kwargs['cavity_freq']
            else:
                raise ValueError("The required keyword argument 'cavity_freq' is missing")

            print('cavity_freq=', cavity_freq)
            print('cavity_mode=', cavity_mode)

        print(f"{cavity} cavity mode is used!")

    def dump_flags(self, verbose=None):
        return rks.RKS.dump_flags(self, verbose)

"""


# this eventually will moved to qedscf/rhf class
#

"""
def get_veff(mf, dm, dm_last=None):
  veff = None

  return veff
"""


def qedrhf(model, options):
    # restricted qed hf
    # make a copy for qedhf
    mf = copy.copy(model.mf)
    conv_tol = 1.0e-10
    conv_tol_grad = None
    dump_chk = False
    callback = None
    conv_check = False
    noscf = False
    if "noscf" in options:
        noscf = options["noscf"]

    # converged bare HF coefficients
    na = int(mf.mo_occ.sum() // 2)
    ca = mf.mo_coeff
    dm = 2.0 * numpy.einsum("ai,bi->ab", ca[:, :na], ca[:, :na])
    mu_ao = model.dmat

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
    mol = mf.mol

    # initial guess
    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)

    e_tot = mf.energy_tot(dm, h1e, vhf)
    nuc = mf.energy_nuc()
    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None
    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    print("converged Tr[D]=", numpy.trace(dm) / 2.0)
    nmode = model.vec.shape[0]

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        # fock = mf.get_fock(h1e, s1e, vhf, dm)
        fock = h1e + vhf

        """
      fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
      """

        mu_mo = lib.einsum("pq, Xpq ->X", dm, mu_ao)

        scaled_mu = 0.0
        mu_lambda = 0.0
        for imode in range(nmode):
            mu_lambda -= numpy.dot(mu_mo, model.vec[imode])
            scaled_mu += numpy.einsum("pq, pq ->", dm, model.gmat[imode])

        dse = 0.5 * mu_lambda * mu_lambda

        # oei = numpy.zeros((h1e.shape[0], h1e.shape[1]))
        oei = model.gmat * mu_lambda
        oei -= model.qd2
        oei = numpy.sum(oei, axis=0)
        fock += oei

        #  <>
        for imode in range(nmode):
            fock += scaled_mu * model.gmat[imode]
        fock -= 0.5 * numpy.einsum("Xpr, Xqs, rs -> pq", model.gmat, model.gmat, dm)

        # e_tot = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse
        e_tot = 0.5 * numpy.einsum("pq,pq->", (oei + h1e + fock), dm) + nuc + dse

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        # factor of 2 is applied (via mo_occ)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)

        """
      # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
      # instead of the statement "fock = h1e + vhf" because Fock matrix may
      # be modified in some methods.

      fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
      oei = model.gmat * model.gmat
      oei -= model.qd2
      oei = numpy.sum(oei, axis=0)

      fock += oei

      mu_mo = lib.einsum('pq, Xpq ->X', 2 * dm, mu_ao)
      mu_lambda = 0.0
      scaled_mu = 0.0
      for imode in range(nmode):
          mu_lambda += -numpy.dot(mu_mo, model.vec[imode])
          scaled_mu += numpy.einsum('pq, pq ->', dm, model.gmat[imode])
      dse = 0.5 * mu_lambda * mu_lambda

      for imode in range(nmode):
          fock += 2 * scaled_mu * model.gmat[imode]
      fock -= numpy.einsum('Xpr, Xqs, rs -> pq', model.gmat, model.gmat, dm)
      """

        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)
        print(
            "cycle= %3d E= %.12g  delta_E= %4.3g |g|= %4.3g |ddm|= %4.3g |d*u|= %4.3g dse= %4.3g"
            % (cycle + 1, e_tot, e_tot - last_hf_e, norm_gorb, norm_ddm, mu_lambda, dse)
        )
        if noscf:
            break

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        if scf_conv:
            break

    # to be updated
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        # fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        # e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, fock - h1e + oei) + dse, e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        print(
            "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
            e_tot,
            e_tot - last_hf_e,
            norm_gorb,
            norm_ddm,
        )
        if dump_chk:
            mf.dump_chk(locals())

    # A post-processing hook before return
    mf.post_kernel(locals())
    print("HOMO-LUMO gap=", mo_energy[na] - mo_energy[na - 1])
    print("QEDHF energy=", e_tot)

    return scf_conv, e_tot, dse, mo_energy, mo_coeff, mo_occ


if __name__ == "__main__":
    # will add a loop
    import numpy
    from pyscf import gto, scf

    itest = 1
    zshift = itest * 2.0

    mol = gto.M(
        atom=f"H          0.86681        0.60144        {5.00000+zshift};\
        F         -0.86681        0.60144        {5.00000+zshift};\
        O          0.00000       -0.07579        {5.00000+zshift};\
        He         0.00000        0.00000        {7.50000+zshift}",
        basis="cc-pvdz",
        unit="Angstrom",
        symmetry=True,
        verbose=3,
    )
    print("mol coordinates=\n", mol.atom_coords())

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.0e-8
    hf.diis_space = 10
    hf.polariton = True
    mf = hf.run(verbose=4)

    print("electronic energies=", mf.energy_elec())
    print("nuclear energy=     ", mf.energy_nuc())
    dm = mf.make_rdm1()

    print("\n=========== QED-HF calculation  ======================\n")

    from qed.scf import hf as qedhf

    nmode = 1
    cavity_freq = numpy.zeros(nmode)
    cavity_mode = numpy.zeros((nmode, 3))
    cavity_freq[0] = 0.5
    cavity_mode[0, :] = 0.05 * numpy.asarray([0, 0, 1])

    qedmf = qedhf.RHF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qedmf.max_cycle = 500
    qedmf.kernel(dm0=dm)
