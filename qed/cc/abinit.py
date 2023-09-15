from pyscf import scf, gto, dft
import sys
import numpy
import numpy as np
import scipy
from cqcpy import utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks, two_e_blocks_full
from . eom_epcc import eom_ee_epccsd_1_s1

au_to_cm = 2.19475 * 1e5

def make_mf(mol, method, xc=None, RUN=True):
    """construct a mean field object from the given molecule and specified method"""
    print("""construct a mean field object from the given molecule and specified method""", RUN,method)
    if method == 'rhf':
        mf = scf.RHF(mol)
    elif method =='uhf':
        mf = scf.UHF(mol)
    elif method=='rks':
        mf = scf.RKS(mol)
        mf.grids.level = 4
    elif method=='uks':
        mf = scf.UKS(mol)
        mf.grids.level = 4
    else:
        raise ValueError
    if xc is not None:
        mf.xc = xc
    mf.max_cycle = 200
    mf.conv_tol = 1e-14
    if RUN:
        #print(xc)
        mf.verbose=0
        mf.kernel()
    return mf

def relax(mol, method, xc, conv_params):
    """Perform Geometry Optimization on the given molecule with geoMETRIC"""
    from pyscf.geomopt.geometric_solver import GeometryOptimizer
    mf = make_mf(mol, method, xc)
    mf.verbose=0
    opt = GeometryOptimizer(mf).set(params=conv_params)
    opt.verbose=0
    opt.max_cycle = 200
    opt.kernel()
    mol_eq =opt.mol
    return mol_eq

def gen_moles(mol, disp):
    """From the given molecule, generate molecules with a shift on + disp/2 and -disp/2 on each Cartesian coordinates"""
    coords = mol.atom_coords()
    natoms = len(coords)
    mol_a, mol_s, coords_a, coords_s = [],[],[],[]
    for i in range(natoms):
        for x in range(3):
            new_coords_a, new_coords_s = coords.copy(), coords.copy()
            new_coords_a[i][x] += disp
            new_coords_s[i][x] -= disp
            coords_a.append(new_coords_a)
            coords_s.append(new_coords_s)
    nconfigs = 3*natoms
    for i in range(nconfigs):
        mola = gto.M()
        mols = gto.M()
        atoma, atoms = [], []
        for j in range(natoms):
            atoma.append([mol.atom_symbol(j), coords_a[i][j]])
            atoms.append([mol.atom_symbol(j), coords_s[i][j]])
        mola.atom = atoma
        mols.atom = atoms
        mola.unit = mols.unit = 'Bohr'
        mola.basis = mols.basis = mol.basis
        mola.verbose=mols.verbose=0
        mola.build()
        mols.build()
        mol_a.append(mola)
        mol_s.append(mols)

    return mol_a, mol_s

def get_dmat(mol, method, xc, disp):
    # get dip matrix
    dmat = None
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = np.einsum('i,ix->x', charges, coords)
    with mol.with_common_orig(charge_center):
        dmat = mol.intor_symmetric('int1e_r', comp=3)
    return dmat

def get_qmat(mol):
    # | xx, xy, xz |
    # | yx, yy, yz |
    # | zx, zy, zz |
    # xx <-> rrmat[0], xy <-> rrmat[3], xz <-> rrmat[6]
    #                  yy <-> rrmat[4], yz <-> rrmat[7]
    #                                   zz <-> rrmat[8]

    qmat = None
    rrmat = None
    #r2mat = None
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = np.einsum('i,ix->x', charges, coords)
    with mol.with_common_orig(charge_center):
        qmat  = -mol.intor('int1e_rr')
    return qmat

class abinit(object):
    def __init__(self, mol, method, xc=None, omega=None, vec=None, gfac=0.0):
        self.mol = mol
        self.method = method
        self.xc = xc
        self.conv_params = {
        'convergence_energy': 1e-6,  # Eh
        'convergence_grms': 1e-5,    # Eh/Bohr
        'convergence_gmax': 1e-5,  # Eh/Bohr
        'convergence_drms': 1e-4,  # Angstrom
        'convergence_dmax': 1e-4,  # Angstrom
        }
        self.dmat = None
        self.qmat = None
        self.omega = omega
        self.vec = vec
        self.gfac = gfac
        self.disp= 1e-5
        self.gmat= None
        self.mo_coeff = None
        self.relaxed = False

    def relax(self, method=None, xc=None, conv_params=None):
        if method is None: method = self.method
        if xc is None: xc = self.xc
        if conv_params is None: conv_params = self.conv_params
        self.mol = relax(self.mol, method, xc, conv_params)
        self.relaxed = True

    def get_dmat(self, disp=None):
        self.dmat = get_dmat(self.mol, self.method, self.xc, disp)
        return self.dmat

    def get_gmat(self):
        if self.omega is None:
            print('warning, omega is not given')
        if self.dmat is None: self.get_dmat()
        if self.qmat is None: self.qmat = get_qmat(self.mol)

        omega, vec, dmat = self.omega, self.vec, self.dmat
        nmode = len(omega)

        # normalize vec
        for i in range(nmode):
            vec[i,:] = vec[i,:] / np.sqrt(np.dot(vec[i,:], vec[i,:]))
        self.vec = vec

        nao, nmode = self.mol.nao_nr(), len(omega)

        # Tensor:  <u|r_i * r_y> * v_x * v_y
        x_out_y = 0.5* np.outer(vec, vec).reshape(-1)
        self.qd2 = self.gfac * self.gfac * np.einsum("J, Juv->uv", x_out_y, self.qmat)

        gmat = np.empty((nmode, nao, nao))
        gmat = np.einsum('Jx,xuv->Juv', vec, dmat) * self.gfac
        #print('dmat=\n', dmat)
        #print('\n|gmat|=', np.linalg.norm(gmat))

        self.gmat = gmat
        return gmat

    kernel = get_gmat

class Model(object):
    def __init__(self, mol, method, xc, omega=None, vec=None, gfac=0.0, ca=None, cb=None, shift=True):
        '''
        mol: unrelaxed molecule object
        method: a string denoting the mean field method for extracting the Hamiltonian, can be 'rhf', 'uhf', 'rks', 'uks'
        xc: a string denoting the exchange correlation functional for extracting Hamiltonian
        ca, cb : mo coefficent for EPH-CC reference
        '''
        self._mol = mol
        self.verbose = mol.verbose
        self.method = method
        self.xc = xc
        #self.openshell = False
        #if method == 'uhf' or method == 'uks':
        #    self.openshell = True

        self.abinit = abinit(mol, method, xc, omega, vec, gfac)

        self.abinit.kernel() # extracting Hamiltonian
        self.mol = self.abinit.mol #update to relaxed molecule
        self.dmat = self.abinit.dmat

        """
        self.nmo = self.mol.nao_nr()
        if self.openshell:
            self.nmo = 2*self.mol.nao_nr()

        if self.openshell:
            if ca is None or cb is None:
                mf = make_mf(self.mol, 'uhf')
                self.energy_nuc = mf.energy_nuc()
                #mf = make_mf(self.mol, method, xc)
                ca, cb = mf.mo_coeff
                na, nb = int(mf.mo_occ[0].sum()), int(mf.mo_occ[1].sum())
                self.na, self.nb = na, nb
                self.mf = mf

            self.ca, self.cb = ca, cb
            self.pa = np.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])
            self.pb = np.einsum('ai,bi->ab',cb[:,:nb],cb[:,:nb])
            self.ptot = utils.block_diag(self.pa,self.pb)
        else:
            mf = make_mf(self.mol, 'rhf')
            #mf = make_mf(self.mol, method, xc)
            self.energy_nuc = mf.energy_nuc()
            ca = mf.mo_coeff
            na = int(mf.mo_occ[0].sum())
            self.na = na
            self.mf = mf

            self.ca = ca
            self.pa = 2.0*np.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])
            self.ptot = self.pa
        """
        self.nmo = 2*self.mol.nao_nr()
        if ca is None or cb is None:
            mf = make_mf(self.mol, 'uhf')
            ca, cb = mf.mo_coeff
            na, nb = int(mf.mo_occ[0].sum()), int(mf.mo_occ[1].sum())
            mf = make_mf(self.mol, 'rhf')
            ca = cb = mf.mo_coeff
            na = nb = int(mf.mo_occ.sum()) // 2

            self.na, self.nb = na, nb
            self.energy_nuc = mf.energy_nuc()
        self.ca, self.cb = ca, cb
        self.pa = np.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])
        self.pb = np.einsum('ai,bi->ab',cb[:,:nb],cb[:,:nb])
        self.ptot = utils.block_diag(self.pa,self.pb)
        self.w = self.abinit.omega
        self.nmode = len(self.w)
        self.gmat = self.abinit.gmat #check this thing
        gmatso = [utils.block_diag(self.gmat[i], self.gmat[i]) for i in range(len(self.gmat))]
        """
        if self.openshell:
            gmatso = [utils.block_diag(self.gmat[i], self.gmat[i]) for i in range(len(self.gmat))]
        else:
            gmatso = [self.gmat[i] for i in range(len(self.gmat))]
        """
        self.gmatso = np.asarray(gmatso)
        self.shift = shift
        if shift:
            self.xi = np.einsum('Iab,ab->I', self.gmatso, self.ptot) / self.w
            self.const = - np.einsum('I,I->',self.w,self.xi**2)

    def tmat(self):
        """ Return T-matrix in the spin orbital basis."""
        t = self.mol.get_hcore()
        return utils.block_diag(t,t)

    def fock(self):
        from pyscf import scf
        if self.pa is None or self.pb is None:
            raise Exception("Cannot build Fock without density ")
        h1 = self.mol.get_hcore()
        ptot = utils.block_diag(self.pa,self.pb)
        h1 = utils.block_diag(h1,h1)
        myhf = scf.GHF(self.mol)
        fock = h1 + myhf.get_veff(self.mol, dm=ptot)
        return fock

    def hf_energy(self):
        F = self.fock()
        T = self.tmat()
        ptot = utils.block_diag(self.pa,self.pb)
        Ehf = np.einsum('ij,ji->',ptot,F)
        Ehf += np.einsum('ij,ji->',ptot,T)
        if self.shift:
            return 0.5*Ehf + self.energy_nuc + self.const
        else:
            return 0.5*Ehf + self.energy_nuc

    def energies(self):
        eo = self.mf.mo_energy[numpy.ix_(self.occ)]
        ev = self.mf.mo_energy[numpy.ix_(self.vir)]
        return numpy.concatenate((eo,eo)), numpy.concatenate((ev,ev))

    def g_fock(self):
        na, nb = self.na, self.nb
        va, vb = self.nmo//2 - na, self.nmo//2 - nb
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        #print('entering fock')
        F = self.fock()
        if self.shift:
            Foo = np.einsum('pi,pq,qj->ij',Co,F,Co) - 2*np.einsum('I,pi,Ipq,qj->ij',self.xi, Co, self.gmatso, Co)
            Fov = np.einsum('pi,pq,qa->ia',Co,F,Cv) - 2*np.einsum('I,pi,Ipq,qa->ia',self.xi, Co, self.gmatso, Cv)
            Fvo = np.einsum('pa,pq,qi->ai',Cv,F,Co) - 2*np.einsum('I,pa,Ipq,qi->ai',self.xi, Cv, self.gmatso, Co)
            Fvv = np.einsum('pa,pq,qb->ab',Cv,F,Cv) - 2*np.einsum('I,pa,Ipq,qb->ab',self.xi, Cv, self.gmatso, Cv)
        else:
            Foo = np.einsum('pi,pq,qj->ij',Co,F,Co)
            Fov = np.einsum('pi,pq,qa->ia',Co,F,Cv)
            Fvo = np.einsum('pa,pq,qi->ai',Cv,F,Co)
            Fvv = np.einsum('pa,pq,qb->ab',Cv,F,Cv)
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def g_aint(self):
        from pyscf import ao2mo
        na, nb = self.na, self.nb
        va, vb = self.nmo//2 - na, self.nmo//2 - nb
        nao = self.nmo//2
        C = np.hstack((self.ca, self.cb))
        eri = ao2mo.general(self.mol, [C,]*4, compact=False).reshape([self.nmo,]*4)
        eri[:nao,nao:] = eri[nao:,:nao] = eri[:,:,:nao,nao:] = eri[:,:,nao:,:nao] = 0
        Ua_mo = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
        temp = [i for i in range(self.nmo)]
        oidx = temp[:na] + temp[self.nmo//2:self.nmo//2 + nb]
        vidx = temp[na:self.nmo//2] + temp[self.nmo//2 + nb:]

        vvvv = Ua_mo[np.ix_(vidx,vidx,vidx,vidx)]
        vvvo = Ua_mo[np.ix_(vidx,vidx,vidx,oidx)]
        vvov = Ua_mo[np.ix_(vidx,vidx,oidx,vidx)]
        vovv = Ua_mo[np.ix_(vidx,oidx,vidx,vidx)]
        ovvv = Ua_mo[np.ix_(oidx,vidx,vidx,vidx)]
        vvoo = Ua_mo[np.ix_(vidx,vidx,oidx,oidx)]
        vovo = Ua_mo[np.ix_(vidx,oidx,vidx,oidx)]
        voov = Ua_mo[np.ix_(vidx,oidx,oidx,vidx)]
        ovvo = Ua_mo[np.ix_(oidx,vidx,vidx,oidx)]
        ovov = Ua_mo[np.ix_(oidx,vidx,oidx,vidx)]
        oovv = Ua_mo[np.ix_(oidx,oidx,vidx,vidx)]
        ooov = Ua_mo[np.ix_(oidx,oidx,oidx,vidx)]
        oovo = Ua_mo[np.ix_(oidx,oidx,vidx,oidx)]
        ovoo = Ua_mo[np.ix_(oidx,vidx,oidx,oidx)]
        vooo = Ua_mo[np.ix_(vidx,oidx,oidx,oidx)]
        oooo = Ua_mo[np.ix_(oidx,oidx,oidx,oidx)]

        return two_e_blocks_full(vvvv=vvvv,
                vvvo=vvvo, vovv=vovv,
                voov=voov, ovvv=ovvv,
                ovoo=ovoo, oovo=oovo,
                vvov=vvov, ovvo=ovvo,
                ovov=ovov,
                vvoo=vvoo, oovv=oovv,
                vovo=vovo, vooo=vooo,
                ooov=ooov, oooo=oooo)

    def omega(self):
        return self.w

    def mfG(self):
        ptot = utils.block_diag(self.pa,self.pb)
        g = self.gmatso
        if self.shift:
            mfG = np.zeros(self.nmode)
        else:
            mfG = np.einsum('Ipq,qp->I',g,ptot)
        return (mfG,mfG)

    def gint(self):
        g = self.gmatso.copy()
        na = self.na
        nb = self.nb
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])

        oo = np.einsum('Ipq,pi,qj->Iij',g,Co,Co)
        ov = np.einsum('Ipq,pi,qa->Iia',g,Co,Cv)
        vo = np.einsum('Ipq,pa,qi->Iai',g,Cv,Co)
        vv = np.einsum('Ipq,pa,qb->Iab',g,Cv,Cv)
        g = one_e_blocks(oo,ov,vo,vv)
        return (g,g)

if __name__ == '__main__':
    mol = gto.M()
    mol.atom = '''H 0 0 0; F 0 0 1.75202'''
    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build()

    method = 'rks'
    xc = 'b3lyp'
    #model = abinit(mol, method, xc)
    #gmat = model.get_gmat()
    #omega =model.omega

    nmode = 1
    gfac = 0.0#2
    omega = np.zeros(nmode)
    vec = np.zeros((nmode,3))
    omega[0] = 2.5
    vec[0,:] = [1.0, 1.0, 1.0]
    print('\n')
    mod = Model(mol, method, xc, omega, vec, gfac, shift=False)

    from polaritoncc import epcc
    options = {"ethresh":1e-8, 'tthresh':1e-7, 'max_iter':500, 'damp':0.4}
    ecc, e_cor, (T1,T2,S1,U11) = epcc(mod, options,ret=True)
    print('correlaiton energy (epcc)=', ecc)

    print('========== eom_ee_epccsd_1_s1 calculaiton ======')
    options = {"nroots":4, 'conv_tol':1e-7, 'max_cycle':500, 'max_space':1000}
    es = eom_ee_epccsd_1_s1(mod, options, amps=(T1,T2,S1,U11))
    print('correlaiton energy(epccsd_2_s1)=', es)
