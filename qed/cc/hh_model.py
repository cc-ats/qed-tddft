import numpy
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy import utils

def get_mo(L, N, U, breaksym, bc='p'):
    from pyscf import scf, gto, ao2mo
    mol = gto.M(verbose=0)
    mol.nelectron = N
    mol.incore_anyway = True
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    h1 = numpy.zeros([L,L])
    for i in range(L-1):
        h1[i,i+1] = h1[i+1,i] = -1
    if bc=='p':
        h1[-1,0] += -1
        h1[0,-1] += -1
    eri = numpy.zeros((L,L,L,L))
    assert(N % 2 == 0)
    na = nb = N//2
    pa0 = numpy.zeros((L,L))
    pb0 = numpy.zeros((L,L))
    if breaksym:
        for i in range(L):
            eri[i,i,i,i] = U
            if i % 2 == 0:
                pa0[i,i] = 1.0
            else:
                pb0[i,i] = 1.0
    else:
        for i in range(L): pa0[i,i] = pb0[i,i] = 0.5
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(L)
    mf._eri = ao2mo.restore(8, eri, L)
    mf.kernel(dm0 = [pa0,pb0])
    dm = mf.make_rdm1()
    return mf.mo_coeff[0], mf.mo_coeff[1]

class HHModel(object):
    def __init__(self, L, M, N, U, w, g=None, lam=None, bc='p', ca=None, cb=None, gij=None, shift=True, breaksym=True):
        self.L = L
        assert(M == L)
        self.M = M
        self.N = N
        self.U = U
        self.w = w
        self.bc = bc
        na = N//2
        nb = N//2
        if ca is None or cb is None:
            ca, cb = get_mo(L,N,U,breaksym,bc)
        self.ca, self.cb = ca, cb
        self.pa = numpy.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])
        self.pb = numpy.einsum('ai,bi->ab',cb[:,:nb],cb[:,:nb])
        self.ptot = utils.block_diag(self.pa,self.pb)
        self.g  = g
        self.gij = gij
        self.gmat = numpy.zeros((M,2*L,2*L))
        if gij is not None:
            idx = numpy.arange(L-1)
            self.gmat[:,idx+1,idx] = self.gmat[:,idx,idx+1] = gij
            self.gmat[:,idx+1+L,idx+L] = self.gmat[:,idx+L,idx+1+L] = gij
            if bc=='p':
                self.gmat[:,0,L-1] = self.gmat[:,L-1,0] = gij
                self.gmat[:,L,-1] = self.gmat[:,-1,L] = gij
        for i in range(L):
            self.gmat[i,i,i] = self.gmat[i,i+L,i+L] = g
        self.xi = numpy.einsum('Iab,ab->I', self.gmat, self.ptot) / self.w
        omega = numpy.zeros([M])
        omega.fill(self.w)
        self.const = - numpy.einsum('I,I->',omega,self.xi**2)
        self.shift = shift
        if not shift:
            self.const = 0
            self.xi[:] = 0


    def fci(self, nphonon=20, nroots=1):
        from .fci import kernel
        t = self.tmatS()
        u = self.U
        w = numpy.ones([self.M])*self.w
        N = self.N
        L = self.L
        gmat = self.gmat[:,:L,:L].transpose(1,2,0)
        M = self.M
        e,c = kernel(t,u,gmat,w,L,N,M,nphonon,space='r',verbose=4,nroots=nroots)
        return e, c

    def _nn(self,i):
        assert(i <= self.L)
        L = self.L
        if self.bc == "p":
            l = L - 1 if i == 0 else i - 1
            r = 0 if i == (L - 1) else i + 1
        elif self.bc is None:
            l = i+1 if i == 0 else i - 1
            r = i-1 if i == (L - 1) else i + 1
        else:
            raise Exception("Unrecognized boundary conditions")
        return (l,r)

    def tmatS(self):
        """ Return T-matrix in the spatial orbital basis."""
        L = self.L
        t = numpy.zeros((L,L))
        for i in range(L):
            nn = self._nn(i)
            t[i,nn[0]] = -1.0
            t[i,nn[1]] = -1.0
        if L == 2:
            t[0,-1] += -1
            t[-1,0] += -1
        return t

    def umatS(self):
        """ Return U-matrix (not antisymmetrized) in the spatial orbital basis."""
        L = self.L
        umat = numpy.zeros((L,L,L,L))
        for i in range(L):
            umat[i,i,i,i] = self.U
        return umat

    def tmat(self):
        """ Return T-matrix in the spin orbital basis."""
        t = self.tmatS()
        return utils.block_diag(t,t)

    def umat(self):
        """ Return U-matrix (not antisymmetrized) in the spin orbital basis."""
        L = self.L
        umat = numpy.zeros((2*L,2*L,2*L,2*L))
        for i in range(L):
            umat[i,L+i,i,L+i] = self.U
            umat[L+i,i,L+i,i] = self.U
            umat[i,i,i,i] = self.U
            umat[L+i,L+i,L+i,L+i] = self.U
        return umat

    def fock(self):
        if self.pa is None or self.pb is None:
            raise Exception("Cannot build Fock without density ")
        ptot = utils.block_diag(self.pa,self.pb)
        U = self.umat()
        T = self.tmat()
        Ua = U - U.transpose((0,1,3,2))
        JK = numpy.einsum('prqs,rs->pq',Ua,ptot)
        return T + JK

    def energies(self):
        f = self.g_fock()
        return f.oo.diagonal(), f.vv.diagonal()

    def hf_energy(self):
        F = self.fock()
        T = self.tmat()
        ptot = utils.block_diag(self.pa,self.pb)
        Ehf = numpy.einsum('ij,ji->',ptot,F)
        Ehf += numpy.einsum('ij,ji->',ptot,T)
        return 0.5*Ehf + self.const

    def u_hcore_tot(self):
        T = self.tmatS()
        ha = numpy.einsum('ij,ip,jq->pq',T,self.ca,self.ca)
        hb = numpy.einsum('ij,ip,jq->pq',T,self.cb,self.cb)
        return ha,hb

    def g_hcore_tot(self):
        T = self.tmatS()
        ha = numpy.einsum('ij,ip,jq->pq',T,self.ca,self.ca)
        hb = numpy.einsum('ij,ip,jq->pq',T,self.cb,self.cb)
        return utils.block_diag(ha,hb)

    def g_fock(self):
        na = self.N//2
        nb = self.N//2
        va = self.L - na
        vb = self.L - nb
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        F = self.fock()
        Foo = numpy.einsum('pi,pq,qj->ij',Co,F,Co) - 2*numpy.einsum('I,pi,Ipq,qj->ij',self.xi, Co, self.gmat, Co)
        Fov = numpy.einsum('pi,pq,qa->ia',Co,F,Cv) - 2*numpy.einsum('I,pi,Ipq,qa->ia',self.xi, Co, self.gmat, Cv)
        Fvo = numpy.einsum('pa,pq,qi->ai',Cv,F,Co) - 2*numpy.einsum('I,pa,Ipq,qi->ai',self.xi, Cv, self.gmat, Co)
        Fvv = numpy.einsum('pa,pq,qb->ab',Cv,F,Cv) - 2*numpy.einsum('I,pa,Ipq,qb->ab',self.xi, Cv, self.gmat, Cv)
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def g_aint(self):
        na = self.N//2
        nb = self.N//2
        va = self.L - na
        vb = self.L - nb
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        C = utils.block_diag(self.ca,self.cb)
        U = self.umat()
        Ua = U - U.transpose((0,1,3,2))
        Uat = numpy.einsum('pqrs,pw->wqrs',Ua,C)
        Uat = numpy.einsum('wqrs,qx->wxrs',Uat,C)
        Uat = numpy.einsum('wxrs,ry->wxys',Uat,C)
        Ua_mo = numpy.einsum('wxys,sz->wxyz',Uat,C)
        temp = [i for i in range(2*self.L)]
        oidx = temp[:na] + temp[self.L:self.L + nb]
        vidx = temp[na:self.L] + temp[self.L + nb:]

        vvvv = Ua_mo[numpy.ix_(vidx,vidx,vidx,vidx)]
        vvvo = Ua_mo[numpy.ix_(vidx,vidx,vidx,oidx)]
        vovv = Ua_mo[numpy.ix_(vidx,oidx,vidx,vidx)]
        vvoo = Ua_mo[numpy.ix_(vidx,vidx,oidx,oidx)]
        oovv = Ua_mo[numpy.ix_(oidx,oidx,vidx,vidx)]
        vovo = Ua_mo[numpy.ix_(vidx,oidx,vidx,oidx)]
        vooo = Ua_mo[numpy.ix_(vidx,oidx,oidx,oidx)]
        ooov = Ua_mo[numpy.ix_(oidx,oidx,oidx,vidx)]
        oooo = Ua_mo[numpy.ix_(oidx,oidx,oidx,oidx)]
        return two_e_blocks(vvvv=vvvv,
                vvvo=vvvo, vovv=vovv,
                vvoo=vvoo, oovv=oovv,
                vovo=vovo, vooo=vooo,
                ooov=ooov, oooo=oooo)

    def omega(self):
        X = numpy.zeros(self.M)
        X.fill(self.w)
        return X

    def mfG(self):
        ptot = utils.block_diag(self.pa,self.pb)
        g = self.gmat
        mfG = numpy.einsum('Ipq,qp->I',g,ptot)
        if self.shift:
            mfG[:] = 0
        return (mfG,mfG)

    def gint(self):
        n = 2*self.L
        g = self.gmat
        na = self.N//2
        nb = self.N//2
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        oo = numpy.einsum('Ipq,pi,qj->Iij',g,Co,Co)
        ov = numpy.einsum('Ipq,pi,qa->Iia',g,Co,Cv)
        vo = numpy.einsum('Ipq,pa,qi->Iai',g,Cv,Co)
        vv = numpy.einsum('Ipq,pa,qb->Iab',g,Cv,Cv)
        g = one_e_blocks(oo,ov,vo,vv)
        return (g,g)

    def gint_tot(self):
        n = 2*self.L
        g = self.gmat
        na = self.N//2
        nb = self.N//2
        Ctot = utils.block_diag(self.ca, self.cb)
        gmo = numpy.einsum('Ipq,pi,qj->Iij',g,Ctot,Ctot)
        return gmo
