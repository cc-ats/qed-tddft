import numpy
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy import utils
from pyscf import lib
einsum = lib.einsum

class SSHModel(object):

    def __init__(self, L, N, U, t, K, alpha, M, ca=None, cb=None, V=None):
        self.L = L
        self.N = N
        self.U = U
        self.t = t
        self.K = K
        self.alpha = alpha
        self.M = M # MASS
        self.V = V
        na = N//2
        nb = N//2
        if ca is None or cb is None:
            ca, cb = self.get_mo()
        self.ca, self.cb = ca, cb
        self.pa = numpy.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])

        self.pb = numpy.einsum('ai,bi->ab',cb[:,:nb],cb[:,:nb])            

    def get_mo(self):
        from pyscf import scf, gto
        def get_jk(mol, dm, *args, **kwargs):
            nsite = self.L
            eri = numpy.zeros([nsite,nsite,nsite,nsite])
            for i in range(nsite):
                for j in range(nsite):
                    for k in range(nsite):
                        l = numpy.mod(i+k-j+nsite,nsite)
                        eri[i,j,k,l] = self.U * 1.0 / nsite
            vj = numpy.einsum('ijkl,slk->sij', eri, dm)
            vk = numpy.einsum('ilkj,slk->sij', eri, dm)
            return vj, vk
        L = self.L
        N = self.N
        mol = gto.M()
        mol.nelectron = N
        mol.incore_anyway = True
        mol.verbose = 0
        mf = scf.UHF(mol)
        h1 = self.tmatS()
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: numpy.eye(L)
        mf.get_jk = get_jk
        mf.conv_tol = 1e-10
        e,v = numpy.linalg.eigh(h1)
        ca = v.copy()
        cb = v.copy()
        ca[:,N//2 - 1] = 1.0/numpy.sqrt(2.0)*(ca[:,N//2 - 1] + ca[:,N//2])
        cb[:,N//2 - 1] = 1.0/numpy.sqrt(2.0)*(cb[:,N//2 - 1] - cb[:,N//2])
        dma = numpy.einsum('pq,rq->pr', ca[:,:N//2], ca[:,:N//2].conj())
        dmb = numpy.einsum('pq,rq->pr', cb[:,:N//2], cb[:,:N//2].conj())
        mf.kernel(dm0=(dma,dmb))
        if not mf.converged:
            print("warning!!reference SCF not converged, please rerun")
            exit()
        return mf.mo_coeff[0], mf.mo_coeff[1]

    def fci(self, nphonon=8,**kwargs):
        import fci
        nsite = self.L
        nelec = self.N
        nmode = nsite - 1
        t = self.tmatS()
        hpp = self.omega()
        u = self.U
        gmat = self.gmat()[0][:,:nsite,:nsite].transpose(1,2,0)
        return fci.kernel(t,u,gmat,hpp,nsite,nelec,nmode,nphonon,space='k',**kwargs)
        

    def tmatS(self):
        """ Return T-matrix in the spatial orbital basis."""
        L = self.L
        kvec = numpy.arange(L) * 2.0 * numpy.pi / L
        ene = - 2 * numpy.cos(kvec) * self.t
        t = numpy.zeros((L,L))
        numpy.fill_diagonal(t, ene)
        
        return t

    def umatS(self):
        """ Return U-matrix (not antisymmetrized) in the spatial orbital basis."""
        L = self.L
        umat = numpy.zeros((L,L,L,L))
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    l = numpy.mod(i+j-k+L,L)
                    umat[i,j,k,l] = self.U * 1.0
        umat /=L
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
            for j in range(L):
                for k in range(L):
                    l = numpy.mod(i+j+L-k,L)
                    umat[i,j,k,l] = self.U
                    umat[i,j+L,k,l+L] = self.U
                    umat[i+L,j,k+L,l] = self.U
                    umat[i+L,j+L,k+L,l+L] = self.U
        umat /= 1.0*L
        return umat


    def energies(self):
        fock = self.g_fock()
        return fock.oo.diagonal(), fock.vv.diagonal()

    def fock(self):
        if self.pa is None or self.pb is None:
            raise Exception("Cannot build Fock without density ")
        ptot = utils.block_diag(self.pa,self.pb)
        U = self.umat()
        T = self.tmat()
        Ua = U - U.transpose((0,1,3,2))
        JK = numpy.einsum('prqs,rs->pq',Ua,ptot)
        return T + JK

    def hf_energy(self):
        F = self.fock()
        T = self.tmat()
        ptot = utils.block_diag(self.pa,self.pb)
        Ehf = numpy.einsum('ij,ji->',ptot,F)
        Ehf += numpy.einsum('ij,ji->',ptot,T)
        return 0.5*Ehf

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
        F = numpy.asarray(self.fock(), dtype=numpy.complex128)
        Foo = numpy.einsum('pi,pq,qj->ij',Co,F,Co)
        Fov = numpy.einsum('pi,pq,qa->ia',Co,F,Cv)
        Fvo = numpy.einsum('pa,pq,qi->ai',Cv,F,Co)
        Fvv = numpy.einsum('pa,pq,qb->ab',Cv,F,Cv)
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def g_int_tot(self):
        V = self.umat()
        C = utils.block_diag(self.ca,self.cb)
        V = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,C,C,C,C)
        return V

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
        Ua_mo = einsum('pqrs,pw,qx,ry,sz->wxyz',Ua,C,C,C,C)
        Ua_mo = numpy.asarray(Ua_mo, dtype=numpy.complex128)
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
        K = self.K
        M = self.M
        L = self.L
        qvec = numpy.arange(1,L) * 2 * numpy.pi / L
        X = 2.0 * numpy.sqrt(K/M) * numpy.sin(qvec/2)
        return X
    
    def gmat(self):
        L = self.L
        M = self.M
        n = 2*self.L
        alpha = self.alpha
        omega = self.omega()
        g = numpy.zeros((L-1,n,n), dtype=numpy.complex128)
        for k1 in range(L):
            for k2 in range(L):
                if k1==k2:
                    continue
                q = numpy.mod(L+k1-k2,L) - 1
                k1vec = 2.0 * numpy.pi / L * k1
                k2vec = 2.0 * numpy.pi / L * k2
                g[q,k1,k2] = 1.0j * alpha * numpy.sqrt(2.0/L/M/omega[q]) * (numpy.sin(k1vec) - numpy.sin(k2vec))
                g[q,k1+L,k2+L] = g[q,k1,k2]
        h = g.transpose(0,2,1).conj()
        return g, h

    def mfG(self):
        ptot = utils.block_diag(self.pa,self.pb)
        g, h = self.gmat()
        mfG = numpy.einsum('Ipq,qp->I',g,ptot)
        mfH = numpy.einsum('Ipq,qp->I',h,ptot)
        return (mfG,mfH)

    def gint(self):
        g, h = self.gmat()
        na = self.N//2
        nb = self.N//2
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        oo = numpy.einsum('Ipq,pi,qj->Iij',g,Co,Co)
        ov = numpy.einsum('Ipq,pi,qa->Iia',g,Co,Cv)
        vo = numpy.einsum('Ipq,pa,qi->Iai',g,Cv,Co)
        vv = numpy.einsum('Ipq,pa,qb->Iab',g,Cv,Cv)

        hoo = numpy.einsum('Ipq,pi,qj->Iij',h,Co,Co)
        hov = numpy.einsum('Ipq,pi,qa->Iia',h,Co,Cv)
        hvo = numpy.einsum('Ipq,pa,qi->Iai',h,Cv,Co)
        hvv = numpy.einsum('Ipq,pa,qb->Iab',h,Cv,Cv)
        g = one_e_blocks(oo,ov,vo,vv)
        h = one_e_blocks(hoo,hov,hvo,hvv) 
        return (g,h)

    def gint_tot(self):
        raise NotImplementError
        n = 2*self.L
        g = numpy.zeros((self.M,n,n))
        if self.gij is not None:
            g.fill(self.gij)
        for i in range(self.M):
            g[i,numpy.arange(n),numpy.arange(n)] = self.g
        na = self.N//2
        nb = self.N//2
        Ctot = utils.block_diag(self.ca,self.cb)
        gmo = numpy.einsum('Ipq,pi,qj->Iij',g,Ctot,Ctot)
        return gmo
