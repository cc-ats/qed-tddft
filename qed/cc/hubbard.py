import numpy
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.ov_blocks import two_e_blocks_full
from cqcpy import utils

class Hubbard1D(object):
    def __init__(self, L, N, U, bc=None, ca=None, cb=None):
        self.L = L
        self.N = N
        self.U = U
        self.bc = bc 
        self.ca = ca
        self.cb = cb
        self.nb = nb = N//2
        self.na = na = N - nb
        if ca is not None:
            self.pa = numpy.einsum('ai,bi->ab',ca[:,:na],ca[:,:na])
        if cb is not None:
            self.pb = numpy.einsum('ai,bi->ab',cb[:,:nb],cb[:,:nb])

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
        return t

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

    def umatS(self):
        """ Return U-matrix (not antisymmetrized) in the spatial orbital basis."""
        L = self.L
        umat = numpy.zeros((L,L,L,L))
        for i in range(L):
            umat[i,i,i,i] = self.U
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
        na = self.na
        nb = self.nb
        va = self.L - na
        vb = self.L - nb
        Co = utils.block_diag(self.ca[:,:na],self.cb[:,:nb])
        Cv = utils.block_diag(self.ca[:,na:],self.cb[:,nb:])
        F = self.fock()
        Ctot = utils.block_diag(self.ca, self.cb)
        Foo = numpy.einsum('pi,pq,qj->ij',Co,F,Co)
        Fov = numpy.einsum('pi,pq,qa->ia',Co,F,Cv)
        Fvo = numpy.einsum('pa,pq,qi->ai',Cv,F,Co)
        Fvv = numpy.einsum('pa,pq,qb->ab',Cv,F,Cv)
        return one_e_blocks(Foo,Fov,Fvo,Fvv)

    def u_fock(self):
        na = self.na
        nb = self.nb
        temp = range(self.L)
        oa = temp[:na]
        ob = temp[:nb]
        va = temp[na:]
        vb = temp[nb:]
        oidxa = numpy.r_[oa]
        vidxa = numpy.r_[va]
        oidxb = numpy.r_[ob]
        vidxb = numpy.r_[vb]
        foa = numpy.zeros(self.L)
        fob = numpy.zeros(self.L)
        for i in range(self.na):
            foa[i] = 1.0
        for i in range(self.nb):
            fob[i] = 1.0
        Va,Vb,Vab = self.u_aint_tot()
        T = self.tmatS()
        Fa = numpy.einsum('ij,ip,jq->pq',T,self.ca,self.ca)
        Fb = numpy.einsum('ij,ip,jq->pq',T,self.cb,self.cb)
        Fa += numpy.einsum('pqrq,q->pr',Va,foa)
        Fa += numpy.einsum('pqrq,q->pr',Vab,fob)
        Fb += numpy.einsum('pqrq,q->pr',Vb,fob)
        Fb += numpy.einsum('pqps,p->qs',Vab,foa)
        Fooa = Fa[numpy.ix_(oidxa,oidxa)]
        Fova = Fa[numpy.ix_(oidxa,vidxa)]
        Fvoa = Fa[numpy.ix_(vidxa,oidxa)]
        Fvva = Fa[numpy.ix_(vidxa,vidxa)]
        Foob = Fb[numpy.ix_(oidxb,oidxb)]
        Fovb = Fb[numpy.ix_(oidxb,vidxb)]
        Fvob = Fb[numpy.ix_(vidxb,oidxb)]
        Fvvb = Fb[numpy.ix_(vidxb,vidxb)]
        Fa_blocks = one_e_blocks(Fooa,Fova,Fvoa,Fvva)
        Fb_blocks = one_e_blocks(Foob,Fovb,Fvob,Fvvb)
        return Fa_blocks,Fb_blocks

    def u_aint_tot(self):
        V = self.umatS()
        Va = V - V.transpose((0,1,3,2))
        ca = self.ca
        cb = self.cb
        Vabab = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,ca,cb,ca,cb)
        Vb = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',Va,cb,cb,cb,cb)
        Va = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',Va,ca,ca,ca,ca)
        return Va,Vb,Vabab

    def u_int_tot(self):
        V = self.umatS()
        ca = self.ca
        cb = self.cb
        Vabab = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,ca,cb,ca,cb)
        Vb = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,cb,cb,cb,cb)
        Va = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,ca,ca,ca,ca)
        return Va,Vb,Vabab

    def u_aint(self):
        na = self.na
        nb = self.nb
        temp = range(self.L)
        oa = temp[:na]
        ob = temp[:nb]
        va = temp[na:]
        vb = temp[nb:]
        oidxa = numpy.r_[oa]
        vidxa = numpy.r_[va]
        oidxb = numpy.r_[ob]
        vidxb = numpy.r_[vb]
        Va,Vb,Vabab = self.u_aint_tot()

        Vvvvv = Va[numpy.ix_(vidxa,vidxa,vidxa,vidxa)]
        Vvvvo = Va[numpy.ix_(vidxa,vidxa,vidxa,oidxa)]
        Vvovv = Va[numpy.ix_(vidxa,oidxa,vidxa,vidxa)]
        Vvvoo = Va[numpy.ix_(vidxa,vidxa,oidxa,oidxa)]
        Vvovo = Va[numpy.ix_(vidxa,oidxa,vidxa,oidxa)]
        Voovv = Va[numpy.ix_(oidxa,oidxa,vidxa,vidxa)]
        Vvooo = Va[numpy.ix_(vidxa,oidxa,oidxa,oidxa)]
        Vooov = Va[numpy.ix_(oidxa,oidxa,oidxa,vidxa)]
        Voooo = Va[numpy.ix_(oidxa,oidxa,oidxa,oidxa)]
        Va = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)
        Vvvvv = Vb[numpy.ix_(vidxb,vidxb,vidxb,vidxb)]
        Vvvvo = Vb[numpy.ix_(vidxb,vidxb,vidxb,oidxb)]
        Vvovv = Vb[numpy.ix_(vidxb,oidxb,vidxb,vidxb)]
        Vvvoo = Vb[numpy.ix_(vidxb,vidxb,oidxb,oidxb)]
        Vvovo = Vb[numpy.ix_(vidxb,oidxb,vidxb,oidxb)]
        Voovv = Vb[numpy.ix_(oidxb,oidxb,vidxb,vidxb)]
        Vvooo = Vb[numpy.ix_(vidxb,oidxb,oidxb,oidxb)]
        Vooov = Vb[numpy.ix_(oidxb,oidxb,oidxb,vidxb)]
        Voooo = Vb[numpy.ix_(oidxb,oidxb,oidxb,oidxb)]
        Vb = two_e_blocks(
            vvvv=Vvvvv,vvvo=Vvvvo,
            vovv=Vvovv,vvoo=Vvvoo,
            vovo=Vvovo,oovv=Voovv,
            vooo=Vvooo,ooov=Vooov,
            oooo=Voooo)

        Vvvvv = Vabab[numpy.ix_(vidxa,vidxb,vidxa,vidxb)]
        Vvvvo = Vabab[numpy.ix_(vidxa,vidxb,vidxa,oidxb)]
        Vvvov = Vabab[numpy.ix_(vidxa,vidxb,oidxa,vidxb)]
        Vvovv = Vabab[numpy.ix_(vidxa,oidxb,vidxa,vidxb)]
        Vovvv = Vabab[numpy.ix_(oidxa,vidxb,vidxa,vidxb)]
        Vvvoo = Vabab[numpy.ix_(vidxa,vidxb,oidxa,oidxb)]
        Vvoov = Vabab[numpy.ix_(vidxa,oidxb,oidxa,vidxb)]
        Vvovo = Vabab[numpy.ix_(vidxa,oidxb,vidxa,oidxb)]
        Vovvo = Vabab[numpy.ix_(oidxa,vidxb,vidxa,oidxb)]
        Vovov = Vabab[numpy.ix_(oidxa,vidxb,oidxa,vidxb)]
        Voovv = Vabab[numpy.ix_(oidxa,oidxb,vidxa,vidxb)]
        Vvooo = Vabab[numpy.ix_(vidxa,oidxb,oidxa,oidxb)]
        Vovoo = Vabab[numpy.ix_(oidxa,vidxb,oidxa,oidxb)]
        Voovo = Vabab[numpy.ix_(oidxa,oidxb,vidxa,oidxb)]
        Vooov = Vabab[numpy.ix_(oidxa,oidxb,oidxa,vidxb)]
        Voooo = Vabab[numpy.ix_(oidxa,oidxb,oidxa,oidxb)]
        Vabab = two_e_blocks_full(vvvv=Vvvvv,
                vvvo=Vvvvo,vvov=Vvvov,
                vovv=Vvovv,ovvv=Vovvv,
                vvoo=Vvvoo,vovo=Vvovo,
                ovvo=Vovvo,voov=Vvoov,
                ovov=Vovov,oovv=Voovv,
                vooo=Vvooo,ovoo=Vovoo,
                oovo=Voovo,ooov=Vooov,
                oooo=Voooo)
        return Va,Vb,Vabab

    def g_aint_tot(self):
        V = self.umat()
        C = utils.block_diag(self.ca,self.cb)
        V = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,C,C,C,C)
        return V - V.transpose((0,1,3,2))

    def g_int_tot(self):
        V = self.umat()
        C = utils.block_diag(self.ca,self.cb)
        V = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',V,C,C,C,C)
        return V

    def g_aint(self):
        nb = self.nb
        na = self.na
        C = utils.block_diag(self.ca,self.cb)
        U = self.umat()
        Ua = U - U.transpose((0,1,3,2))
        Ua_mo = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',Ua,C,C,C,C)
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
