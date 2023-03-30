import numpy
from cqcpy import cc_equations
from . import epcc_energy

#einsum = numpy.einsum
from pyscf import lib
einsum = lib.einsum


def eom_epccsd_1_s1_Wov(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wov = F.ov.copy()
    Wov += einsum('x,xia->ia', S1, g.ov)
    Wov -= einsum('bj,jiab->ia', T1, I.oovv)
    return Wov

def eom_epccsd_1_s1_Wvv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wvv = F.vv.copy()
    Xov = F.ov.copy()
    Xov += einsum('x,xib->ib', S1, g.ov)
    Wvv -= einsum('ai,ib->ab', T1, Xov)

    Wvv += einsum('x,xab->ab', S1, g.vv)
    Wvv += einsum('ci,aibc->ab', T1, I.vovv)
    Wvv -= einsum('xai,xib->ab', U11, g.ov)

    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Wvv += 0.5*einsum('acij,jibc->ab', T2B, I.oovv)
    return Wvv

def eom_epccsd_1_s1_Woo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Woo = F.oo.copy()
    Xov = F.ov.copy()
    Xov += einsum('x,xia->ia', S1, g.ov)
    Woo += einsum('aj,ia->ij', T1, Xov)

    Woo += einsum('x,xij->ij', S1, g.oo)
    Woo -= einsum('ak,kija->ij', T1, I.ooov)
    Woo += einsum('xaj,xia->ij', U11, g.ov)

    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Woo += 0.5*einsum('abjk,kiba->ij', T2B, I.oovv)
    return Woo

def eom_epccsd_1_s1_Woovv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Woovv = I.oovv
    return Woovv

def eom_epccsd_1_s1_Wvovv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wvovv = I.vovv.copy()
    Wvovv -= einsum('aj,jibc->aibc', T1, I.oovv)
    return Wvovv

def eom_epccsd_1_s1_Wooov(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wooov = I.ooov.copy()
    Wooov += einsum('bk,ijba->ijka', T1, I.oovv)
    return Wooov

def eom_epccsd_1_s1_Wvvvv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wvvvv = I.vvvv.copy()
    temp_ab = einsum('bi,aidc->abcd', T1, I.vovv)
    Wvvvv += temp_ab - temp_ab.transpose((1,0,2,3))
    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Wvvvv += 0.5*einsum('baij,ijdc->abcd', T2B, I.oovv)
    return Wvvvv

def eom_epccsd_1_s1_Woooo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Woooo = I.oooo.copy()
    temp_kl = einsum('al,ijka->ijkl', T1, I.ooov)
    Woooo += temp_kl - temp_kl.transpose((0,1,3,2))
    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Woooo += 0.5*einsum('abkl,jiba->ijkl', T2B, I.oovv)
    return Woooo

def eom_epccsd_1_s1_Wvoov(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wvoov = -I.vovo.transpose((0,1,3,2)).copy()
    Wvoov += einsum('xaj,xib->aijb', U11, g.ov)
    Wvoov -= einsum('ak,kijb->aijb', T1, I.ooov)
    Wvoov -= einsum('cj,aibc->aijb', T1, I.vovv)
    T2A = T2 + einsum('di,aj->adji', T1, T1)
    Wvoov += einsum('ackj,kibc->aijb', T2A, I.oovv)
    return Wvoov

def eom_epccsd_1_s1_Wvvvo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wvvvo = I.vvvo.copy()
    Wvvvo += einsum('di,abcd->abci', T1, I.vvvv)

    Xov = F.ov.copy()
    Xov += einsum('x,xkc->kc', S1, g.ov)
    Xov -= einsum('dj,jkcd->kc', T1, I.oovv)
    Wvvvo -= einsum('abki,kc->abci', T2, Xov)

    T2A = T2 + einsum('di,aj->adji', T1, T1)
    temp_ab = einsum('aj,bjci->abci', T1, I.vovo)
    temp_ab += einsum('xbi,xac->abci', U11, g.vv)
    temp_ab += einsum('adji,bjcd->abci', T2A, I.vovv)
    Xmvv = einsum('bj,xjc->xbc', T1, g.ov)
    temp_ab += einsum('xai,xbc->abci', U11, Xmvv)
    Xvovv = einsum('aj,jkcd->akcd', T1, I.oovv)
    temp_ab += einsum('bdki,akcd->abci', T2, Xvovv)
    Wvvvo += temp_ab - temp_ab.transpose((1,0,2,3))
    T2A = None

    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Xoovo = I.ooov.transpose((0,1,3,2)).copy()
    Xoovo -= einsum('di,jkcd->jkci', T1, I.oovv)
    Wvvvo += 0.5*einsum('bajk,jkci->abci', T2B, Xoovo)

    return Wvvvo

def eom_epccsd_1_s1_Wovoo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wovoo = -I.vooo.transpose((1,0,2,3)).copy()
    Wovoo += -1.0*einsum('al,likj->iajk', T1, I.oooo)

    Xov = F.ov.copy()
    Xov += einsum('x,xib->ib', S1, g.ov)
    Xov -= einsum('bl,licb->ic', T1, I.oovv)
    Wovoo += einsum('abkj,ib->iajk', T2, Xov)

    T2A = T2 + einsum('di,aj->adji', T1, T1)
    temp_jk = einsum('bk,aibj->iajk', T1, I.vovo)
    temp_jk += einsum('xak,xij->iajk', U11, g.oo)
    Xmoo = einsum('bj,xib->xij', T1, g.ov)
    temp_jk += einsum('xak,xij->iajk', U11, Xmoo)
    temp_jk += einsum('ablk,lijb->iajk', T2A, I.ooov)
    Xoovo = einsum('bk,licb->lick', T1, I.oovv)
    temp_jk += einsum('aclj,lick->iajk', T2, Xoovo)
    Wovoo += temp_jk - temp_jk.transpose((0,1,3,2))
    T2A = None

    T2B = T2 + 2.0*einsum('bj,ak->bajk', T1, T1)
    Xvovv = I.vovv.copy()
    Xvovv -= einsum('al,licb->aicb', T1, I.oovv)
    Wovoo += 0.5*einsum('bcjk,aicb->iajk', T2B, Xvovv)
    return Wovoo

def eom_epccsd_1_s1_Wp(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp = G.copy()
    Wp += einsum('ai,xia->x', T1, g.ov)
    return Wp

def eom_epccsd_1_s1_Wp_p(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    nm = w.shape[0]
    Wp_p = numpy.zeros((nm,nm), dtype=U11.dtype)
    Wp_p += numpy.diag(w)
    Wp_p += einsum('xai,yia->xy', U11, g.ov)
    return Wp_p

def eom_epccsd_1_s1_Wpov(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wpov = g.ov.copy()
    return Wpov

def eom_epccsd_1_s1_Wpvv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wpvv = g.vv.copy()
    Wpvv -= einsum('ai,xib->xab', T1, g.ov)
    return Wpvv

def eom_epccsd_1_s1_Wpoo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wpoo = g.oo.copy()
    Wpoo += einsum('ai,xja->xji', T1, g.ov)
    return Wpoo

def eom_epccsd_1_s1_Wpvo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wpvo = g.vo.copy()
    Wpvo -= einsum('aj,xji->xai', T1, g.oo)
    Wpvo += einsum('bi,xab->xai', T1, g.vv)
    T2A = T2 + einsum('di,aj->adji', T1, T1)
    Wpvo -= einsum('abji,xjb->xai', T2A, g.ov)
    return Wpvo

def eom_epccsd_1_s1_Wp_ov(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp_ov = h.ov.copy()
    Wp_ov -= einsum('xbj,jiab->xia', U11, I.oovv)
    return Wp_ov

def eom_epccsd_1_s1_Wp_vv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp_vv = h.vv.copy()
    Wp_vv -= einsum('ai,xib->xab', T1, h.ov)
    Xov = F.ov.copy()
    Xov += einsum('y,yib->ib', S1, g.ov)
    Xov -= einsum('cj,jibc->ib', T1, I.oovv)
    Wp_vv -= einsum('xai,ib->xab', U11, Xov)
    Xvovv = I.vovv.copy()
    Xvovv -= einsum('aj,jibc->aibc', T1, I.oovv)
    Wp_vv += einsum('xci,aibc->xab', U11, Xvovv)
    return Wp_vv

def eom_epccsd_1_s1_Wp_oo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp_oo = h.oo.copy()
    Wp_oo += 1.0*einsum('ai,xja->xji', T1, h.ov)
    Xov = F.ov.copy()
    Xov += einsum('y,yia->ia', S1, g.ov)
    Xov -= einsum('ak,kjba->jb', T1, I.oovv)
    Wp_oo += einsum('xai,ja->xji', U11, Xov)
    Xooov = I.ooov.copy()
    Xooov += einsum('ai,kjab->kjib', T1, I.oovv)
    Wp_oo += -1.0*einsum('xbk,kjib->xji', U11, Xooov)
    return Wp_oo

def eom_epccsd_1_s1_Wp_poo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp_poo = einsum('xaj,yia->xyij', U11, g.ov)
    return Wp_poo

def eom_epccsd_1_s1_Wp_pvo(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Xmvv = g.vv.copy()
    Xmvv -= einsum('aj,yjb->yab', T1, g.ov)
    Wp_pvo = einsum('xbi,yab->xyai', U11, Xmvv)
    Xmoo = g.oo.copy()
    Xmoo += einsum('bi,yjb->yji', T1, g.ov)
    Wp_pvo -= einsum('xaj,yji->xyai', U11, Xmoo)
    return Wp_pvo

def eom_epccsd_1_s1_Wp_pvv(amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    Wp_pvv = -einsum('xai,yib->xyab', U11, g.ov)
    return Wp_pvv


class GRADImds(object):

    def __init__(self):

        self.foo = None
        self.fov = None
        self.fvo = None
        self.fvv = None
        
        self.Yoo = None
        self.Yov = None
        self.Yvv = None
        self.Yovov = None
        self.Yoooo = None
        self.Yvvvv = None

        self.Xoo = None
        self.Xov = None
        self.Xvv = None

        self.Wov = None
        self.Wvv = None
        
        self.Woo = None
        self.Woovv = None

        self.Iovvv = None
        self.Iooov = None

    def build(self, amps, f, I, L1, R1, L2, R2, Z1, Z2):
        """
        Ref[1] J. Chem. Phys. 122, 224106 (2005
        """
        T1, T2, S1, U11 = amps
        no, nv = T1.shape

        # F_{ia} = f_{ia} + (ij|ab) t^b_j
        self.Fov = f[:no,no:] 
        self.Fov += einsum('jb, ijab->ia', T1, I.oovv)

        # I6_{ijka} = (ij|ka) - t^c_k (ij|ac)
        self.Iooov = I.ooov
        self.Iooov -= einsum('ck,ijac->ijka', T1, I.oovv)

        # I7_{iabc} = (ia||bc) - t^a_j (ij|bc)
        self.Iovvv = Iovvv
        self.Iovvv -= einsum('aj,ijbc->iabc', T1, I.oovv)
        # T3_{ij} = 0.5 r^{cd}_{ik} (jk|cd)
        self.Too = 0.5 * einsum('cdik,jkcd->ij', T2, I.oovv)
        # T4_{ab} = 0.5 * r^{ac}_{kl} (kl|bc)
        self.Tvv = 0.5 * einsum('ackl, klbc->ab', T2, I.oovv)
        
        #X1_{ab} = \sum l^i r^b_i
        self.Xvv = einsum('li, bi-> ab', L1, R1)
        self.Xoo = einsum('ai, aj-> ij', L1, R1)

        # Y1-Y6
        self.Yov = einsum('abij,bj->ia', L2, R1)
        self.Yvv = einsum('acij,bcij->ab', L2, R2)
        self.Yoo = einsum('abik,abjk->ij', L2, R2)
        self.Yovov = einsum('acik,bcjk->iajb', L2, R2)
        self.Yovov = einsum('acik,bcjk->iajb', L2, R2)
        self.Yoooo = einsum('abij,abkl->ijkl', L2, R2)
        self.Yvvvv = einsum('abij,cdij->abcd', L2, R2)

        #
        omega = 1.0 # todo
        self.r0 = einsum('ai,ia->0', T1, Fov) + einsum('abij,ijab->0', R2, I.oovv) / 4.0
        self.r0 = self.r0/omega

        # r0 = - r^a_i z^a_i - 0.25 r^{ab}_{ij} * z^ab_ij
        self.r0 = - einsum('ai,ai->0', T1, Z1) - 0.25*einsum('abij,abij->0', T2, Z2)

        # rt^a_i = r^a_i + 0.5 * r0 t^a_i
        self.rt1 = R1 + 0.5 * self.r0 * T1

        # rtt^a_i = r^a_i + r0 * T1
        self.rtt1 = R1 + self.r0 * T1

        self.rt2 = R2 + 0.5 * self.r0 * T2
        self.rtt2 = R2 + self.r0 * T2
        
        #TODO
        return None

class EEImds(object):
    def __init__(self):
        self.Wov = None
        self.Wvv = None
        self.Woo = None
        self.Woovv = None
        self.Wvovv = None
        self.Wooov = None
        self.Wvvvv = None
        self.Woooo = None
        self.Wvoov = None
        self.Wvvvo = None
        self.Wovoo = None
        self.Wp = None
        self.Wp_p = None
        self.Wpov = None
        self.Wpvv = None
        self.Wpoo = None
        self.Wpvo = None
        self.Wp_ov = None
        self.Wp_vv = None
        self.Wp_oo = None
        self.Wp_pvo = None
        self.Wp_pvv = None
        self.Wp_poo = None

    def build(self, amps, F, I, w, g, h, G, H):
        self.Wov = eom_epccsd_1_s1_Wov(amps, F, I, w, g, h, G, H)
        self.Wvv = eom_epccsd_1_s1_Wvv(amps, F, I, w, g, h, G, H)
        self.Woo = eom_epccsd_1_s1_Woo(amps, F, I, w, g, h, G, H)
        self.Woovv = eom_epccsd_1_s1_Woovv(amps, F, I, w, g, h, G, H)
        self.Wvovv = eom_epccsd_1_s1_Wvovv(amps, F, I, w, g, h, G, H)
        self.Wooov = eom_epccsd_1_s1_Wooov(amps, F, I, w, g, h, G, H)
        self.Wvvvv = eom_epccsd_1_s1_Wvvvv(amps, F, I, w, g, h, G, H)
        self.Woooo = eom_epccsd_1_s1_Woooo(amps, F, I, w, g, h, G, H)
        self.Wvoov = eom_epccsd_1_s1_Wvoov(amps, F, I, w, g, h, G, H)
        self.Wvvvo = eom_epccsd_1_s1_Wvvvo(amps, F, I, w, g, h, G, H)
        self.Wovoo = eom_epccsd_1_s1_Wovoo(amps, F, I, w, g, h, G, H)
        self.Wp = eom_epccsd_1_s1_Wp(amps, F, I, w, g, h, G, H)
        self.Wp_p = eom_epccsd_1_s1_Wp_p(amps, F, I, w, g, h, G, H)
        self.Wpov = eom_epccsd_1_s1_Wpov(amps, F, I, w, g, h, G, H)
        self.Wpvv = eom_epccsd_1_s1_Wpvv(amps, F, I, w, g, h, G, H)
        self.Wpoo = eom_epccsd_1_s1_Wpoo(amps, F, I, w, g, h, G, H)
        self.Wpvo = eom_epccsd_1_s1_Wpvo(amps, F, I, w, g, h, G, H)
        self.Wp_ov = eom_epccsd_1_s1_Wp_ov(amps, F, I, w, g, h, G, H)
        self.Wp_vv = eom_epccsd_1_s1_Wp_vv(amps, F, I, w, g, h, G, H)
        self.Wp_oo = eom_epccsd_1_s1_Wp_oo(amps, F, I, w, g, h, G, H)
        self.Wp_pvo = eom_epccsd_1_s1_Wp_pvo(amps, F, I, w, g, h, G, H)
        self.Wp_pvv = eom_epccsd_1_s1_Wp_pvv(amps, F, I, w, g, h, G, H)
        self.Wp_poo = eom_epccsd_1_s1_Wp_poo(amps, F, I, w, g, h, G, H)

class IPImds(object):
    def __init__(self):
        self.Wov = None
        self.Wvv = None
        self.Woo = None
        self.Wooov = None
        self.Woooo = None
        self.Wvoov = None
        self.Wovoo = None
        self.Wp = None
        self.Wp_p = None
        self.Wpoo = None
        self.Wpvo = None
        self.Wp_ov = None
        self.Wp_oo = None
        self.Wp_poo = None

    def build(self, amps, F, I, w, g, h, G, H):
        self.Wov = eom_epccsd_1_s1_Wov(amps, F, I, w, g, h, G, H)
        self.Woo = eom_epccsd_1_s1_Woo(amps, F, I, w, g, h, G, H)
        self.Wvv = eom_epccsd_1_s1_Wvv(amps, F, I, w, g, h, G, H)
        self.Woooo = eom_epccsd_1_s1_Woooo(amps, F, I, w, g, h, G, H)
        self.Wooov = eom_epccsd_1_s1_Wooov(amps, F, I, w, g, h, G, H)
        self.Wvoov = eom_epccsd_1_s1_Wvoov(amps, F, I, w, g, h, G, H)
        self.Wovoo = eom_epccsd_1_s1_Wovoo(amps, F, I, w, g, h, G, H)
        self.Wp = eom_epccsd_1_s1_Wp(amps, F, I, w, g, h, G, H)
        self.Wp_p = eom_epccsd_1_s1_Wp_p(amps, F, I, w, g, h, G, H)
        self.Wpoo = eom_epccsd_1_s1_Wpoo(amps, F, I, w, g, h, G, H)
        self.Wpvo = eom_epccsd_1_s1_Wpvo(amps, F, I, w, g, h, G, H)
        self.Wp_oo = eom_epccsd_1_s1_Wp_oo(amps, F, I, w, g, h, G, H)
        self.Wp_ov = eom_epccsd_1_s1_Wp_ov(amps, F, I, w, g, h, G, H)
        self.Wp_poo = eom_epccsd_1_s1_Wp_poo(amps, F, I, w, g, h, G, H)

class EAImds(object):
    def __init__(self):
        self.Wov = None
        self.Woo = None
        self.Wvv = None
        self.Wvovv = None
        self.Wvvvo = None
        self.Wvoov = None
        self.Wvvvv = None
        self.Wp = None
        self.Wp_p = None
        self.Wpvv = None
        self.Wpvo = None
        self.Wp_vv = None
        self.Wp_ov = None

    def build(self, amps, F, I, w, g, h, G, H):
        self.Wov = eom_epccsd_1_s1_Wov(amps, F, I, w, g, h, G, H)
        self.Woo = eom_epccsd_1_s1_Woo(amps, F, I, w, g, h, G, H)
        self.Wvv = eom_epccsd_1_s1_Wvv(amps, F, I, w, g, h, G, H)
        self.Wvovv = eom_epccsd_1_s1_Wvovv(amps, F, I, w, g, h, G, H)
        self.Wvvvo = eom_epccsd_1_s1_Wvvvo(amps, F, I, w, g, h, G, H)
        self.Wvoov = eom_epccsd_1_s1_Wvoov(amps, F, I, w, g, h, G, H)
        self.Wvvvv = eom_epccsd_1_s1_Wvvvv(amps, F, I, w, g, h, G, H)
        self.Wp = eom_epccsd_1_s1_Wp(amps, F, I, w, g, h, G, H)
        self.Wp_p = eom_epccsd_1_s1_Wp_p(amps, F, I, w, g, h, G, H)
        self.Wpvv = eom_epccsd_1_s1_Wpvv(amps, F, I, w, g, h, G, H)
        self.Wpvo = eom_epccsd_1_s1_Wpvo(amps, F, I, w, g, h, G, H)
        self.Wp_vv = eom_epccsd_1_s1_Wp_vv(amps, F, I, w, g, h, G, H)
        self.Wp_ov = eom_epccsd_1_s1_Wp_ov(amps, F, I, w, g, h, G, H)
        self.Wp_pvv = eom_epccsd_1_s1_Wp_pvv(amps, F, I, w, g, h, G, H)

def eom_ee_epccsd_1_s1_sigma_slow(RS, RD, R1, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps

    sigS = numpy.zeros(T1.shape)
    sigD = numpy.zeros(T2.shape)
    sig1 = numpy.zeros(S1.shape)
    sigS1 = numpy.zeros(U11.shape)

    sigS += -1.0*einsum('ji,aj->ai', F.oo, RS)
    sigS += 1.0*einsum('ab,bi->ai', F.vv, RS)
    sigS += 1.0*einsum('I,Iai->ai', G, RS1)
    sigS += 1.0*einsum('Iai,I->ai', g.vo, R1)
    sigS += -1.0*einsum('jb,abji->ai', F.ov, RD)
    sigS += -1.0*einsum('ajbi,bj->ai', I.vovo, RS)
    sigS += -1.0*einsum('Iji,Iaj->ai', g.oo, RS1)
    sigS += 1.0*einsum('Iab,Ibi->ai', g.vv, RS1)
    sigS += 0.5*einsum('jkib,abkj->ai', I.ooov, RD)
    sigS += 0.5*einsum('ajbc,cbji->ai', I.vovv, RD)
    sigS += -1.0*einsum('jb,bi,aj->ai', F.ov, T1, RS)
    sigS += -1.0*einsum('jb,aj,bi->ai', F.ov, T1, RS)
    sigS += -1.0*einsum('Iji,aj,I->ai', g.oo, T1, R1)
    sigS += -1.0*einsum('Iji,I,aj->ai', g.oo, S1, RS)
    sigS += 1.0*einsum('Iab,bi,I->ai', g.vv, T1, R1)
    sigS += 1.0*einsum('Iab,I,bi->ai', g.vv, S1, RS)
    sigS += -1.0*einsum('jkib,aj,bk->ai', I.ooov, T1, RS)
    sigS += 1.0*einsum('jkib,bj,ak->ai', I.ooov, T1, RS)
    sigS += -1.0*einsum('ajbc,ci,bj->ai', I.vovv, T1, RS)
    sigS += 1.0*einsum('ajbc,cj,bi->ai', I.vovv, T1, RS)
    sigS += -1.0*einsum('Ijb,bi,Iaj->ai', g.ov, T1, RS1)
    sigS += -1.0*einsum('Ijb,aj,Ibi->ai', g.ov, T1, RS1)
    sigS += 1.0*einsum('Ijb,bj,Iai->ai', g.ov, T1, RS1)
    sigS += -1.0*einsum('Ijb,abji,I->ai', g.ov, T2, R1)
    sigS += -1.0*einsum('Ijb,I,abji->ai', g.ov, S1, RD)
    sigS += 1.0*einsum('Ijb,Iai,bj->ai', g.ov, U11, RS)
    sigS += -1.0*einsum('Ijb,Ibi,aj->ai', g.ov, U11, RS)
    sigS += -1.0*einsum('Ijb,Iaj,bi->ai', g.ov, U11, RS)
    sigS += -0.5*einsum('jkbc,ci,abkj->ai', I.oovv, T1, RD)
    sigS += -0.5*einsum('jkbc,aj,cbki->ai', I.oovv, T1, RD)
    sigS += 1.0*einsum('jkbc,cj,abki->ai', I.oovv, T1, RD)
    sigS += 1.0*einsum('jkbc,acji,bk->ai', I.oovv, T2, RS)
    sigS += 0.5*einsum('jkbc,cbji,ak->ai', I.oovv, T2, RS)
    sigS += 0.5*einsum('jkbc,ackj,bi->ai', I.oovv, T2, RS)
    sigS += -1.0*einsum('Ijb,bi,aj,I->ai', g.ov, T1, T1, R1)
    sigS += -1.0*einsum('Ijb,bi,I,aj->ai', g.ov, T1, S1, RS)
    sigS += -1.0*einsum('Ijb,aj,I,bi->ai', g.ov, T1, S1, RS)
    sigS += 1.0*einsum('jkbc,ci,aj,bk->ai', I.oovv, T1, T1, RS)
    sigS += -1.0*einsum('jkbc,ci,bj,ak->ai', I.oovv, T1, T1, RS)
    sigS += -1.0*einsum('jkbc,aj,ck,bi->ai', I.oovv, T1, T1, RS)

    sigD += 1.0*einsum('ki,bakj->abij', F.oo, RD)
    sigD += -1.0*einsum('kj,baki->abij', F.oo, RD)
    sigD += -1.0*einsum('bc,acji->abij', F.vv, RD)
    sigD += 1.0*einsum('ac,bcji->abij', F.vv, RD)
    sigD += -1.0*einsum('bkji,ak->abij', I.vooo, RS)
    sigD += 1.0*einsum('akji,bk->abij', I.vooo, RS)
    sigD += 1.0*einsum('baci,cj->abij', I.vvvo, RS)
    sigD += -1.0*einsum('bacj,ci->abij', I.vvvo, RS)
    sigD += -1.0*einsum('Ibi,Iaj->abij', g.vo, RS1)
    sigD += 1.0*einsum('Iai,Ibj->abij', g.vo, RS1)
    sigD += 1.0*einsum('Ibj,Iai->abij', g.vo, RS1)
    sigD += -1.0*einsum('Iaj,Ibi->abij', g.vo, RS1)
    sigD += -0.5*einsum('klji,balk->abij', I.oooo, RD)
    sigD += -1.0*einsum('bkci,ackj->abij', I.vovo, RD)
    sigD += 1.0*einsum('akci,bckj->abij', I.vovo, RD)
    sigD += 1.0*einsum('bkcj,acki->abij', I.vovo, RD)
    sigD += -1.0*einsum('akcj,bcki->abij', I.vovo, RD)
    sigD += -0.5*einsum('bacd,dcji->abij', I.vvvv, RD)
    sigD += 1.0*einsum('kc,ci,bakj->abij', F.ov, T1, RD)
    sigD += -1.0*einsum('kc,cj,baki->abij', F.ov, T1, RD)
    sigD += 1.0*einsum('kc,bk,acji->abij', F.ov, T1, RD)
    sigD += -1.0*einsum('kc,ak,bcji->abij', F.ov, T1, RD)
    sigD += -1.0*einsum('kc,bcji,ak->abij', F.ov, T2, RS)
    sigD += 1.0*einsum('kc,acji,bk->abij', F.ov, T2, RS)
    sigD += -1.0*einsum('kc,baki,cj->abij', F.ov, T2, RS)
    sigD += 1.0*einsum('kc,bakj,ci->abij', F.ov, T2, RS)
    sigD += 1.0*einsum('klji,bk,al->abij', I.oooo, T1, RS)
    sigD += -1.0*einsum('klji,ak,bl->abij', I.oooo, T1, RS)
    sigD += -1.0*einsum('bkci,cj,ak->abij', I.vovo, T1, RS)
    sigD += 1.0*einsum('akci,cj,bk->abij', I.vovo, T1, RS)
    sigD += -1.0*einsum('bkci,ak,cj->abij', I.vovo, T1, RS)
    sigD += 1.0*einsum('akci,bk,cj->abij', I.vovo, T1, RS)
    sigD += 1.0*einsum('bkcj,ci,ak->abij', I.vovo, T1, RS)
    sigD += -1.0*einsum('akcj,ci,bk->abij', I.vovo, T1, RS)
    sigD += 1.0*einsum('bkcj,ak,ci->abij', I.vovo, T1, RS)
    sigD += -1.0*einsum('akcj,bk,ci->abij', I.vovo, T1, RS)
    sigD += 1.0*einsum('bacd,di,cj->abij', I.vvvv, T1, RS)
    sigD += -1.0*einsum('bacd,dj,ci->abij', I.vvvv, T1, RS)
    sigD += 1.0*einsum('Iki,bk,Iaj->abij', g.oo, T1, RS1)
    sigD += -1.0*einsum('Iki,ak,Ibj->abij', g.oo, T1, RS1)
    sigD += -1.0*einsum('Ikj,bk,Iai->abij', g.oo, T1, RS1)
    sigD += 1.0*einsum('Ikj,ak,Ibi->abij', g.oo, T1, RS1)
    sigD += 1.0*einsum('Iki,bakj,I->abij', g.oo, T2, R1)
    sigD += -1.0*einsum('Ikj,baki,I->abij', g.oo, T2, R1)
    sigD += 1.0*einsum('Iki,I,bakj->abij', g.oo, S1, RD)
    sigD += -1.0*einsum('Ikj,I,baki->abij', g.oo, S1, RD)
    sigD += -1.0*einsum('Iki,Ibj,ak->abij', g.oo, U11, RS)
    sigD += 1.0*einsum('Iki,Iaj,bk->abij', g.oo, U11, RS)
    sigD += 1.0*einsum('Ikj,Ibi,ak->abij', g.oo, U11, RS)
    sigD += -1.0*einsum('Ikj,Iai,bk->abij', g.oo, U11, RS)
    sigD += -1.0*einsum('Ibc,ci,Iaj->abij', g.vv, T1, RS1)
    sigD += 1.0*einsum('Iac,ci,Ibj->abij', g.vv, T1, RS1)
    sigD += 1.0*einsum('Ibc,cj,Iai->abij', g.vv, T1, RS1)
    sigD += -1.0*einsum('Iac,cj,Ibi->abij', g.vv, T1, RS1)
    sigD += -1.0*einsum('Ibc,acji,I->abij', g.vv, T2, R1)
    sigD += 1.0*einsum('Iac,bcji,I->abij', g.vv, T2, R1)
    sigD += -1.0*einsum('Ibc,I,acji->abij', g.vv, S1, RD)
    sigD += 1.0*einsum('Iac,I,bcji->abij', g.vv, S1, RD)
    sigD += 1.0*einsum('Ibc,Iai,cj->abij', g.vv, U11, RS)
    sigD += -1.0*einsum('Iac,Ibi,cj->abij', g.vv, U11, RS)
    sigD += -1.0*einsum('Ibc,Iaj,ci->abij', g.vv, U11, RS)
    sigD += 1.0*einsum('Iac,Ibj,ci->abij', g.vv, U11, RS)
    sigD += 0.5*einsum('klic,cj,balk->abij', I.ooov, T1, RD)
    sigD += -1.0*einsum('klic,bk,aclj->abij', I.ooov, T1, RD)
    sigD += 1.0*einsum('klic,ak,bclj->abij', I.ooov, T1, RD)
    sigD += -1.0*einsum('klic,ck,balj->abij', I.ooov, T1, RD)
    sigD += -0.5*einsum('kljc,ci,balk->abij', I.ooov, T1, RD)
    sigD += 1.0*einsum('kljc,bk,acli->abij', I.ooov, T1, RD)
    sigD += -1.0*einsum('kljc,ak,bcli->abij', I.ooov, T1, RD)
    sigD += 1.0*einsum('kljc,ck,bali->abij', I.ooov, T1, RD)
    sigD += 1.0*einsum('klic,bakj,cl->abij', I.ooov, T2, RS)
    sigD += -1.0*einsum('klic,bckj,al->abij', I.ooov, T2, RS)
    sigD += 1.0*einsum('klic,ackj,bl->abij', I.ooov, T2, RS)
    sigD += 0.5*einsum('klic,balk,cj->abij', I.ooov, T2, RS)
    sigD += -1.0*einsum('kljc,baki,cl->abij', I.ooov, T2, RS)
    sigD += 1.0*einsum('kljc,bcki,al->abij', I.ooov, T2, RS)
    sigD += -1.0*einsum('kljc,acki,bl->abij', I.ooov, T2, RS)
    sigD += -0.5*einsum('kljc,balk,ci->abij', I.ooov, T2, RS)
    sigD += -1.0*einsum('bkcd,di,ackj->abij', I.vovv, T1, RD)
    sigD += 1.0*einsum('akcd,di,bckj->abij', I.vovv, T1, RD)
    sigD += 1.0*einsum('bkcd,dj,acki->abij', I.vovv, T1, RD)
    sigD += -1.0*einsum('akcd,dj,bcki->abij', I.vovv, T1, RD)
    sigD += 0.5*einsum('bkcd,ak,dcji->abij', I.vovv, T1, RD)
    sigD += -1.0*einsum('bkcd,dk,acji->abij', I.vovv, T1, RD)
    sigD += -0.5*einsum('akcd,bk,dcji->abij', I.vovv, T1, RD)
    sigD += 1.0*einsum('akcd,dk,bcji->abij', I.vovv, T1, RD)
    sigD += 1.0*einsum('bkcd,adji,ck->abij', I.vovv, T2, RS)
    sigD += 0.5*einsum('bkcd,dcji,ak->abij', I.vovv, T2, RS)
    sigD += -1.0*einsum('akcd,bdji,ck->abij', I.vovv, T2, RS)
    sigD += -0.5*einsum('akcd,dcji,bk->abij', I.vovv, T2, RS)
    sigD += -1.0*einsum('bkcd,adki,cj->abij', I.vovv, T2, RS)
    sigD += 1.0*einsum('akcd,bdki,cj->abij', I.vovv, T2, RS)
    sigD += 1.0*einsum('bkcd,adkj,ci->abij', I.vovv, T2, RS)
    sigD += -1.0*einsum('akcd,bdkj,ci->abij', I.vovv, T2, RS)
    sigD += -1.0*einsum('Ikc,bcji,Iak->abij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikc,acji,Ibk->abij', g.ov, T2, RS1)
    sigD += -1.0*einsum('Ikc,baki,Icj->abij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikc,bcki,Iaj->abij', g.ov, T2, RS1)
    sigD += -1.0*einsum('Ikc,acki,Ibj->abij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikc,bakj,Ici->abij', g.ov, T2, RS1)
    sigD += -1.0*einsum('Ikc,bckj,Iai->abij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikc,ackj,Ibi->abij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikc,Ibi,ackj->abij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikc,Iai,bckj->abij', g.ov, U11, RD)
    sigD += 1.0*einsum('Ikc,Ici,bakj->abij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikc,Ibj,acki->abij', g.ov, U11, RD)
    sigD += 1.0*einsum('Ikc,Iaj,bcki->abij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikc,Icj,baki->abij', g.ov, U11, RD)
    sigD += 1.0*einsum('Ikc,Ibk,acji->abij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikc,Iak,bcji->abij', g.ov, U11, RD)
    sigD += -0.5*einsum('klcd,bdji,aclk->abij', I.oovv, T2, RD)
    sigD += 0.5*einsum('klcd,adji,bclk->abij', I.oovv, T2, RD)
    sigD += 0.25*einsum('klcd,dcji,balk->abij', I.oovv, T2, RD)
    sigD += -0.5*einsum('klcd,baki,dclj->abij', I.oovv, T2, RD)
    sigD += 1.0*einsum('klcd,bdki,aclj->abij', I.oovv, T2, RD)
    sigD += -1.0*einsum('klcd,adki,bclj->abij', I.oovv, T2, RD)
    sigD += -0.5*einsum('klcd,dcki,balj->abij', I.oovv, T2, RD)
    sigD += 0.5*einsum('klcd,bakj,dcli->abij', I.oovv, T2, RD)
    sigD += -1.0*einsum('klcd,bdkj,acli->abij', I.oovv, T2, RD)
    sigD += 1.0*einsum('klcd,adkj,bcli->abij', I.oovv, T2, RD)
    sigD += 0.5*einsum('klcd,dckj,bali->abij', I.oovv, T2, RD)
    sigD += 0.25*einsum('klcd,balk,dcji->abij', I.oovv, T2, RD)
    sigD += -0.5*einsum('klcd,bdlk,acji->abij', I.oovv, T2, RD)
    sigD += 0.5*einsum('klcd,adlk,bcji->abij', I.oovv, T2, RD)
    sigD += -1.0*einsum('klic,cj,bk,al->abij', I.ooov, T1, T1, RS)
    sigD += 1.0*einsum('klic,cj,ak,bl->abij', I.ooov, T1, T1, RS)
    sigD += -1.0*einsum('klic,bk,al,cj->abij', I.ooov, T1, T1, RS)
    sigD += 1.0*einsum('kljc,ci,bk,al->abij', I.ooov, T1, T1, RS)
    sigD += -1.0*einsum('kljc,ci,ak,bl->abij', I.ooov, T1, T1, RS)
    sigD += 1.0*einsum('kljc,bk,al,ci->abij', I.ooov, T1, T1, RS)
    sigD += -1.0*einsum('bkcd,di,cj,ak->abij', I.vovv, T1, T1, RS)
    sigD += 1.0*einsum('akcd,di,cj,bk->abij', I.vovv, T1, T1, RS)
    sigD += -1.0*einsum('bkcd,di,ak,cj->abij', I.vovv, T1, T1, RS)
    sigD += 1.0*einsum('akcd,di,bk,cj->abij', I.vovv, T1, T1, RS)
    sigD += 1.0*einsum('bkcd,dj,ak,ci->abij', I.vovv, T1, T1, RS)
    sigD += -1.0*einsum('akcd,dj,bk,ci->abij', I.vovv, T1, T1, RS)
    sigD += 1.0*einsum('Ikc,ci,bk,Iaj->abij', g.ov, T1, T1, RS1)
    sigD += -1.0*einsum('Ikc,ci,ak,Ibj->abij', g.ov, T1, T1, RS1)
    sigD += -1.0*einsum('Ikc,cj,bk,Iai->abij', g.ov, T1, T1, RS1)
    sigD += 1.0*einsum('Ikc,cj,ak,Ibi->abij', g.ov, T1, T1, RS1)
    sigD += 1.0*einsum('Ikc,ci,bakj,I->abij', g.ov, T1, T2, R1)
    sigD += -1.0*einsum('Ikc,cj,baki,I->abij', g.ov, T1, T2, R1)
    sigD += 1.0*einsum('Ikc,bk,acji,I->abij', g.ov, T1, T2, R1)
    sigD += -1.0*einsum('Ikc,ak,bcji,I->abij', g.ov, T1, T2, R1)
    sigD += 1.0*einsum('Ikc,ci,I,bakj->abij', g.ov, T1, S1, RD)
    sigD += -1.0*einsum('Ikc,cj,I,baki->abij', g.ov, T1, S1, RD)
    sigD += 1.0*einsum('Ikc,bk,I,acji->abij', g.ov, T1, S1, RD)
    sigD += -1.0*einsum('Ikc,ak,I,bcji->abij', g.ov, T1, S1, RD)
    sigD += -1.0*einsum('Ikc,ci,Ibj,ak->abij', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ikc,ci,Iaj,bk->abij', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ikc,cj,Ibi,ak->abij', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ikc,cj,Iai,bk->abij', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ikc,bk,Iai,cj->abij', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ikc,ak,Ibi,cj->abij', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ikc,bk,Iaj,ci->abij', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ikc,ak,Ibj,ci->abij', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ikc,bcji,I,ak->abij', g.ov, T2, S1, RS)
    sigD += 1.0*einsum('Ikc,acji,I,bk->abij', g.ov, T2, S1, RS)
    sigD += -1.0*einsum('Ikc,baki,I,cj->abij', g.ov, T2, S1, RS)
    sigD += 1.0*einsum('Ikc,bakj,I,ci->abij', g.ov, T2, S1, RS)
    sigD += -0.5*einsum('klcd,di,cj,balk->abij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klcd,di,bk,aclj->abij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klcd,di,ak,bclj->abij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klcd,di,ck,balj->abij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klcd,dj,bk,acli->abij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klcd,dj,ak,bcli->abij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klcd,dj,ck,bali->abij', I.oovv, T1, T1, RD)
    sigD += -0.5*einsum('klcd,bk,al,dcji->abij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klcd,bk,dl,acji->abij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klcd,ak,dl,bcji->abij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klcd,di,bakj,cl->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,di,bckj,al->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,di,ackj,bl->abij', I.oovv, T1, T2, RS)
    sigD += -0.5*einsum('klcd,di,balk,cj->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,dj,baki,cl->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,dj,bcki,al->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,dj,acki,bl->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,bk,adji,cl->abij', I.oovv, T1, T2, RS)
    sigD += -0.5*einsum('klcd,bk,dcji,al->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,ak,bdji,cl->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,dk,bcji,al->abij', I.oovv, T1, T2, RS)
    sigD += 0.5*einsum('klcd,ak,dcji,bl->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,dk,acji,bl->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,bk,adli,cj->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,ak,bdli,cj->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,dk,bali,cj->abij', I.oovv, T1, T2, RS)
    sigD += 0.5*einsum('klcd,dj,balk,ci->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,bk,adlj,ci->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,ak,bdlj,ci->abij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klcd,dk,balj,ci->abij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klcd,di,cj,bk,al->abij', I.oovv, T1, T1, T1, RS)
    sigD += -1.0*einsum('klcd,di,cj,ak,bl->abij', I.oovv, T1, T1, T1, RS)
    sigD += 1.0*einsum('klcd,di,bk,al,cj->abij', I.oovv, T1, T1, T1, RS)
    sigD += -1.0*einsum('klcd,dj,bk,al,ci->abij', I.oovv, T1, T1, T1, RS)

    sig1 += 1.0*einsum('I,I->I', w, R1)
    sig1 += 1.0*einsum('ia,Iai->I', F.ov, RS1)
    sig1 += 1.0*einsum('Iia,ai->I', h.ov, RS)
    sig1 += 1.0*einsum('Jia,J,Iai->I', g.ov, S1, RS1)
    sig1 += 1.0*einsum('Jia,Iai,J->I', g.ov, U11, R1)
    sig1 += -1.0*einsum('ijab,bi,Iaj->I', I.oovv, T1, RS1)
    sig1 += -1.0*einsum('ijab,Ibi,aj->I', I.oovv, U11, RS)

    sigS1 += -1.0*einsum('ji,Iaj->Iai', F.oo, RS1)
    sigS1 += 1.0*einsum('ab,Ibi->Iai', F.vv, RS1)
    sigS1 += 1.0*einsum('I,Iai->Iai', w, RS1)
    #sigS1 += 1.0*einsum('I,I,ai->Iai', w, S1, RS)  # added by YZ
    sigS1 += -1.0*einsum('Iji,aj->Iai', h.oo, RS)
    sigS1 += 1.0*einsum('Iab,bi->Iai', h.vv, RS)
    sigS1 += -1.0*einsum('ajbi,Ibj->Iai', I.vovo, RS1)
    sigS1 += -1.0*einsum('Ijb,abji->Iai', h.ov, RD)
    sigS1 += -1.0*einsum('jb,bi,Iaj->Iai', F.ov, T1, RS1)
    sigS1 += -1.0*einsum('jb,aj,Ibi->Iai', F.ov, T1, RS1)
    sigS1 += -1.0*einsum('jb,Ibi,aj->Iai', F.ov, U11, RS)
    sigS1 += -1.0*einsum('jb,Iaj,bi->Iai', F.ov, U11, RS)
    sigS1 += -1.0*einsum('Jji,J,Iaj->Iai', g.oo, S1, RS1)
    sigS1 += -1.0*einsum('Jji,Iaj,J->Iai', g.oo, U11, R1)
    sigS1 += -1.0*einsum('Ijb,bi,aj->Iai', h.ov, T1, RS)
    sigS1 += -1.0*einsum('Ijb,aj,bi->Iai', h.ov, T1, RS)
    sigS1 += 1.0*einsum('Jab,J,Ibi->Iai', g.vv, S1, RS1)
    sigS1 += 1.0*einsum('Jab,Ibi,J->Iai', g.vv, U11, R1)
    sigS1 += -1.0*einsum('jkib,aj,Ibk->Iai', I.ooov, T1, RS1)
    sigS1 += 1.0*einsum('jkib,bj,Iak->Iai', I.ooov, T1, RS1)
    sigS1 += -1.0*einsum('jkib,Iaj,bk->Iai', I.ooov, U11, RS)
    sigS1 += 1.0*einsum('jkib,Ibj,ak->Iai', I.ooov, U11, RS)
    sigS1 += -1.0*einsum('ajbc,ci,Ibj->Iai', I.vovv, T1, RS1)
    sigS1 += 1.0*einsum('ajbc,cj,Ibi->Iai', I.vovv, T1, RS1)
    sigS1 += -1.0*einsum('ajbc,Ici,bj->Iai', I.vovv, U11, RS)
    sigS1 += 1.0*einsum('ajbc,Icj,bi->Iai', I.vovv, U11, RS)
    sigS1 += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g.ov, U11, RS1)
    sigS1 += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g.ov, U11, RS1)
    sigS1 += 1.0*einsum('Jjb,Ibj,Jai->Iai', g.ov, U11, RS1)
    sigS1 += 1.0*einsum('Jjb,Jai,Ibj->Iai', g.ov, U11, RS1)
    sigS1 += -1.0*einsum('Jjb,Jbi,Iaj->Iai', g.ov, U11, RS1)
    sigS1 += -1.0*einsum('Jjb,Jaj,Ibi->Iai', g.ov, U11, RS1)
    sigS1 += 1.0*einsum('jkbc,acji,Ibk->Iai', I.oovv, T2, RS1)
    sigS1 += 0.5*einsum('jkbc,cbji,Iak->Iai', I.oovv, T2, RS1)
    sigS1 += 0.5*einsum('jkbc,ackj,Ibi->Iai', I.oovv, T2, RS1)
    sigS1 += -0.5*einsum('jkbc,Ici,abkj->Iai', I.oovv, U11, RD)
    sigS1 += -0.5*einsum('jkbc,Iaj,cbki->Iai', I.oovv, U11, RD)
    sigS1 += 1.0*einsum('jkbc,Icj,abki->Iai', I.oovv, U11, RD)
    sigS1 += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g.ov, T1, S1, RS1)
    sigS1 += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g.ov, T1, S1, RS1)
    sigS1 += -1.0*einsum('Jjb,bi,Iaj,J->Iai', g.ov, T1, U11, R1)
    sigS1 += -1.0*einsum('Jjb,aj,Ibi,J->Iai', g.ov, T1, U11, R1)
    sigS1 += -1.0*einsum('Jjb,J,Ibi,aj->Iai', g.ov, S1, U11, RS)
    sigS1 += -1.0*einsum('Jjb,J,Iaj,bi->Iai', g.ov, S1, U11, RS)
    sigS1 += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I.oovv, T1, T1, RS1)
    sigS1 += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I.oovv, T1, T1, RS1)
    sigS1 += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I.oovv, T1, T1, RS1)
    sigS1 += 1.0*einsum('jkbc,ci,Iaj,bk->Iai', I.oovv, T1, U11, RS)
    sigS1 += -1.0*einsum('jkbc,ci,Ibj,ak->Iai', I.oovv, T1, U11, RS)
    sigS1 += 1.0*einsum('jkbc,aj,Ici,bk->Iai', I.oovv, T1, U11, RS)
    sigS1 += 1.0*einsum('jkbc,cj,Ibi,ak->Iai', I.oovv, T1, U11, RS)
    sigS1 += -1.0*einsum('jkbc,aj,Ick,bi->Iai', I.oovv, T1, U11, RS)
    sigS1 += 1.0*einsum('jkbc,cj,Iak,bi->Iai', I.oovv, T1, U11, RS)

    return sigS, sigD, sig1, sigS1

def eom_ee_epccsd_1_s1_sigma_int(RS, RD, R1, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps

    nm,nv,no = U11.shape
    Wov = numpy.zeros((no,nv))
    Wov += 1.0*einsum('ia->ia', F.ov)
    Wov += 1.0*einsum('I,Iia->ia', S1, g.ov)
    Wov += -1.0*einsum('bj,jiab->ia', T1, I.oovv)

    Wvv = numpy.zeros((nv,nv))
    Wvv += 1.0*einsum('ab->ab', F.vv)
    Wvv += -1.0*einsum('ai,ib->ab', T1, F.ov)
    Wvv += 1.0*einsum('I,Iab->ab', S1, g.vv)
    Wvv += 1.0*einsum('ci,aibc->ab', T1, I.vovv)
    Wvv += -1.0*einsum('Iai,Iib->ab', U11, g.ov)
    Wvv += 0.5*einsum('acij,jibc->ab', T2, I.oovv)
    Wvv += -1.0*einsum('ai,I,Iib->ab', T1, S1, g.ov)
    Wvv += -1.0*einsum('ai,cj,ijbc->ab', T1, T1, I.oovv)

    Woo = numpy.zeros((no,no))
    Woo += 1.0*einsum('ij->ij', F.oo)
    Woo += 1.0*einsum('aj,ia->ij', T1, F.ov)
    Woo += 1.0*einsum('I,Iij->ij', S1, g.oo)
    Woo += -1.0*einsum('ak,kija->ij', T1, I.ooov)
    Woo += 1.0*einsum('Iaj,Iia->ij', U11, g.ov)
    Woo += -0.5*einsum('abkj,kiba->ij', T2, I.oovv)
    Woo += 1.0*einsum('aj,I,Iia->ij', T1, S1, g.ov)
    Woo += 1.0*einsum('aj,bk,kiba->ij', T1, T1, I.oovv)

    Woovv = numpy.zeros((no,no,nv,nv))
    Woovv += 1.0*einsum('jiba->ijab', I.oovv)

    Wvovv = numpy.zeros((nv,no,nv,nv))
    Wvovv += -1.0*einsum('aicb->aibc', I.vovv)
    Wvovv += 1.0*einsum('aj,jicb->aibc', T1, I.oovv)

    Wooov = numpy.zeros((no,no,no,nv))
    Wooov += -1.0*einsum('jika->ijka', I.ooov)
    Wooov += 1.0*einsum('bk,jiab->ijka', T1, I.oovv)

    Wvvvv = numpy.zeros((nv,nv,nv,nv))
    Wvvvv += 1.0*einsum('badc->abcd', I.vvvv)
    Wvvvv += -1.0*einsum('ai,bidc->abcd', T1, I.vovv)
    Wvvvv += 1.0*einsum('bi,aidc->abcd', T1, I.vovv)
    Wvvvv += -0.5*einsum('baij,jidc->abcd', T2, I.oovv)
    Wvvvv += 1.0*einsum('bi,aj,ijdc->abcd', T1, T1, I.oovv)

    Woooo = numpy.zeros((no,no,no,no))
    Woooo += 1.0*einsum('jilk->ijkl', I.oooo)
    Woooo += -1.0*einsum('al,jika->ijkl', T1, I.ooov)
    Woooo += 1.0*einsum('ak,jila->ijkl', T1, I.ooov)
    Woooo += -0.5*einsum('ablk,jiba->ijkl', T2, I.oovv)
    Woooo += 1.0*einsum('ak,bl,jiba->ijkl', T1, T1, I.oovv)

    Wvoov = numpy.zeros((nv,no,no,nv))
    Wvoov += -1.0*einsum('aibj->aijb', I.vovo)
    Wvoov += -1.0*einsum('ak,kijb->aijb', T1, I.ooov)
    Wvoov += -1.0*einsum('cj,aibc->aijb', T1, I.vovv)
    Wvoov += 1.0*einsum('Iaj,Iib->aijb', U11, g.ov)
    Wvoov += 1.0*einsum('ackj,kibc->aijb', T2, I.oovv)
    Wvoov += 1.0*einsum('cj,ak,kibc->aijb', T1, T1, I.oovv)

    Wvvvo = numpy.zeros((nv,nv,nv,no))
    Wvvvo += -1.0*einsum('baci->abci', I.vvvo)
    Wvvvo += 1.0*einsum('baji,jc->abci', T2, F.ov)
    Wvvvo += 1.0*einsum('aj,bjci->abci', T1, I.vovo)
    Wvvvo += -1.0*einsum('bj,ajci->abci', T1, I.vovo)
    Wvvvo += -1.0*einsum('di,bacd->abci', T1, I.vvvv)
    Wvvvo += -1.0*einsum('Iai,Ibc->abci', U11, g.vv)
    Wvvvo += 1.0*einsum('Ibi,Iac->abci', U11, g.vv)
    Wvvvo += -0.5*einsum('bajk,kjic->abci', T2, I.ooov)
    Wvvvo += 1.0*einsum('adji,bjcd->abci', T2, I.vovv)
    Wvvvo += -1.0*einsum('bdji,ajcd->abci', T2, I.vovv)
    Wvvvo += 1.0*einsum('bj,ak,jkic->abci', T1, T1, I.ooov)
    Wvvvo += 1.0*einsum('di,aj,bjcd->abci', T1, T1, I.vovv)
    Wvvvo += -1.0*einsum('di,bj,ajcd->abci', T1, T1, I.vovv)
    Wvvvo += 1.0*einsum('bj,Iai,Ijc->abci', T1, U11, g.ov)
    Wvvvo += -1.0*einsum('aj,Ibi,Ijc->abci', T1, U11, g.ov)
    Wvvvo += 1.0*einsum('baji,I,Ijc->abci', T2, S1, g.ov)
    Wvvvo += 0.5*einsum('di,bajk,kjcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('bj,adki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += 1.0*einsum('aj,bdki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('dj,baki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('di,bj,ak,jkcd->abci', T1, T1, T1, I.oovv)

    Wovoo = numpy.zeros((no,nv,no,no))
    Wovoo += 1.0*einsum('aikj->iajk', I.vooo)
    Wovoo += 1.0*einsum('abkj,ib->iajk', T2, F.ov)
    Wovoo += -1.0*einsum('al,likj->iajk', T1, I.oooo)
    Wovoo += 1.0*einsum('bk,aibj->iajk', T1, I.vovo)
    Wovoo += -1.0*einsum('bj,aibk->iajk', T1, I.vovo)
    Wovoo += 1.0*einsum('Iak,Iij->iajk', U11, g.oo)
    Wovoo += -1.0*einsum('Iaj,Iik->iajk', U11, g.oo)
    Wovoo += 1.0*einsum('ablk,lijb->iajk', T2, I.ooov)
    Wovoo += -1.0*einsum('ablj,likb->iajk', T2, I.ooov)
    Wovoo += -0.5*einsum('bckj,aicb->iajk', T2, I.vovv)
    Wovoo += 1.0*einsum('bk,al,lijb->iajk', T1, T1, I.ooov)
    Wovoo += -1.0*einsum('bj,al,likb->iajk', T1, T1, I.ooov)
    Wovoo += 1.0*einsum('bj,ck,aicb->iajk', T1, T1, I.vovv)
    Wovoo += 1.0*einsum('bj,Iak,Iib->iajk', T1, U11, g.ov)
    Wovoo += -1.0*einsum('bk,Iaj,Iib->iajk', T1, U11, g.ov)
    Wovoo += 1.0*einsum('abkj,I,Iib->iajk', T2, S1, g.ov)
    Wovoo += -1.0*einsum('bj,aclk,licb->iajk', T1, T2, I.oovv)
    Wovoo += 1.0*einsum('bk,aclj,licb->iajk', T1, T2, I.oovv)
    Wovoo += 0.5*einsum('al,bckj,licb->iajk', T1, T2, I.oovv)
    Wovoo += -1.0*einsum('bl,ackj,licb->iajk', T1, T2, I.oovv)
    Wovoo += -1.0*einsum('bj,ck,al,licb->iajk', T1, T1, T1, I.oovv)

    Wvovvvo = numpy.zeros((nv,no,nv,nv,nv,no))
    Wvovvvo += -1.0*einsum('bakj,kidc->aibcdj', T2, I.oovv)

    Woovovo = numpy.zeros((no,no,nv,no,nv,no))
    Woovovo += 1.0*einsum('aclk,jibc->ijakbl', T2, I.oovv)

    Wovvvoo = numpy.zeros((no,nv,nv,nv,no,no))
    Wovvvoo += 1.0*einsum('balk,lijc->iabcjk', T2, I.ooov)
    Wovvvoo += -1.0*einsum('balj,likc->iabcjk', T2, I.ooov)
    Wovvvoo += 1.0*einsum('adkj,bicd->iabcjk', T2, I.vovv)
    Wovvvoo += -1.0*einsum('bdkj,aicd->iabcjk', T2, I.vovv)
    Wovvvoo += -1.0*einsum('dj,balk,licd->iabcjk', T1, T2, I.oovv)
    Wovvvoo += 1.0*einsum('dk,balj,licd->iabcjk', T1, T2, I.oovv)
    Wovvvoo += -1.0*einsum('bl,adkj,licd->iabcjk', T1, T2, I.oovv)
    Wovvvoo += 1.0*einsum('al,bdkj,licd->iabcjk', T1, T2, I.oovv)

    Wp = numpy.zeros((nm))
    Wp += 1.0*einsum('I->I', G)
    Wp += 1.0*einsum('ai,Iia->I', T1, g.ov)

    Wp_p = numpy.zeros((nm,nm))
    Wp_p += numpy.diag(w)
    Wp_p += 1.0*einsum('Iai,Jia->IJ', U11, g.ov)

    Wpov = numpy.zeros((nm,no,nv))
    Wpov += 1.0*einsum('Iia->Iia', g.ov)

    Wpvv = numpy.zeros((nm,nv,nv))
    Wpvv += 1.0*einsum('Iab->Iab', g.vv)
    Wpvv += -1.0*einsum('ai,Iib->Iab', T1, g.ov)

    Wpoo = numpy.zeros((nm,no,no))
    Wpoo += 1.0*einsum('Iji->Iji', g.oo)
    Wpoo += 1.0*einsum('ai,Ija->Iji', T1, g.ov)

    Wpvo = numpy.zeros((nm,nv,no))
    Wpvo += 1.0*einsum('Iai->Iai', g.vo)
    Wpvo += -1.0*einsum('aj,Iji->Iai', T1, g.oo)
    Wpvo += 1.0*einsum('bi,Iab->Iai', T1, g.vv)
    Wpvo += -1.0*einsum('abji,Ijb->Iai', T2, g.ov)
    Wpvo += -1.0*einsum('bi,aj,Ijb->Iai', T1, T1, g.ov)

    Wp_ov = numpy.zeros((nm,no,nv))
    Wp_ov += 1.0*einsum('Iia->Iia', h.ov)
    Wp_ov += -1.0*einsum('Ibj,jiab->Iia', U11, I.oovv)

    Wp_vv = numpy.zeros((nm,nv,nv))
    Wp_vv += 1.0*einsum('Iab->Iab', h.vv)
    Wp_vv += -1.0*einsum('Iai,ib->Iab', U11, F.ov)
    Wp_vv += -1.0*einsum('ai,Iib->Iab', T1, h.ov)
    Wp_vv += 1.0*einsum('Ici,aibc->Iab', U11, I.vovv)
    Wp_vv += -1.0*einsum('J,Iai,Jib->Iab', S1, U11, g.ov)
    Wp_vv += -1.0*einsum('ai,Icj,ijbc->Iab', T1, U11, I.oovv)
    Wp_vv += 1.0*einsum('ci,Iaj,ijbc->Iab', T1, U11, I.oovv)

    Wp_oo = numpy.zeros((nm,no,no))
    Wp_oo += 1.0*einsum('Iji->Iji', h.oo)
    Wp_oo += 1.0*einsum('Iai,ja->Iji', U11, F.ov)
    Wp_oo += 1.0*einsum('ai,Ija->Iji', T1, h.ov)
    Wp_oo += -1.0*einsum('Iak,kjia->Iji', U11, I.ooov)
    Wp_oo += 1.0*einsum('J,Iai,Jja->Iji', S1, U11, g.ov)
    Wp_oo += 1.0*einsum('ai,Ibk,kjba->Iji', T1, U11, I.oovv)
    Wp_oo += -1.0*einsum('ak,Ibi,kjba->Iji', T1, U11, I.oovv)

    Wpvvoo = numpy.zeros((nm,nv,nv,no,no))
    Wpvvoo += 1.0*einsum('bakj,Iki->Iabij', T2, g.oo)
    Wpvvoo += -1.0*einsum('baki,Ikj->Iabij', T2, g.oo)
    Wpvvoo += -1.0*einsum('acji,Ibc->Iabij', T2, g.vv)
    Wpvvoo += 1.0*einsum('bcji,Iac->Iabij', T2, g.vv)
    Wpvvoo += 1.0*einsum('ci,bakj,Ikc->Iabij', T1, T2, g.ov)
    Wpvvoo += -1.0*einsum('cj,baki,Ikc->Iabij', T1, T2, g.ov)
    Wpvvoo += 1.0*einsum('bk,acji,Ikc->Iabij', T1, T2, g.ov)
    Wpvvoo += -1.0*einsum('ak,bcji,Ikc->Iabij', T1, T2, g.ov)

    Wpvvov = numpy.zeros((nm,nv,nv,no,nv))
    Wpvvov += -1.0*einsum('baji,Ijc->Iabic', T2, g.ov)

    Wpvooo = numpy.zeros((nm,nv,no,no,no))
    Wpvooo += -1.0*einsum('abkj,Iib->Iaijk', T2, g.ov)

    Wp_vovo = numpy.zeros((nm,nv,no,nv,no))
    Wp_vovo += 1.0*einsum('Iak,kijb->Iaibj', U11, I.ooov)
    Wp_vovo += 1.0*einsum('Icj,aibc->Iaibj', U11, I.vovv)
    Wp_vovo += -1.0*einsum('cj,Iak,kibc->Iaibj', T1, U11, I.oovv)
    Wp_vovo += -1.0*einsum('ak,Icj,kibc->Iaibj', T1, U11, I.oovv)

    Wp_ovvv = numpy.zeros((nm,no,nv,nv,nv))
    Wp_ovvv += -1.0*einsum('Iaj,jicb->Iiabc', U11, I.oovv)

    Wp_oovo = numpy.zeros((nm,no,no,nv,no))
    Wp_oovo += -1.0*einsum('Ibk,jiab->Iijak', U11, I.oovv)

    Wp_pvo = numpy.zeros((nm,nm,nv,no))
    Wp_pvo += -1.0*einsum('Iaj,Jji->IJai', U11, g.oo)
    Wp_pvo += 1.0*einsum('Ibi,Jab->IJai', U11, g.vv)
    Wp_pvo += -1.0*einsum('bi,Iaj,Jjb->IJai', T1, U11, g.ov)
    Wp_pvo += -1.0*einsum('aj,Ibi,Jjb->IJai', T1, U11, g.ov)

    Wp_pvv = numpy.zeros((nm,nm,nv,nv))
    Wp_pvv += -1.0*einsum('Iai,Jib->IJab', U11, g.ov)

    Wp_poo = numpy.zeros((nm,nm,no,no))
    Wp_poo += 1.0*einsum('Iaj,Jia->IJij', U11, g.ov)

    sigS = numpy.zeros(T1.shape)
    sigD = numpy.zeros(T2.shape)
    sig1 = numpy.zeros(S1.shape)
    sigS1 = numpy.zeros(U11.shape)

    sigS += -1.0*einsum('ji,aj->ai', Woo, RS)
    sigS += 1.0*einsum('ab,bi->ai', Wvv, RS)
    sigS += 1.0*einsum('I,Iai->ai', Wp, RS1)
    sigS += 1.0*einsum('Iai,I->ai', Wpvo, R1)
    sigS += -1.0*einsum('jb,abji->ai', Wov, RD)
    sigS += 1.0*einsum('ajib,bj->ai', Wvoov, RS)
    sigS += -1.0*einsum('Iji,Iaj->ai', Wpoo, RS1)
    sigS += 1.0*einsum('Iab,Ibi->ai', Wpvv, RS1)
    sigS += 0.5*einsum('jkib,abkj->ai', Wooov, RD)
    sigS += 0.5*einsum('ajbc,cbji->ai', Wvovv, RD)

    sigD += 1.0*einsum('ki,bakj->abij', Woo, RD)
    sigD += -1.0*einsum('kj,baki->abij', Woo, RD)
    sigD += -1.0*einsum('bc,acji->abij', Wvv, RD)
    sigD += 1.0*einsum('ac,bcji->abij', Wvv, RD)
    sigD += 1.0*einsum('kbji,ak->abij', Wovoo, RS)
    sigD += -1.0*einsum('kaji,bk->abij', Wovoo, RS)
    sigD += 1.0*einsum('baci,cj->abij', Wvvvo, RS)
    sigD += -1.0*einsum('bacj,ci->abij', Wvvvo, RS)
    sigD += -1.0*einsum('Ibi,Iaj->abij', Wpvo, RS1)
    sigD += 1.0*einsum('Iai,Ibj->abij', Wpvo, RS1)
    sigD += 1.0*einsum('Ibj,Iai->abij', Wpvo, RS1)
    sigD += -1.0*einsum('Iaj,Ibi->abij', Wpvo, RS1)
    sigD += -0.5*einsum('klji,balk->abij', Woooo, RD)
    sigD += 1.0*einsum('bkic,ackj->abij', Wvoov, RD)
    sigD += -1.0*einsum('akic,bckj->abij', Wvoov, RD)
    sigD += -1.0*einsum('bkjc,acki->abij', Wvoov, RD)
    sigD += 1.0*einsum('akjc,bcki->abij', Wvoov, RD)
    sigD += -0.5*einsum('bacd,dcji->abij', Wvvvv, RD)

    sigD += 1.0*einsum('Iabij,I->abij', Wpvvoo, R1)
    sigD += 1.0*einsum('labdij,dl->abij', Wovvvoo, RS)
    sigD += 1.0*einsum('Iabic,Icj->abij', Wpvvov, RS1)
    sigD += -1.0*einsum('Iabjc,Ici->abij', Wpvvov, RS1)
    sigD += -1.0*einsum('Iakij,Ibk->abij', Wpvooo, RS1)
    sigD += +1.0*einsum('Ibkij,Iak->abij', Wpvooo, RS1)
    sigD += 0.5*einsum('albcdj,cdil->abij', Wvovvvo, RD)
    sigD += -0.5*einsum('albcdi,cdjl->abij', Wvovvvo, RD)
    sigD += -0.5*einsum('klbidj,adkl->abij', Woovovo, RD)
    sigD += 0.5*einsum('klaidj,bdkl->abij', Woovovo, RD)

    sig1 += 1.0*einsum('IJ,J->I', Wp_p, R1)
    sig1 += 1.0*einsum('ia,Iai->I', Wov, RS1)
    sig1 += 1.0*einsum('Iia,ai->I', Wp_ov, RS)

    sigS1 += -1.0*einsum('ji,Iaj->Iai', Woo, RS1)
    sigS1 += 1.0*einsum('ab,Ibi->Iai', Wvv, RS1)
    sigS1 += 1.0*einsum('IJ,Jai->Iai', Wp_p, RS1)
    sigS1 += -1.0*einsum('Iji,aj->Iai', Wp_oo, RS)
    sigS1 += 1.0*einsum('Iab,bi->Iai', Wp_vv, RS)
    sigS1 += 1.0*einsum('ajib,Ibj->Iai', Wvoov, RS1)
    sigS1 += -1.0*einsum('Ijb,abji->Iai', Wp_ov, RD)

    sigS1 += einsum('IJab,Jbi->Iai', Wp_pvv, RS1)
    sigS1 -= einsum('IJji,Jaj->Iai', Wp_poo, RS1)
    sigS1 -= einsum('Iajbi,bj->Iai',Wp_vovo, RS)
    sigS1 += 0.5*einsum('Ijabc,bcji->Iai',Wp_ovvv, RD)
    sigS1 -= 0.5*einsum('Ijkbi,bajk->Iai',Wp_oovo, RD)
    sigS1 += einsum('IJai,J->Iai',Wp_pvo, R1)

    return sigS, sigD, sig1, sigS1

def eom_ee_epccsd_1_s1_sigma_opt(RS, RD, R1, RS1, amps, F, I, w, g, h, G, H, imds=None):
    T1,T2,S1,U11 = amps
    nm,nv,no = U11.shape

    if imds is None:
        imds = EEImds()
        imds.build(amps, F, I, w, g, h, G, H)

    sigS = numpy.zeros(T1.shape)
    sigD = numpy.zeros(T2.shape)
    sig1 = numpy.zeros(S1.shape)
    sigS1 = numpy.zeros(U11.shape)

    sigS -= einsum('ji,aj->ai', imds.Woo, RS)
    sigS += einsum('ab,bi->ai', imds.Wvv, RS)
    sigS += einsum('x,xai->ai', imds.Wp, RS1)
    sigS += einsum('xai,x->ai', imds.Wpvo, R1)
    sigS -= einsum('jb,abji->ai', imds.Wov, RD)
    sigS += einsum('ajib,bj->ai', imds.Wvoov, RS)
    sigS -= einsum('xji,xaj->ai', imds.Wpoo, RS1)
    sigS += einsum('xab,xbi->ai', imds.Wpvv, RS1)
    sigS += 0.5*einsum('jkib,abkj->ai', imds.Wooov, RD)
    sigS += 0.5*einsum('ajbc,cbji->ai', imds.Wvovv, RD)

    temp_ij = einsum('ki,bakj->abij', imds.Woo, RD)
    temp_ab = einsum('ac,bcji->abij', imds.Wvv, RD)
    temp_ab += einsum('kbji,ak->abij', imds.Wovoo, RS)
    temp_ij += einsum('baci,cj->abij', imds.Wvvvo, RS)
    temp_ijab = einsum('xai,xbj->abij', imds.Wpvo, RS1)
    temp_ijab += einsum('bkic,ackj->abij', imds.Wvoov, RD)
    sigD += -0.5*einsum('klji,balk->abij', imds.Woooo, RD)
    sigD += -0.5*einsum('bacd,dcji->abij', imds.Wvvvv, RD)
    sigD += temp_ijab
    sigD -= temp_ijab.transpose((0,1,3,2))
    sigD -= temp_ijab.transpose((1,0,2,3))
    sigD += temp_ijab.transpose((1,0,3,2))

    sig1 += 1.0*einsum('xy,y->x', imds.Wp_p, R1)
    sig1 += 1.0*einsum('ia,xai->x', imds.Wov, RS1)
    sig1 += 1.0*einsum('xia,ai->x', imds.Wp_ov, RS)

    sigS1 -= einsum('ji,xaj->xai', imds.Woo, RS1)
    sigS1 += einsum('ab,xbi->xai', imds.Wvv, RS1)
    sigS1 += einsum('xy,yai->xai', imds.Wp_p, RS1)
    sigS1 -= einsum('xji,aj->xai', imds.Wp_oo, RS)
    sigS1 += einsum('xab,bi->xai', imds.Wp_vv, RS)
    sigS1 += einsum('ajib,xbj->xai', imds.Wvoov, RS1)
    sigS1 -= einsum('xjb,abji->xai', imds.Wp_ov, RD)
    sigS1 += einsum('xyab,ybi->xai', imds.Wp_pvv, RS1)
    sigS1 -= einsum('xyji,yaj->xai', imds.Wp_poo, RS1)
    sigS1 += einsum('xyai,y->xai', imds.Wp_pvo, R1)


    # Wvovvvo
    Xoo = einsum('kldc,cdil->ki', I.oovv, RD)
    temp_ij += 0.5*einsum('abkj,ki->abij', T2, Xoo)
    Xoo = None

    #Woovovo
    Xvv = einsum('lkdc,bdkl->cb', I.oovv, RD)
    temp_ab += 0.5*einsum('ca,bcij->abij', Xvv, T2)
    Xvv = None

    #Wpvvoo
    X1oo = einsum('xki,x->ki', g.oo, R1)
    temp_ij += einsum('bakj,ki->abij', T2, X1oo)
    X1oo = None

    X1vv = einsum('xbc,x->bc', g.vv, R1)
    temp_ab += einsum('acij,bc->abij', T2, X1vv)
    X1vv = None

    X1ov = einsum('xkc,x->kc', g.ov, R1)
    Yoo = einsum('ci,kc->ki', T1, X1ov)
    Yvv = einsum('ak,kc->ac', T1, X1ov)
    temp_ij += einsum('abjk,ki->abij', T2, Yoo)
    temp_ab += einsum('bcij,ac->abij', T2, Yvv)
    X1ov = None

    XSoo = einsum('lkib,bk->li', I.ooov, RS)
    XSvv = einsum('bicd,ci->bd', I.vovv, RS)
    XSov = einsum('licd,ci->ld', I.oovv, RS)

    #Wovvvoo
    temp_ij += einsum('bakj,ki->abij', T2, XSoo)
    temp_ab += einsum('adji,bd->abij', T2, XSvv)

    XXoo = einsum('di,ld->li', T1, XSov)
    XXvv = einsum('bl,ld->bd', T1, XSov)
    temp_ij += einsum('ablj,li->abij', T2, XXoo)
    temp_ab += einsum('adij,bd->abij', T2, XXvv)

    #Wp_vovo
    sigS1 -= einsum('xak,ki->xai', U11, XSoo)
    sigS1 -= einsum('xci,ac->xai', U11, XSvv)
    Xmvv = einsum('xak,kc->xac', U11, XSov)
    Xmoo = einsum('xci,kc->xki', U11, XSov)
    sigS1 += einsum('ci,xac->xai', T1, Xmvv)
    sigS1 += einsum('ak,xki->xai', T1, Xmoo)
    Xmvv = None
    Xmoo = None

    XSvv = None
    XSoo = None
    XSov = None

    #Wpvvov
    XS1oo = einsum('xkc,xcj->kj', g.ov, RS1)
    temp_ij += einsum('abki,kj->abij', T2, XS1oo)
    XS1oo = None

    #Wpvooo
    XS1vv = numpy.einsum('xkd,xbk->bd', g.ov, RS1)
    temp_ab += einsum('bdij,ad->abij', T2, XS1vv)
    XS1vv = None

    sigD += temp_ij - temp_ij.transpose((0,1,3,2))
    sigD += temp_ab - temp_ab.transpose((1,0,2,3))

    #Wp_ovvv
    XDoo = einsum('jkbc,bcij->ki', I.oovv, RD)
    sigS1 += 0.5*einsum('xak,ki->xai', U11, XDoo)
   
    #Wp_oovo
    XDvv = einsum('jkbd,abjk->da', I.oovv, RD)
    sigS1 += 0.5*einsum('xdi,da->xai', U11, XDvv)
    
    # added by YZ
    #sig1 += 1.0*einsum('I,I->I', w, R1)
    #sigS1 += 1.0*einsum('I,Iai->Iai', w, RS1)
    #sigS1 += 1.0*einsum('I,I,ai->Iai', w, S1, RS)

    return sigS, sigD, sig1, sigS1

def eom_ip_epccsd_1_s1_sigma_slow(RS, RD, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((no))
    sigD = numpy.zeros((nv,no,no))
    sigS1 = numpy.zeros((nm,no))

    sigS += -1.0*einsum('ji,j->i', F.oo, RS)
    sigS += 1.0*einsum('I,Ii->i', G, RS1)
    sigS += -1.0*einsum('ja,aji->i', F.ov, RD)
    sigS += -1.0*einsum('Iji,Ij->i', g.oo, RS1)
    sigS += 0.5*einsum('jkia,akj->i', I.ooov, RD)
    sigS += -1.0*einsum('ja,ai,j->i', F.ov, T1, RS)
    sigS += -1.0*einsum('Iji,I,j->i', g.oo, S1, RS)
    sigS += 1.0*einsum('jkia,aj,k->i', I.ooov, T1, RS)
    sigS += -1.0*einsum('Ija,ai,Ij->i', g.ov, T1, RS1)
    sigS += 1.0*einsum('Ija,aj,Ii->i', g.ov, T1, RS1)
    sigS += -1.0*einsum('Ija,I,aji->i', g.ov, S1, RD)
    sigS += -1.0*einsum('Ija,Iai,j->i', g.ov, U11, RS)
    sigS += -0.5*einsum('jkab,bi,akj->i', I.oovv, T1, RD)
    sigS += 1.0*einsum('jkab,bj,aki->i', I.oovv, T1, RD)
    sigS += 0.5*einsum('jkab,baji,k->i', I.oovv, T2, RS)
    sigS += -1.0*einsum('Ija,ai,I,j->i', g.ov, T1, S1, RS)
    sigS += -1.0*einsum('jkab,bi,aj,k->i', I.oovv, T1, T1, RS)

    sigD += -1.0*einsum('ki,akj->aij', F.oo, RD)
    sigD += 1.0*einsum('kj,aki->aij', F.oo, RD)
    sigD += -1.0*einsum('ab,bji->aij', F.vv, RD)
    sigD += -1.0*einsum('akji,k->aij', I.vooo, RS)
    sigD += -1.0*einsum('Iai,Ij->aij', g.vo, RS1)
    sigD += 1.0*einsum('Iaj,Ii->aij', g.vo, RS1)
    sigD += 0.5*einsum('klji,alk->aij', I.oooo, RD)
    sigD += -1.0*einsum('akbi,bkj->aij', I.vovo, RD)
    sigD += 1.0*einsum('akbj,bki->aij', I.vovo, RD)
    sigD += -1.0*einsum('kb,bi,akj->aij', F.ov, T1, RD)
    sigD += 1.0*einsum('kb,bj,aki->aij', F.ov, T1, RD)
    sigD += 1.0*einsum('kb,ak,bji->aij', F.ov, T1, RD)
    sigD += -1.0*einsum('kb,abji,k->aij', F.ov, T2, RS)
    sigD += 1.0*einsum('klji,ak,l->aij', I.oooo, T1, RS)
    sigD += -1.0*einsum('akbi,bj,k->aij', I.vovo, T1, RS)
    sigD += 1.0*einsum('akbj,bi,k->aij', I.vovo, T1, RS)
    sigD += 1.0*einsum('Iki,ak,Ij->aij', g.oo, T1, RS1)
    sigD += -1.0*einsum('Ikj,ak,Ii->aij', g.oo, T1, RS1)
    sigD += -1.0*einsum('Iki,I,akj->aij', g.oo, S1, RD)
    sigD += 1.0*einsum('Ikj,I,aki->aij', g.oo, S1, RD)
    sigD += -1.0*einsum('Iki,Iaj,k->aij', g.oo, U11, RS)
    sigD += 1.0*einsum('Ikj,Iai,k->aij', g.oo, U11, RS)
    sigD += -1.0*einsum('Iab,bi,Ij->aij', g.vv, T1, RS1)
    sigD += 1.0*einsum('Iab,bj,Ii->aij', g.vv, T1, RS1)
    sigD += -1.0*einsum('Iab,I,bji->aij', g.vv, S1, RD)
    sigD += -0.5*einsum('klib,bj,alk->aij', I.ooov, T1, RD)
    sigD += -1.0*einsum('klib,ak,blj->aij', I.ooov, T1, RD)
    sigD += 1.0*einsum('klib,bk,alj->aij', I.ooov, T1, RD)
    sigD += 0.5*einsum('kljb,bi,alk->aij', I.ooov, T1, RD)
    sigD += 1.0*einsum('kljb,ak,bli->aij', I.ooov, T1, RD)
    sigD += -1.0*einsum('kljb,bk,ali->aij', I.ooov, T1, RD)
    sigD += -1.0*einsum('klib,abkj,l->aij', I.ooov, T2, RS)
    sigD += 1.0*einsum('kljb,abki,l->aij', I.ooov, T2, RS)
    sigD += -1.0*einsum('akbc,ci,bkj->aij', I.vovv, T1, RD)
    sigD += 1.0*einsum('akbc,cj,bki->aij', I.vovv, T1, RD)
    sigD += -1.0*einsum('akbc,ck,bji->aij', I.vovv, T1, RD)
    sigD += 0.5*einsum('akbc,cbji,k->aij', I.vovv, T2, RS)
    sigD += -1.0*einsum('Ikb,abji,Ik->aij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikb,abki,Ij->aij', g.ov, T2, RS1)
    sigD += -1.0*einsum('Ikb,abkj,Ii->aij', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ikb,Iai,bkj->aij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikb,Ibi,akj->aij', g.ov, U11, RD)
    sigD += -1.0*einsum('Ikb,Iaj,bki->aij', g.ov, U11, RD)
    sigD += 1.0*einsum('Ikb,Ibj,aki->aij', g.ov, U11, RD)
    sigD += 1.0*einsum('Ikb,Iak,bji->aij', g.ov, U11, RD)
    sigD += -0.5*einsum('klbc,acji,blk->aij', I.oovv, T2, RD)
    sigD += -0.25*einsum('klbc,cbji,alk->aij', I.oovv, T2, RD)
    sigD += 1.0*einsum('klbc,acki,blj->aij', I.oovv, T2, RD)
    sigD += 0.5*einsum('klbc,cbki,alj->aij', I.oovv, T2, RD)
    sigD += -1.0*einsum('klbc,ackj,bli->aij', I.oovv, T2, RD)
    sigD += -0.5*einsum('klbc,cbkj,ali->aij', I.oovv, T2, RD)
    sigD += -0.5*einsum('klbc,aclk,bji->aij', I.oovv, T2, RD)
    sigD += -1.0*einsum('klib,bj,ak,l->aij', I.ooov, T1, T1, RS)
    sigD += 1.0*einsum('kljb,bi,ak,l->aij', I.ooov, T1, T1, RS)
    sigD += -1.0*einsum('akbc,ci,bj,k->aij', I.vovv, T1, T1, RS)
    sigD += 1.0*einsum('Ikb,bi,ak,Ij->aij', g.ov, T1, T1, RS1)
    sigD += -1.0*einsum('Ikb,bj,ak,Ii->aij', g.ov, T1, T1, RS1)
    sigD += -1.0*einsum('Ikb,bi,I,akj->aij', g.ov, T1, S1, RD)
    sigD += 1.0*einsum('Ikb,bj,I,aki->aij', g.ov, T1, S1, RD)
    sigD += 1.0*einsum('Ikb,ak,I,bji->aij', g.ov, T1, S1, RD)
    sigD += -1.0*einsum('Ikb,bi,Iaj,k->aij', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ikb,bj,Iai,k->aij', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ikb,abji,I,k->aij', g.ov, T2, S1, RS)
    sigD += 0.5*einsum('klbc,ci,bj,alk->aij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klbc,ci,ak,blj->aij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klbc,ci,bk,alj->aij', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('klbc,cj,ak,bli->aij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klbc,cj,bk,ali->aij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klbc,ak,cl,bji->aij', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('klbc,ci,abkj,l->aij', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('klbc,cj,abki,l->aij', I.oovv, T1, T2, RS)
    sigD += -0.5*einsum('klbc,ak,cbji,l->aij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klbc,ck,abji,l->aij', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('klbc,ci,bj,ak,l->aij', I.oovv, T1, T1, T1, RS)

    sigS1 += -1.0*einsum('ji,Ij->Ii', F.oo, RS1)
    sigS1 += 1.0*einsum('I,Ii->Ii', w, RS1)
    sigS1 += 1.0*einsum('I,I,ai->Iai', w, S1, RS)
    sigS1 += -1.0*einsum('Iji,j->Ii', h.oo, RS)
    sigS1 += -1.0*einsum('Ija,aji->Ii', h.ov, RD)
    sigS1 += -1.0*einsum('ja,ai,Ij->Ii', F.ov, T1, RS1)
    sigS1 += -1.0*einsum('ja,Iai,j->Ii', F.ov, U11, RS)
    sigS1 += -1.0*einsum('Jji,J,Ij->Ii', g.oo, S1, RS1)
    sigS1 += -1.0*einsum('Ija,ai,j->Ii', h.ov, T1, RS)
    sigS1 += 1.0*einsum('jkia,aj,Ik->Ii', I.ooov, T1, RS1)
    sigS1 += 1.0*einsum('jkia,Iaj,k->Ii', I.ooov, U11, RS)
    sigS1 += -1.0*einsum('Jja,Iai,Jj->Ii', g.ov, U11, RS1)
    sigS1 += 1.0*einsum('Jja,Iaj,Ji->Ii', g.ov, U11, RS1)
    sigS1 += -1.0*einsum('Jja,Jai,Ij->Ii', g.ov, U11, RS1)
    sigS1 += 0.5*einsum('jkab,baji,Ik->Ii', I.oovv, T2, RS1)
    sigS1 += -0.5*einsum('jkab,Ibi,akj->Ii', I.oovv, U11, RD)
    sigS1 += 1.0*einsum('jkab,Ibj,aki->Ii', I.oovv, U11, RD)
    sigS1 += -1.0*einsum('Jja,ai,J,Ij->Ii', g.ov, T1, S1, RS1)
    sigS1 += -1.0*einsum('Jja,J,Iai,j->Ii', g.ov, S1, U11, RS)
    sigS1 += -1.0*einsum('jkab,bi,aj,Ik->Ii', I.oovv, T1, T1, RS1)
    sigS1 += -1.0*einsum('jkab,bi,Iaj,k->Ii', I.oovv, T1, U11, RS)
    sigS1 += 1.0*einsum('jkab,bj,Iai,k->Ii', I.oovv, T1, U11, RS)

    return sigS, sigD, sigS1

def eom_ip_epccsd_1_s1_sigma_int(RS, RD, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((no))
    sigD = numpy.zeros((nv,no,no))
    sigS1 = numpy.zeros((nm,no))

    Wov = numpy.zeros((no,nv))
    Wov += 1.0*einsum('ia->ia', F.ov)
    Wov += 1.0*einsum('I,Iia->ia', S1, g.ov)
    Wov += -1.0*einsum('bj,jiab->ia', T1, I.oovv)

    Woo = numpy.zeros((no,no))
    Woo += 1.0*einsum('ij->ij', F.oo)
    Woo += 1.0*einsum('aj,ia->ij', T1, F.ov)
    Woo += 1.0*einsum('I,Iij->ij', S1, g.oo)
    Woo += -1.0*einsum('ak,kija->ij', T1, I.ooov)
    Woo += 1.0*einsum('Iaj,Iia->ij', U11, g.ov)
    Woo += -0.5*einsum('abkj,kiba->ij', T2, I.oovv)
    Woo += 1.0*einsum('aj,I,Iia->ij', T1, S1, g.ov)
    Woo += 1.0*einsum('aj,bk,kiba->ij', T1, T1, I.oovv)

    Wvv = numpy.zeros((nv,nv))
    Wvv += 1.0*einsum('ab->ab', F.vv)
    Wvv += -1.0*einsum('ai,ib->ab', T1, F.ov)
    Wvv += 1.0*einsum('I,Iab->ab', S1, g.vv)
    Wvv += 1.0*einsum('ci,aibc->ab', T1, I.vovv)
    Wvv += -1.0*einsum('Iai,Iib->ab', U11, g.ov)
    Wvv += 0.5*einsum('acij,jibc->ab', T2, I.oovv)
    Wvv += -1.0*einsum('ai,I,Iib->ab', T1, S1, g.ov)
    Wvv += -1.0*einsum('ai,cj,ijbc->ab', T1, T1, I.oovv)

    Woooo = numpy.zeros((no,no,no,no))
    Woooo += 1.0*einsum('jilk->ijkl', I.oooo)
    Woooo += -1.0*einsum('al,jika->ijkl', T1, I.ooov)
    Woooo += 1.0*einsum('ak,jila->ijkl', T1, I.ooov)
    Woooo += -0.5*einsum('ablk,jiba->ijkl', T2, I.oovv)
    Woooo += 1.0*einsum('ak,bl,jiba->ijkl', T1, T1, I.oovv)

    Wooov = numpy.zeros((no,no,no,nv))
    Wooov += -1.0*einsum('jika->ijka', I.ooov)
    Wooov += 1.0*einsum('bk,jiab->ijka', T1, I.oovv)

    Wvoov = numpy.zeros((nv,no,no,nv))
    Wvoov += -1.0*einsum('aibj->aijb', I.vovo)
    Wvoov += -1.0*einsum('ak,kijb->aijb', T1, I.ooov)
    Wvoov += -1.0*einsum('cj,aibc->aijb', T1, I.vovv)
    Wvoov += 1.0*einsum('Iaj,Iib->aijb', U11, g.ov)
    Wvoov += 1.0*einsum('ackj,kibc->aijb', T2, I.oovv)
    Wvoov += 1.0*einsum('cj,ak,kibc->aijb', T1, T1, I.oovv)

    Wovoo = numpy.zeros((no,nv,no,no))
    Wovoo += 1.0*einsum('aikj->iajk', I.vooo)
    Wovoo += 1.0*einsum('abkj,ib->iajk', T2, F.ov)
    Wovoo += -1.0*einsum('al,likj->iajk', T1, I.oooo)
    Wovoo += 1.0*einsum('bk,aibj->iajk', T1, I.vovo)
    Wovoo += -1.0*einsum('bj,aibk->iajk', T1, I.vovo)
    Wovoo += 1.0*einsum('Iak,Iij->iajk', U11, g.oo)
    Wovoo += -1.0*einsum('Iaj,Iik->iajk', U11, g.oo)
    Wovoo += 1.0*einsum('ablk,lijb->iajk', T2, I.ooov)
    Wovoo += -1.0*einsum('ablj,likb->iajk', T2, I.ooov)
    Wovoo += -0.5*einsum('bckj,aicb->iajk', T2, I.vovv)
    Wovoo += 1.0*einsum('bk,al,lijb->iajk', T1, T1, I.ooov)
    Wovoo += -1.0*einsum('bj,al,likb->iajk', T1, T1, I.ooov)
    Wovoo += 1.0*einsum('bj,ck,aicb->iajk', T1, T1, I.vovv)
    Wovoo += 1.0*einsum('bj,Iak,Iib->iajk', T1, U11, g.ov)
    Wovoo += -1.0*einsum('bk,Iaj,Iib->iajk', T1, U11, g.ov)
    Wovoo += 1.0*einsum('abkj,I,Iib->iajk', T2, S1, g.ov)
    Wovoo += -1.0*einsum('bj,aclk,licb->iajk', T1, T2, I.oovv)
    Wovoo += 1.0*einsum('bk,aclj,licb->iajk', T1, T2, I.oovv)
    Wovoo += 0.5*einsum('al,bckj,licb->iajk', T1, T2, I.oovv)
    Wovoo += -1.0*einsum('bl,ackj,licb->iajk', T1, T2, I.oovv)
    Wovoo += -1.0*einsum('bj,ck,al,licb->iajk', T1, T1, T1, I.oovv)

    Woovovo = numpy.zeros((no,no,nv,no,nv,no))
    Woovovo += 1.0*einsum('acij,klbc->lkajbi', T2, I.oovv)

    Wp = numpy.zeros((nm))
    Wp += 1.0*einsum('I->I', G)
    Wp += 1.0*einsum('ai,Iia->I', T1, g.ov)

    Wpoo = numpy.zeros((nm,no,no))
    Wpoo += 1.0*einsum('Iji->Iji', g.oo)
    Wpoo += 1.0*einsum('ai,Ija->Iji', T1, g.ov)

    Wpvo = numpy.zeros((nm,nv,no))
    Wpvo += 1.0*einsum('Iai->Iai', g.vo)
    Wpvo += -1.0*einsum('aj,Iji->Iai', T1, g.oo)
    Wpvo += 1.0*einsum('bi,Iab->Iai', T1, g.vv)
    Wpvo += -1.0*einsum('abji,Ijb->Iai', T2, g.ov)
    Wpvo += -1.0*einsum('bi,aj,Ijb->Iai', T1, T1, g.ov)

    Wp_oo = numpy.zeros((nm,no,no))
    Wp_oo += 1.0*einsum('Iji->Iji', h.oo)
    Wp_oo += 1.0*einsum('Iai,ja->Iji', U11, F.ov)
    Wp_oo += 1.0*einsum('ai,Ija->Iji', T1, h.ov)
    Wp_oo += -1.0*einsum('Iak,kjia->Iji', U11, I.ooov)
    Wp_oo += 1.0*einsum('J,Iai,Jja->Iji', S1, U11, g.ov)
    Wp_oo += 1.0*einsum('ai,Ibk,kjba->Iji', T1, U11, I.oovv)
    Wp_oo += -1.0*einsum('ak,Ibi,kjba->Iji', T1, U11, I.oovv)

    Wp_ov = numpy.zeros((nm,no,nv))
    Wp_ov += 1.0*einsum('Iia->Iia', h.ov)
    Wp_ov += -1.0*einsum('Ibj,jiab->Iia', U11, I.oovv)

    Wp_oovo = numpy.zeros((nm,no,no,nv,no))
    Wp_oovo += -1.0*einsum('Ibk,jiab->Iijak', U11, I.oovv)

    Wp_p = numpy.zeros((nm,nm))
    Wp_p += numpy.diag(w)
    Wp_p += 1.0*einsum('Iai,Jia->IJ', U11, g.ov)

    Wp_poo = numpy.zeros((nm,nm,no,no))
    Wp_poo += 1.0*einsum('Iaj,Jia->IJij', U11, g.ov)

    Wpvooo = numpy.zeros((nm,nv,no,no,no))
    Wpvooo += -1.0*einsum('abkj,Iib->Iaijk', T2, g.ov)

    sigS += -1.0*einsum('ji,j->i', Woo, RS)
    sigS += 1.0*einsum('I,Ii->i', Wp, RS1)
    sigS += -1.0*einsum('ja,aji->i', Wov, RD)
    sigS += -1.0*einsum('Iji,Ij->i', Wpoo, RS1)
    sigS += 0.5*einsum('jkia,akj->i', Wooov, RD)

    sigD += -1.0*einsum('ki,akj->aij', Woo, RD)
    sigD += 1.0*einsum('kj,aki->aij', Woo, RD)
    sigD += -1.0*einsum('ab,bji->aij', Wvv, RD)
    sigD += 1.0*einsum('kaji,k->aij', Wovoo, RS)
    sigD += -1.0*einsum('Iai,Ij->aij', Wpvo, RS1)
    sigD += 1.0*einsum('Iaj,Ii->aij', Wpvo, RS1)
    sigD += 0.5*einsum('klji,alk->aij', Woooo, RD)
    sigD += 1.0*einsum('akib,bkj->aij', Wvoov, RD)
    sigD += -1.0*einsum('akjb,bki->aij', Wvoov, RD)

    sigD += 1.0*einsum('Iakij,Ik->aij',Wpvooo, RS1)
    sigD += 0.5*einsum('lkajbi,blk->aij',Woovovo, RD)

    sigS1 += -1.0*einsum('ji,Ij->Ii', Woo, RS1)
    sigS1 += 1.0*einsum('IJ,Ji->Ii', Wp_p, RS1)
    sigS1 += -1.0*einsum('Iji,j->Ii', Wp_oo, RS)
    sigS1 += -1.0*einsum('Ija,aji->Ii', Wp_ov, RD)

    sigS1 += 0.5*einsum('Ijkbi,bjk->Ii', Wp_oovo, RD)
    sigS1 += -einsum('IJji,Jj->Ii', Wp_poo, RS1)

    return sigS, sigD, sigS1

def eom_ip_epccsd_1_s1_sigma_opt(RS, RD, RS1, amps, F, I, w, g, h, G, H, imds=None):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((no),dtype=RS.dtype)
    sigD = numpy.zeros((nv,no,no),dtype=RD.dtype)
    sigS1 = numpy.zeros((nm,no),dtype=RS1.dtype)

    if imds is None:
        imds = IPImds()
        imds.build(amps, F, I, w, g, h, G, H)

    sigS -= einsum('ji,j->i', imds.Woo, RS)
    sigS += einsum('x,xi->i', imds.Wp, RS1)
    sigS -= einsum('ja,aji->i', imds.Wov, RD)
    sigS -= einsum('xji,xj->i', imds.Wpoo, RS1)
    sigS += 0.5*einsum('jkia,akj->i', imds.Wooov, RD)

    temp_ij = einsum('kj,aki->aij', imds.Woo, RD)
    temp_ij += einsum('xaj,xi->aij', imds.Wpvo, RS1)
    temp_ij += einsum('akib,bkj->aij', imds.Wvoov, RD)

    sigD += 0.5*einsum('klji,alk->aij', imds.Woooo, RD)
    sigD += -1.0*einsum('ab,bji->aij', imds.Wvv, RD)
    sigD += 1.0*einsum('kaji,k->aij', imds.Wovoo, RS)
    sigD += temp_ij - temp_ij.transpose((0,2,1))

    sigS1 -= einsum('ji,xj->xi', imds.Woo, RS1)
    sigS1 += einsum('xy,yi->xi', imds.Wp_p, RS1)
    sigS1 -= einsum('xji,j->xi', imds.Wp_oo, RS)
    sigS1 -= einsum('xja,aji->xi', imds.Wp_ov, RD)
    sigS1 -= einsum('xyji,yj->xi', imds.Wp_poo, RS1)

    #Woovovo
    XDv = einsum('klbc,blk->c', I.oovv, RD)
    sigD += 0.5*einsum('acij,c->aij', T2, XDv)

    #Wpvooo
    XS1v = einsum('xkb,xk->b', g.ov, RS1)
    sigD += einsum('abij,b->aij', T2, XS1v)

    #Wp_oovo
    XDv = einsum('jkbc,bjk->c', I.oovv, RD)
    sigS1 += 0.5*einsum('xci,c->xi', U11, XDv)

    return sigS, sigD, sigS1

def eom_ea_epccsd_1_s1_sigma_slow(RS, RD, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((nv))
    sigD = numpy.zeros((nv,nv,no))
    sigS1 = numpy.zeros((nm,nv))

    sigS += 1.0*einsum('ab,b->a', F.vv, RS)
    sigS += 1.0*einsum('I,Ia->a', G, RS1)
    sigS += 1.0*einsum('ib,abi->a', F.ov, RD)
    sigS += 1.0*einsum('Iab,Ib->a', g.vv, RS1)
    sigS += -0.5*einsum('aibc,cbi->a', I.vovv, RD)
    sigS += -1.0*einsum('ib,ai,b->a', F.ov, T1, RS)
    sigS += 1.0*einsum('Iab,I,b->a', g.vv, S1, RS)
    sigS += 1.0*einsum('aibc,ci,b->a', I.vovv, T1, RS)
    sigS += -1.0*einsum('Iib,ai,Ib->a', g.ov, T1, RS1)
    sigS += 1.0*einsum('Iib,bi,Ia->a', g.ov, T1, RS1)
    sigS += 1.0*einsum('Iib,I,abi->a', g.ov, S1, RD)
    sigS += -1.0*einsum('Iib,Iai,b->a', g.ov, U11, RS)
    sigS += 0.5*einsum('ijbc,ai,cbj->a', I.oovv, T1, RD)
    sigS += -1.0*einsum('ijbc,ci,abj->a', I.oovv, T1, RD)
    sigS += 0.5*einsum('ijbc,acji,b->a', I.oovv, T2, RS)
    sigS += -1.0*einsum('Iib,ai,I,b->a', g.ov, T1, S1, RS)
    sigS += -1.0*einsum('ijbc,ai,cj,b->a', I.oovv, T1, T1, RS)

    sigD += 1.0*einsum('ji,baj->abi', F.oo, RD)
    sigD += 1.0*einsum('bc,aci->abi', F.vv, RD)
    sigD += -1.0*einsum('ac,bci->abi', F.vv, RD)
    sigD += -1.0*einsum('baci,c->abi', I.vvvo, RS)
    sigD += 1.0*einsum('Ibi,Ia->abi', g.vo, RS1)
    sigD += -1.0*einsum('Iai,Ib->abi', g.vo, RS1)
    sigD += -1.0*einsum('bjci,acj->abi', I.vovo, RD)
    sigD += 1.0*einsum('ajci,bcj->abi', I.vovo, RD)
    sigD += 0.5*einsum('bacd,dci->abi', I.vvvv, RD)
    sigD += 1.0*einsum('jc,ci,baj->abi', F.ov, T1, RD)
    sigD += -1.0*einsum('jc,bj,aci->abi', F.ov, T1, RD)
    sigD += 1.0*einsum('jc,aj,bci->abi', F.ov, T1, RD)
    sigD += 1.0*einsum('jc,baji,c->abi', F.ov, T2, RS)
    sigD += 1.0*einsum('bjci,aj,c->abi', I.vovo, T1, RS)
    sigD += -1.0*einsum('ajci,bj,c->abi', I.vovo, T1, RS)
    sigD += -1.0*einsum('bacd,di,c->abi', I.vvvv, T1, RS)
    sigD += -1.0*einsum('Iji,bj,Ia->abi', g.oo, T1, RS1)
    sigD += 1.0*einsum('Iji,aj,Ib->abi', g.oo, T1, RS1)
    sigD += 1.0*einsum('Iji,I,baj->abi', g.oo, S1, RD)
    sigD += 1.0*einsum('Ibc,ci,Ia->abi', g.vv, T1, RS1)
    sigD += -1.0*einsum('Iac,ci,Ib->abi', g.vv, T1, RS1)
    sigD += 1.0*einsum('Ibc,I,aci->abi', g.vv, S1, RD)
    sigD += -1.0*einsum('Iac,I,bci->abi', g.vv, S1, RD)
    sigD += -1.0*einsum('Ibc,Iai,c->abi', g.vv, U11, RS)
    sigD += 1.0*einsum('Iac,Ibi,c->abi', g.vv, U11, RS)
    sigD += -1.0*einsum('jkic,bj,ack->abi', I.ooov, T1, RD)
    sigD += 1.0*einsum('jkic,aj,bck->abi', I.ooov, T1, RD)
    sigD += -1.0*einsum('jkic,cj,bak->abi', I.ooov, T1, RD)
    sigD += -0.5*einsum('jkic,bakj,c->abi', I.ooov, T2, RS)
    sigD += -1.0*einsum('bjcd,di,acj->abi', I.vovv, T1, RD)
    sigD += 1.0*einsum('ajcd,di,bcj->abi', I.vovv, T1, RD)
    sigD += -0.5*einsum('bjcd,aj,dci->abi', I.vovv, T1, RD)
    sigD += 1.0*einsum('bjcd,dj,aci->abi', I.vovv, T1, RD)
    sigD += 0.5*einsum('ajcd,bj,dci->abi', I.vovv, T1, RD)
    sigD += -1.0*einsum('ajcd,dj,bci->abi', I.vovv, T1, RD)
    sigD += 1.0*einsum('bjcd,adji,c->abi', I.vovv, T2, RS)
    sigD += -1.0*einsum('ajcd,bdji,c->abi', I.vovv, T2, RS)
    sigD += 1.0*einsum('Ijc,baji,Ic->abi', g.ov, T2, RS1)
    sigD += -1.0*einsum('Ijc,bcji,Ia->abi', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ijc,acji,Ib->abi', g.ov, T2, RS1)
    sigD += 1.0*einsum('Ijc,Ibi,acj->abi', g.ov, U11, RD)
    sigD += -1.0*einsum('Ijc,Iai,bcj->abi', g.ov, U11, RD)
    sigD += 1.0*einsum('Ijc,Ici,baj->abi', g.ov, U11, RD)
    sigD += -1.0*einsum('Ijc,Ibj,aci->abi', g.ov, U11, RD)
    sigD += 1.0*einsum('Ijc,Iaj,bci->abi', g.ov, U11, RD)
    sigD += -0.5*einsum('jkcd,baji,dck->abi', I.oovv, T2, RD)
    sigD += 1.0*einsum('jkcd,bdji,ack->abi', I.oovv, T2, RD)
    sigD += -1.0*einsum('jkcd,adji,bck->abi', I.oovv, T2, RD)
    sigD += -0.5*einsum('jkcd,dcji,bak->abi', I.oovv, T2, RD)
    sigD += -0.25*einsum('jkcd,bakj,dci->abi', I.oovv, T2, RD)
    sigD += 0.5*einsum('jkcd,bdkj,aci->abi', I.oovv, T2, RD)
    sigD += -0.5*einsum('jkcd,adkj,bci->abi', I.oovv, T2, RD)
    sigD += 1.0*einsum('jkic,bj,ak,c->abi', I.ooov, T1, T1, RS)
    sigD += 1.0*einsum('bjcd,di,aj,c->abi', I.vovv, T1, T1, RS)
    sigD += -1.0*einsum('ajcd,di,bj,c->abi', I.vovv, T1, T1, RS)
    sigD += -1.0*einsum('Ijc,ci,bj,Ia->abi', g.ov, T1, T1, RS1)
    sigD += 1.0*einsum('Ijc,ci,aj,Ib->abi', g.ov, T1, T1, RS1)
    sigD += 1.0*einsum('Ijc,ci,I,baj->abi', g.ov, T1, S1, RD)
    sigD += -1.0*einsum('Ijc,bj,I,aci->abi', g.ov, T1, S1, RD)
    sigD += 1.0*einsum('Ijc,aj,I,bci->abi', g.ov, T1, S1, RD)
    sigD += 1.0*einsum('Ijc,bj,Iai,c->abi', g.ov, T1, U11, RS)
    sigD += -1.0*einsum('Ijc,aj,Ibi,c->abi', g.ov, T1, U11, RS)
    sigD += 1.0*einsum('Ijc,baji,I,c->abi', g.ov, T2, S1, RS)
    sigD += 1.0*einsum('jkcd,di,bj,ack->abi', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('jkcd,di,aj,bck->abi', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('jkcd,di,cj,bak->abi', I.oovv, T1, T1, RD)
    sigD += 0.5*einsum('jkcd,bj,ak,dci->abi', I.oovv, T1, T1, RD)
    sigD += -1.0*einsum('jkcd,bj,dk,aci->abi', I.oovv, T1, T1, RD)
    sigD += 1.0*einsum('jkcd,aj,dk,bci->abi', I.oovv, T1, T1, RD)
    sigD += 0.5*einsum('jkcd,di,bakj,c->abi', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('jkcd,bj,adki,c->abi', I.oovv, T1, T2, RS)
    sigD += 1.0*einsum('jkcd,aj,bdki,c->abi', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('jkcd,dj,baki,c->abi', I.oovv, T1, T2, RS)
    sigD += -1.0*einsum('jkcd,di,bj,ak,c->abi', I.oovv, T1, T1, T1, RS)

    sigS1 += 1.0*einsum('ab,Ib->Ia', F.vv, RS1)
    sigS1 += 1.0*einsum('I,Ia->Ia', w, RS1)
    sigS1 += 1.0*einsum('Iab,b->Ia', h.vv, RS)
    sigS1 += 1.0*einsum('Iib,abi->Ia', h.ov, RD)
    sigS1 += -1.0*einsum('ib,ai,Ib->Ia', F.ov, T1, RS1)
    sigS1 += -1.0*einsum('ib,Iai,b->Ia', F.ov, U11, RS)
    sigS1 += -1.0*einsum('Iib,ai,b->Ia', h.ov, T1, RS)
    sigS1 += 1.0*einsum('Jab,J,Ib->Ia', g.vv, S1, RS1)
    sigS1 += 1.0*einsum('aibc,ci,Ib->Ia', I.vovv, T1, RS1)
    sigS1 += 1.0*einsum('aibc,Ici,b->Ia', I.vovv, U11, RS)
    sigS1 += -1.0*einsum('Jib,Iai,Jb->Ia', g.ov, U11, RS1)
    sigS1 += 1.0*einsum('Jib,Ibi,Ja->Ia', g.ov, U11, RS1)
    sigS1 += -1.0*einsum('Jib,Jai,Ib->Ia', g.ov, U11, RS1)
    sigS1 += 0.5*einsum('ijbc,acji,Ib->Ia', I.oovv, T2, RS1)
    sigS1 += 0.5*einsum('ijbc,Iai,cbj->Ia', I.oovv, U11, RD)
    sigS1 += -1.0*einsum('ijbc,Ici,abj->Ia', I.oovv, U11, RD)
    sigS1 += -1.0*einsum('Jib,ai,J,Ib->Ia', g.ov, T1, S1, RS1)
    sigS1 += -1.0*einsum('Jib,J,Iai,b->Ia', g.ov, S1, U11, RS)
    sigS1 += -1.0*einsum('ijbc,ai,cj,Ib->Ia', I.oovv, T1, T1, RS1)
    sigS1 += -1.0*einsum('ijbc,ai,Icj,b->Ia', I.oovv, T1, U11, RS)
    sigS1 += 1.0*einsum('ijbc,ci,Iaj,b->Ia', I.oovv, T1, U11, RS)

    return sigS, sigD, sigS1

def eom_ea_epccsd_1_s1_sigma_int(RS, RD, RS1, amps, F, I, w, g, h, G, H):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((nv))
    sigD = numpy.zeros((nv,nv,no))
    sigS1 = numpy.zeros((nm,nv))

    Wov = numpy.zeros((no,nv))
    Wov += 1.0*einsum('ia->ia', F.ov)
    Wov += 1.0*einsum('I,Iia->ia', S1, g.ov)
    Wov += -1.0*einsum('bj,jiab->ia', T1, I.oovv)

    Woo = numpy.zeros((no,no))
    Woo += 1.0*einsum('ij->ij', F.oo)
    Woo += 1.0*einsum('aj,ia->ij', T1, F.ov)
    Woo += 1.0*einsum('I,Iij->ij', S1, g.oo)
    Woo += -1.0*einsum('ak,kija->ij', T1, I.ooov)
    Woo += 1.0*einsum('Iaj,Iia->ij', U11, g.ov)
    Woo += -0.5*einsum('abkj,kiba->ij', T2, I.oovv)
    Woo += 1.0*einsum('aj,I,Iia->ij', T1, S1, g.ov)
    Woo += 1.0*einsum('aj,bk,kiba->ij', T1, T1, I.oovv)

    Wvv = numpy.zeros((nv,nv))
    Wvv += 1.0*einsum('ab->ab', F.vv)
    Wvv += -1.0*einsum('ai,ib->ab', T1, F.ov)
    Wvv += 1.0*einsum('I,Iab->ab', S1, g.vv)
    Wvv += 1.0*einsum('ci,aibc->ab', T1, I.vovv)
    Wvv += -1.0*einsum('Iai,Iib->ab', U11, g.ov)
    Wvv += 0.5*einsum('acij,jibc->ab', T2, I.oovv)
    Wvv += -1.0*einsum('ai,I,Iib->ab', T1, S1, g.ov)
    Wvv += -1.0*einsum('ai,cj,ijbc->ab', T1, T1, I.oovv)

    Wvovv = numpy.zeros((nv,no,nv,nv))
    Wvovv += -1.0*einsum('aicb->aibc', I.vovv)
    Wvovv += 1.0*einsum('aj,jicb->aibc', T1, I.oovv)

    Wvvvo = numpy.zeros((nv,nv,nv,no))
    Wvvvo += -1.0*einsum('baci->abci', I.vvvo)
    Wvvvo += 1.0*einsum('baji,jc->abci', T2, F.ov)
    Wvvvo += 1.0*einsum('aj,bjci->abci', T1, I.vovo)
    Wvvvo += -1.0*einsum('bj,ajci->abci', T1, I.vovo)
    Wvvvo += -1.0*einsum('di,bacd->abci', T1, I.vvvv)
    Wvvvo += -1.0*einsum('Iai,Ibc->abci', U11, g.vv)
    Wvvvo += 1.0*einsum('Ibi,Iac->abci', U11, g.vv)
    Wvvvo += -0.5*einsum('bajk,kjic->abci', T2, I.ooov)
    Wvvvo += 1.0*einsum('adji,bjcd->abci', T2, I.vovv)
    Wvvvo += -1.0*einsum('bdji,ajcd->abci', T2, I.vovv)
    Wvvvo += 1.0*einsum('bj,ak,jkic->abci', T1, T1, I.ooov)
    Wvvvo += 1.0*einsum('di,aj,bjcd->abci', T1, T1, I.vovv)
    Wvvvo += -1.0*einsum('di,bj,ajcd->abci', T1, T1, I.vovv)
    Wvvvo += 1.0*einsum('bj,Iai,Ijc->abci', T1, U11, g.ov)
    Wvvvo += -1.0*einsum('aj,Ibi,Ijc->abci', T1, U11, g.ov)
    Wvvvo += 1.0*einsum('baji,I,Ijc->abci', T2, S1, g.ov)
    Wvvvo += 0.5*einsum('di,bajk,kjcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('bj,adki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += 1.0*einsum('aj,bdki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('dj,baki,jkcd->abci', T1, T2, I.oovv)
    Wvvvo += -1.0*einsum('di,bj,ak,jkcd->abci', T1, T1, T1, I.oovv)

    Wvoov = numpy.zeros((nv,no,no,nv))
    Wvoov += -1.0*einsum('aibj->aijb', I.vovo)
    Wvoov += -1.0*einsum('ak,kijb->aijb', T1, I.ooov)
    Wvoov += -1.0*einsum('cj,aibc->aijb', T1, I.vovv)
    Wvoov += 1.0*einsum('Iaj,Iib->aijb', U11, g.ov)
    Wvoov += 1.0*einsum('ackj,kibc->aijb', T2, I.oovv)
    Wvoov += 1.0*einsum('cj,ak,kibc->aijb', T1, T1, I.oovv)

    Wvvvv = numpy.zeros((nv,nv,nv,nv))
    Wvvvv += 1.0*einsum('badc->abcd', I.vvvv)
    Wvvvv += -1.0*einsum('ai,bidc->abcd', T1, I.vovv)
    Wvvvv += 1.0*einsum('bi,aidc->abcd', T1, I.vovv)
    Wvvvv += -0.5*einsum('baij,jidc->abcd', T2, I.oovv)
    Wvvvv += 1.0*einsum('bi,aj,ijdc->abcd', T1, T1, I.oovv)

    Wvovvvo = numpy.zeros((nv,no,nv,nv,nv,no))
    Wvovvvo += -1.0*einsum('bakj,kidc->aibcdj', T2, I.oovv)

    Wpvvov = numpy.zeros((nm,nv,nv,no,nv))
    Wpvvov += -1.0*einsum('baji,Ijc->Iabic', T2, g.ov)

    Wp = numpy.zeros((nm))
    Wp += 1.0*einsum('I->I', G)
    Wp += 1.0*einsum('ai,Iia->I', T1, g.ov)

    Wp_p = numpy.zeros((nm,nm))
    Wp_p += numpy.diag(w)
    Wp_p += 1.0*einsum('Iai,Jia->IJ', U11, g.ov)

    Wp_pvv = numpy.zeros((nm,nm,nv,nv))
    Wp_pvv += -1.0*einsum('Iai,Jib->IJab', U11, g.ov)

    Wpvv = numpy.zeros((nm,nv,nv))
    Wpvv += 1.0*einsum('Iab->Iab', g.vv)
    Wpvv += -1.0*einsum('ai,Iib->Iab', T1, g.ov)

    Wpvo = numpy.zeros((nm,nv,no))
    Wpvo += 1.0*einsum('Iai->Iai', g.vo)
    Wpvo += -1.0*einsum('aj,Iji->Iai', T1, g.oo)
    Wpvo += 1.0*einsum('bi,Iab->Iai', T1, g.vv)
    Wpvo += -1.0*einsum('abji,Ijb->Iai', T2, g.ov)
    Wpvo += -1.0*einsum('bi,aj,Ijb->Iai', T1, T1, g.ov)

    Wp_vv = numpy.zeros((nm,nv,nv))
    Wp_vv += 1.0*einsum('Iab->Iab', h.vv)
    Wp_vv += -1.0*einsum('Iai,ib->Iab', U11, F.ov)
    Wp_vv += -1.0*einsum('ai,Iib->Iab', T1, h.ov)
    Wp_vv += 1.0*einsum('Ici,aibc->Iab', U11, I.vovv)
    Wp_vv += -1.0*einsum('J,Iai,Jib->Iab', S1, U11, g.ov)
    Wp_vv += -1.0*einsum('ai,Icj,ijbc->Iab', T1, U11, I.oovv)
    Wp_vv += 1.0*einsum('ci,Iaj,ijbc->Iab', T1, U11, I.oovv)

    Wp_ov = numpy.zeros((nm,no,nv))
    Wp_ov += 1.0*einsum('Iia->Iia', h.ov)
    Wp_ov += -1.0*einsum('Ibj,jiab->Iia', U11, I.oovv)

    Wp_ovvv = numpy.zeros((nm,no,nv,nv,nv))
    Wp_ovvv += -1.0*einsum('Iaj,jicb->Iiabc', U11, I.oovv)

    sigS += 1.0*einsum('ab,b->a', Wvv, RS)
    sigS += 1.0*einsum('I,Ia->a', Wp, RS1)
    sigS += 1.0*einsum('ib,abi->a', Wov, RD)
    sigS += 1.0*einsum('Iab,Ib->a', Wpvv, RS1)
    sigS += -0.5*einsum('aibc,cbi->a', Wvovv, RD)

    sigD += 1.0*einsum('ji,baj->abi', Woo, RD)
    sigD += 1.0*einsum('bc,aci->abi', Wvv, RD)
    sigD += -1.0*einsum('ac,bci->abi', Wvv, RD)
    sigD += -1.0*einsum('baci,c->abi', Wvvvo, RS)
    sigD += 1.0*einsum('Ibi,Ia->abi', Wpvo, RS1)
    sigD += -1.0*einsum('Iai,Ib->abi', Wpvo, RS1)
    sigD += 1.0*einsum('bjic,acj->abi', Wvoov, RD)
    sigD += -1.0*einsum('ajic,bcj->abi', Wvoov, RD)
    sigD += 0.5*einsum('bacd,dci->abi', Wvvvv, RD)

    sigD += 0.5*einsum('blacdi,dcl->abi', Wvovvvo, RD)
    sigD += -1.0*einsum('Iabic,Ic->abi', Wpvvov, RS1)

    sigS1 += 1.0*einsum('ab,Ib->Ia', Wvv, RS1)
    sigS1 += 1.0*einsum('IJ,Ja->Ia', Wp_p, RS1)
    sigS1 += 1.0*einsum('Iab,b->Ia', Wp_vv, RS)
    sigS1 += 1.0*einsum('Iib,abi->Ia', Wp_ov, RD)

    sigS1 += -0.5*einsum('Ijabc,bcj->Ia', Wp_ovvv, RD)
    sigS1 += 1.0*einsum('IJab,Jb->Ia', Wp_pvv, RS1)

    return sigS, sigD, sigS1

def eom_ea_epccsd_1_s1_sigma_opt(RS, RD, RS1, amps, F, I, w, g, h, G, H, imds=None):
    T1,T2,S1,U11 = amps
    nv,no = T1.shape
    nm = S1.shape[0]

    sigS = numpy.zeros((nv),dtype=RS.dtype)
    sigD = numpy.zeros((nv,nv,no),dtype=RD.dtype)
    sigS1 = numpy.zeros((nm,nv),dtype=RS1.dtype)

    if imds is None:
        imds = EAImds()
        imds.build(amps, F, I, w, g, h, G, H)

    sigS += einsum('ab,b->a', imds.Wvv, RS)
    sigS += einsum('x,xa->a', imds.Wp, RS1)
    sigS += einsum('ib,abi->a', imds.Wov, RD)
    sigS += einsum('xab,xb->a', imds.Wpvv, RS1)
    sigS -= 0.5*einsum('aibc,cbi->a', imds.Wvovv, RD)

    sigD += einsum('ji,baj->abi', imds.Woo, RD)
    sigD -= einsum('baci,c->abi', imds.Wvvvo, RS)
    sigD += 0.5*einsum('bacd,dci->abi', imds.Wvvvv, RD)

    temp_ab = einsum('bc,aci->abi', imds.Wvv, RD)
    temp_ab += einsum('xbi,xa->abi', imds.Wpvo, RS1)
    temp_ab += einsum('bjic,acj->abi', imds.Wvoov, RD)
    sigD += temp_ab - temp_ab.transpose((1,0,2))

    sigS1 += einsum('ab,xb->xa', imds.Wvv, RS1)
    sigS1 += einsum('xy,ya->xa', imds.Wp_p, RS1)
    sigS1 += einsum('xab,b->xa', imds.Wp_vv, RS)
    sigS1 += einsum('xib,abi->xa', imds.Wp_ov, RD)
    sigS1 += einsum('xyab,yb->xa', imds.Wp_pvv, RS1)

    #Wvovvvo
    XDo = einsum('kjdc,dcj->k', I.oovv, RD)
    sigD += 0.5*einsum('abik,k->abi', T2, XDo)

    #Wpvvov
    XS1o = einsum('xjc,xc->j', g.ov, RS1)
    sigD += einsum('abij,j->abi', T2, XS1o)

    #Wp_ovvv
    XDo = einsum('jkbc,bcj->k', I.oovv, RD)
    sigS1 += 0.5*einsum('xak,k->xa', U11, XDo)

    return sigS, sigD, sigS1

def eom_ee_epccsd_n_sn_sigma_opt(nfock1, nfock2, RS, RD,
        Rn, RSn, amps, F, I, w, g, h, G, H, imds=None):


    # start of the copy of eom_ee_epccsd_1_s1_sigma_opt (with minor modification)
    T1,T2,Sn,U1n = amps
    
    #print('test-zy: in epccsd_n_sn_sigma_opt', nfock1)

    nm,nv,no = U1n[0].shape

    amps_s1_U11 = (T1, T2, Sn[0], U1n[0]) # Create small amps
    if imds is None:
        imds = EEImds() # extend it for general nfock
        imds.build(amps_s1_U11, F, I, w, g, h, G, H)

    sigS = numpy.zeros(T1.shape)
    sigD = numpy.zeros(T2.shape)
    
    sign = [None] * nfock1
    for k in range(nfock1):
        sign[k] = numpy.zeros(Sn[k].shape)

    sigSn = [None] * nfock2
    for k in range(nfock2):
        sigSn[k] = numpy.zeros(U1n[k].shape)

    # Define individual variables to match code generator
    sig1 = sign[0]
    sigS1 = sigSn[0]

    R1 = Rn[0]
    RS1 = RSn[0]
    S1 = Sn[0]
    U11 = U1n[0]

    if ( nfock1 > 1 ):
        R2 = Rn[1]
        S2 = Sn[1]
    if ( nfock1 > 2 ):
        R3 = Rn[2]
        S3 = Sn[2]
    if ( nfock1 > 3 ):
        R4 = Rn[3]
        S4 = Sn[3]
    if ( nfock1 > 4 ):
        R5 = Rn[4]
        S5 = Sn[4]
    if ( nfock1 > 5 ):
        assert(False), "nfock1 >= 6 not yet implemented."

    if ( nfock2 > 1 ):
        assert(False), "nfock2 >= 2 not yet implemented."
    
    sigS -= einsum('ji,aj->ai', imds.Woo, RS)
    sigS += einsum('ab,bi->ai', imds.Wvv, RS)
    sigS += einsum('x,xai->ai', imds.Wp, RS1)
    sigS += einsum('xai,x->ai', imds.Wpvo, R1)
    sigS -= einsum('jb,abji->ai', imds.Wov, RD)
    sigS += einsum('ajib,bj->ai', imds.Wvoov, RS)
    sigS -= einsum('xji,xaj->ai', imds.Wpoo, RS1)
    sigS += einsum('xab,xbi->ai', imds.Wpvv, RS1)
    sigS += 0.5*einsum('jkib,abkj->ai', imds.Wooov, RD)
    sigS += 0.5*einsum('ajbc,cbji->ai', imds.Wvovv, RD)

    temp_ij = einsum('ki,bakj->abij', imds.Woo, RD)
    temp_ab = einsum('ac,bcji->abij', imds.Wvv, RD)
    temp_ab += einsum('kbji,ak->abij', imds.Wovoo, RS)
    temp_ij += einsum('baci,cj->abij', imds.Wvvvo, RS)
    temp_ijab = einsum('xai,xbj->abij', imds.Wpvo, RS1)
    temp_ijab += einsum('bkic,ackj->abij', imds.Wvoov, RD)
    sigD += -0.5*einsum('klji,balk->abij', imds.Woooo, RD)
    sigD += -0.5*einsum('bacd,dcji->abij', imds.Wvvvv, RD)
    sigD += temp_ijab
    sigD -= temp_ijab.transpose((0,1,3,2))
    sigD -= temp_ijab.transpose((1,0,2,3))
    sigD += temp_ijab.transpose((1,0,3,2))

    sig1 += 1.0*einsum('xy,y->x', imds.Wp_p, R1)
    sig1 += 1.0*einsum('ia,xai->x', imds.Wov, RS1)
    sig1 += 1.0*einsum('xia,ai->x', imds.Wp_ov, RS)

    sigS1 -= einsum('ji,xaj->xai', imds.Woo, RS1)
    sigS1 += einsum('ab,xbi->xai', imds.Wvv, RS1)
    sigS1 += einsum('xy,yai->xai', imds.Wp_p, RS1)
    sigS1 -= einsum('xji,aj->xai', imds.Wp_oo, RS)
    sigS1 += einsum('xab,bi->xai', imds.Wp_vv, RS)
    sigS1 += einsum('ajib,xbj->xai', imds.Wvoov, RS1)
    sigS1 -= einsum('xjb,abji->xai', imds.Wp_ov, RD)
    sigS1 += einsum('xyab,ybi->xai', imds.Wp_pvv, RS1)
    sigS1 -= einsum('xyji,yaj->xai', imds.Wp_poo, RS1)
    sigS1 += einsum('xyai,y->xai', imds.Wp_pvo, R1)


    # Wvovvvo
    Xoo = einsum('kldc,cdil->ki', I.oovv, RD)
    temp_ij += 0.5*einsum('abkj,ki->abij', T2, Xoo)
    Xoo = None

    #Woovovo
    Xvv = einsum('lkdc,bdkl->cb', I.oovv, RD)
    temp_ab += 0.5*einsum('ca,bcij->abij', Xvv, T2)
    Xvv = None

    #Wpvvoo
    X1oo = einsum('xki,x->ki', g.oo, R1)
    temp_ij += einsum('bakj,ki->abij', T2, X1oo)
    X1oo = None

    X1vv = einsum('xbc,x->bc', g.vv, R1)
    temp_ab += einsum('acij,bc->abij', T2, X1vv)
    X1vv = None

    X1ov = einsum('xkc,x->kc', g.ov, R1)
    Yoo = einsum('ci,kc->ki', T1, X1ov)
    Yvv = einsum('ak,kc->ac', T1, X1ov)
    temp_ij += einsum('abjk,ki->abij', T2, Yoo)
    temp_ab += einsum('bcij,ac->abij', T2, Yvv)
    X1ov = None

    XSoo = einsum('lkib,bk->li', I.ooov, RS)
    XSvv = einsum('bicd,ci->bd', I.vovv, RS)
    XSov = einsum('licd,ci->ld', I.oovv, RS)

    #Wovvvoo
    temp_ij += einsum('bakj,ki->abij', T2, XSoo)
    temp_ab += einsum('adji,bd->abij', T2, XSvv)

    XXoo = einsum('di,ld->li', T1, XSov)
    XXvv = einsum('bl,ld->bd', T1, XSov)
    temp_ij += einsum('ablj,li->abij', T2, XXoo)
    temp_ab += einsum('adij,bd->abij', T2, XXvv)

    #Wp_vovo
    sigS1 -= einsum('xak,ki->xai', U11, XSoo)
    sigS1 -= einsum('xci,ac->xai', U11, XSvv)
    Xmvv = einsum('xak,kc->xac', U11, XSov)
    Xmoo = einsum('xci,kc->xki', U11, XSov)
    sigS1 += einsum('ci,xac->xai', T1, Xmvv)
    sigS1 += einsum('ak,xki->xai', T1, Xmoo)
    Xmvv = None
    Xmoo = None

    XSvv = None
    XSoo = None
    XSov = None

    #Wpvvov
    XS1oo = einsum('xkc,xcj->kj', g.ov, RS1)
    temp_ij += einsum('abki,kj->abij', T2, XS1oo)

    #Wpvooo
    XS1vv = numpy.einsum('xkd,xbk->bd', g.ov, RS1)
    temp_ab += einsum('bdij,ad->abij', T2, XS1vv)
    XS1vv = None

    sigD += temp_ij - temp_ij.transpose((0,1,3,2))
    sigD += temp_ab - temp_ab.transpose((1,0,2,3))

    #Wp_ovvv
    XDoo = einsum('jkbc,bcij->ki', I.oovv, RD)
    sigS1 += 0.5*einsum('xak,ki->xai', U11, XDoo)

    #Wp_oovo
    XDvv = einsum('jkbd,abjk->da', I.oovv, RD)
    sigS1 += 0.5*einsum('xdi,da->xai', U11, XDvv)
    
    # added by YZ
    #sig1 += 1.0*einsum('I,I->I', w, R1)
    #sigS1 += 1.0*einsum('I,Iai->Iai', w, RS1)
    #sigS1 += 1.0*einsum('I,I,ai->Iai', w, S1, RS)
    # end of the copy

    np = w.shape[0]
    w2d = numpy.zeros((np,np))
    for k in range(np):
        w2d[k,k] = w[k]

    # here, we only need to add the difference (compared to nfock==1)
    # Note: the sign terms are wrong, more terms should appear, is re-generating the code.
    
    FUov = einsum('ia,Iai->I', F.ov, U11)
    gSov = einsum('Iia,ai->I', g.ov, RS)
    gS1ov = 1.0*einsum('Iia,Jai->IJ', g.ov, RS1)
    gTov = einsum('Iia,ai->I', g.ov, T1)
    gUov = einsum('Iia,Jai->IJ', g.ov, U11)
    IT1ov = 1.0*einsum('ijab,bi->ja', I.oovv, T1)
    ITUov = 1.0*einsum('ja,Iaj->I', IT1ov, U11)
    
    IUUov = einsum('ijab,Ibi,Jaj->IJ', I.oovv, U11, U11)

    if nfock1 > 1 and nfock2 == 1:
        # add nfock == 2 term
        # not fully optimized!!!
        sig2 = sign[1]

        sig1 += 1.0*einsum('J,IJ->I', gSov, S2)
        sig1 += 1.0*einsum('J,IJ->I', gTov, R2)
        
        sig2 += 1.0*einsum('IK,JK->IJ', w2d, R2)
        sig2 += 1.0*einsum('JK,IK->IJ', w2d, R2)
        sig2 += 1.0*einsum('IK,K,J->IJ', w2d, S1, R1)
        sig2 += 1.0*einsum('JK,K,I->IJ', w2d, S1, R1)
        
        sig2 += gS1ov
        sig2 += gS1ov.T

        sig2 += 1.0*einsum('I,J->IJ', FUov, R1)
        sig2 += 1.0*einsum('J,I->IJ', FUov, R1)
        sig2 += 1.0*einsum('I,J->IJ', gTov, R1)
        sig2 += 1.0*einsum('J,I->IJ', gTov, R1)

        sig2 += 1.0*einsum('KJ,IK->IJ', gS1ov, S2)
        sig2 += 1.0*einsum('KI,JK->IJ', gS1ov, S2)
        sig2 += 1.0*einsum('KI,JK->IJ', gUov, R2)
        sig2 += 1.0*einsum('KJ,IK->IJ', gUov, R2)

        sig2 += 1.0*einsum('K,IK,J->IJ', gTov, S2, R1)
        sig2 += 1.0*einsum('K,JK,I->IJ', gTov, S2, R1)
        sig2 += 1.0*einsum('KI,K,J->IJ', gUov, S1, R1)
        sig2 += 1.0*einsum('KJ,K,I->IJ', gUov, S1, R1)

        sig2 += -1.0*einsum('ijab,Ibi,Jaj->IJ', I.oovv, U11, RS1)
        sig2 += -1.0*einsum('ijab,Jbi,Iaj->IJ', I.oovv, U11, RS1)
        sig2 += -1.0*einsum('ijab,bi,Iaj,J->IJ', I.oovv, T1, U11, R1)
        sig2 += -1.0*einsum('ijab,bi,Jaj,I->IJ', I.oovv, T1, U11, R1)
        
        sigS1 +=  1.0*einsum('Jai,IJ->Iai', g.vo, R2)
        sigS1 += -1.0*einsum('Jji,IJ,aj->Iai', g.oo, S2, RS)
        sigS1 += -1.0*einsum('Jji,aj,IJ->Iai', g.oo, T1, R2)
        sigS1 +=  1.0*einsum('Jab,IJ,bi->Iai', g.vv, S2, RS)
        sigS1 +=  1.0*einsum('Jab,bi,IJ->Iai', g.vv, T1, R2)
        sigS1 += -1.0*einsum('Jjb,IJ,abji->Iai', g.ov, S2, RD)
        sigS1 += -1.0*einsum('Jjb,abji,IJ->Iai', g.ov, T2, R2)

        sigS1 += -1.0*einsum('Jjb,aj,IJ,bi->Iai', g.ov, T1, S2, RS)
        sigS1 += -1.0*einsum('Jjb,bi,IJ,aj->Iai', g.ov, T1, S2, RS)
        sigS1 += -1.0*einsum('Jjb,bi,aj,IJ->Iai', g.ov, T1, T1, R2)
        sigS1 +=  1.0*einsum('Jjb,bj,IJ,ai->Iai', g.ov, T1, S2, RS)

    if nfock1 > 2 and nfock2 == 1:
        # add nfock == 3 term
        sig3 = sign[2]

        sig2 += 0.33333333333333333*einsum('Kia,ai,IJK->IJ', g.ov, T1, R3)
        sig2 += 0.33333333333333333*einsum('Kia,ai,IKJ->IJ', g.ov, T1, R3)
        sig2 += 0.16666666666666666*einsum('Kia,ai,JKI->IJ', g.ov, T1, R3)
        sig2 += 0.16666666666666666*einsum('Kia,ai,KJI->IJ', g.ov, T1, R3)
        sig2 += 0.33333333333333333*einsum('Kia,IJK,ai->IJ', g.ov, S3, RS)
        sig2 += 0.33333333333333333*einsum('Kia,IKJ,ai->IJ', g.ov, S3, RS)
        sig2 += 0.16666666666666666*einsum('Kia,JKI,ai->IJ', g.ov, S3, RS)
        sig2 += 0.16666666666666666*einsum('Kia,KJI,ai->IJ', g.ov, S3, RS)
        
        sig3 += 0.33333333333333333*einsum('IL,JKL->IJK', w2d, R3)
        sig3 += 0.33333333333333333*einsum('IL,JLK->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('IL,KLJ->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('IL,LKJ->IJK', w2d, R3)
        sig3 += 0.33333333333333333*einsum('JL,IKL->IJK', w2d, R3)
        sig3 += 0.33333333333333333*einsum('JL,ILK->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('JL,KLI->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('JL,LKI->IJK', w2d, R3)
        sig3 += 0.33333333333333333*einsum('KL,IJL->IJK', w2d, R3)
        sig3 += 0.33333333333333333*einsum('KL,ILJ->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('KL,JLI->IJK', w2d, R3)
        sig3 += 0.16666666666666666*einsum('KL,LJI->IJK', w2d, R3)

        sig3 += 1.0*einsum('IL,L,JK->IJK', w2d, S1, R2)
        sig3 += 1.0*einsum('IL,JL,K->IJK', w2d, S2, R1)
        sig3 += 1.0*einsum('IL,KL,J->IJK', w2d, S2, R1)
        sig3 += 1.0*einsum('JL,L,IK->IJK', w2d, S1, R2)
        sig3 += 1.0*einsum('JL,IL,K->IJK', w2d, S2, R1)
        sig3 += 1.0*einsum('JL,KL,I->IJK', w2d, S2, R1)
        sig3 += 1.0*einsum('KL,L,IJ->IJK', w2d, S1, R2)
        sig3 += 1.0*einsum('KL,IL,J->IJK', w2d, S2, R1)
        sig3 += 1.0*einsum('KL,JL,I->IJK', w2d, S2, R1)

        sig3 += 1.0*einsum('I,JK->IJK', FUov, R2)
        sig3 += 1.0*einsum('J,IK->IJK', FUov, R2)
        sig3 += 1.0*einsum('K,IJ->IJK', FUov, R2)
        sig3 += 1.0*einsum('I,JK->IJK', gTov, R2)
        sig3 += 1.0*einsum('J,IK->IJK', gTov, R2)
        sig3 += 1.0*einsum('K,IJ->IJK', gTov, R2)
        sig3 += 1.0*einsum('IJ,K->IJK', gUov, R1)
        sig3 += 1.0*einsum('IK,J->IJK', gUov, R1)
        sig3 += 1.0*einsum('JI,K->IJK', gUov, R1)
        sig3 += 1.0*einsum('JK,I->IJK', gUov, R1)
        sig3 += 1.0*einsum('KI,J->IJK', gUov, R1)
        sig3 += 1.0*einsum('KJ,I->IJK', gUov, R1)
        
        sig3 += 0.33333333333333333*einsum('LK,IJL->IJK', gS1ov, S3)
        sig3 += 0.33333333333333333*einsum('LJ,IKL->IJK', gS1ov, S3)
        sig3 += 0.33333333333333333*einsum('LK,ILJ->IJK', gS1ov, S3)
        sig3 += 0.33333333333333333*einsum('LJ,ILK->IJK', gS1ov, S3)
        sig3 += 0.33333333333333333*einsum('LI,JKL->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LK,JLI->IJK', gS1ov, S3)
        sig3 += 0.33333333333333333*einsum('LI,JLK->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LJ,KLI->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LI,KLJ->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LK,LJI->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LJ,LKI->IJK', gS1ov, S3)
        sig3 += 0.16666666666666666*einsum('LI,LKJ->IJK', gS1ov, S3)

        sig3 += 0.33333333333333333*einsum('LI,JKL->IJK', gUov, R3)
        sig3 += 0.33333333333333333*einsum('LI,JLK->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LI,KLJ->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LI,LKJ->IJK', gUov, R3)
        sig3 += 0.33333333333333333*einsum('LJ,IKL->IJK', gUov, R3)
        sig3 += 0.33333333333333333*einsum('LJ,ILK->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LJ,KLI->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LJ,LKI->IJK', gUov, R3)
        sig3 += 0.33333333333333333*einsum('LK,IJL->IJK', gUov, R3)
        sig3 += 0.33333333333333333*einsum('LK,ILJ->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LK,JLI->IJK', gUov, R3)
        sig3 += 0.16666666666666666*einsum('LK,LJI->IJK', gUov, R3)
        
        #make gTS2ov (todo)
        sig3 += 1.0*einsum('L,IL,JK->IJK', gTov, S2, R2)
        sig3 += 1.0*einsum('L,JL,IK->IJK', gTov, S2, R2)
        sig3 += 1.0*einsum('L,KL,IJ->IJK', gTov, S2, R2)

        sig3 += 0.33333333333333333*einsum('L,IJL,K->IJK', gTov, S3, R1)
        sig3 += 0.33333333333333333*einsum('L,IKL,J->IJK', gTov, S3, R1)
        sig3 += 0.33333333333333333*einsum('L,ILJ,K->IJK', gTov, S3, R1)
        sig3 += 0.33333333333333333*einsum('L,ILK,J->IJK', gTov, S3, R1)
        sig3 += 0.33333333333333333*einsum('L,JKL,I->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,JLI,K->IJK', gTov, S3, R1)
        sig3 += 0.33333333333333333*einsum('L,JLK,I->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,KLI,J->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,KLJ,I->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,LJI,K->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,LKI,J->IJK', gTov, S3, R1)
        sig3 += 0.16666666666666666*einsum('L,LKJ,I->IJK', gTov, S3, R1)

        sig3 += 1.0*einsum('LI,L,JK->IJK', gUov, S1, R2)
        sig3 += 1.0*einsum('LJ,L,IK->IJK', gUov, S1, R2)
        sig3 += 1.0*einsum('LK,L,IJ->IJK', gUov, S1, R2)

        sig3 += 1.0*einsum('LI,JL,K->IJK', gUov, S2, R1)
        sig3 += 1.0*einsum('LI,KL,J->IJK', gUov, S2, R1)
        sig3 += 1.0*einsum('LJ,IL,K->IJK', gUov, S2, R1)
        sig3 += 1.0*einsum('LJ,KL,I->IJK', gUov, S2, R1)
        sig3 += 1.0*einsum('LK,IL,J->IJK', gUov, S2, R1)
        sig3 += 1.0*einsum('LK,JL,I->IJK', gUov, S2, R1)

        sig3 += -1.0*einsum('I,JK->IJK', ITUov, R2)
        sig3 += -1.0*einsum('J,IK->IJK', ITUov, R2)
        sig3 += -1.0*einsum('K,IJ->IJK', ITUov, R2)
        sig3 += -1.0*einsum('IJ,K->IJK', IUUov, R1)
        sig3 += -1.0*einsum('IK,J->IJK', IUUov, R1)
        sig3 += -1.0*einsum('JK,I->IJK', IUUov, R1)

    if nfock1 > 3 and nfock2 == 1:
        # add nfock == 4 term
        sig4 = sign[3]
        
        sig3 += 0.08333333333333333*einsum('L,IJKL->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,IJLK->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,IKJL->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,IKLJ->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,ILJK->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,ILKJ->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,JKIL->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,JLIK->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,KJIL->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,KLIJ->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,LJIK->IJK', gTov, R4)
        sig3 += 0.08333333333333333*einsum('L,LKIJ->IJK', gTov, R4)

        sig3 += 0.08333333333333333*einsum('L,IJKL->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,IJLK->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,IKJL->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,IKLJ->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,ILJK->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,ILKJ->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,JKIL->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,JLIK->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,KJIL->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,KLIJ->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,LJIK->IJK', gSov, S4)
        sig3 += 0.08333333333333333*einsum('L,LKIJ->IJK', gSov, S4)

        
        sig4 += 0.08333333333333333*einsum('IM,JKLM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,JKML->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,JLKM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,JLMK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,JMKL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,JMLK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,KLJM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,KMJL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,LKJM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,LMJK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,MKJL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('IM,MLJK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,IKLM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,IKML->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,ILKM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,ILMK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,IMKL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,IMLK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,KLIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,KMIL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,LKIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,LMIK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,MKIL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('JM,MLIK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,IJLM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,IJML->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,ILJM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,ILMJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,IMJL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,IMLJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,JLIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,JMIL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,LJIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,LMIJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,MJIL->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('KM,MLIJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IJKM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IJMK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IKJM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IKMJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IMJK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,IMKJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,JKIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,JMIK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,KJIM->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,KMIJ->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,MJIK->IJKL', w2d, R4)
        sig4 += 0.08333333333333333*einsum('LM,MKIJ->IJKL', w2d, R4)
        
        sig4 += 1.0*einsum('IM,JM,KL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('IM,KM,JL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('IM,LM,JK->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('JM,IM,KL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('JM,KM,IL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('JM,LM,IK->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('KM,IM,JL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('KM,JM,IL->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('KM,LM,IJ->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('LM,IM,JK->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('LM,JM,IK->IJKL', w2d, S2, R2)
        sig4 += 1.0*einsum('LM,KM,IJ->IJKL', w2d, S2, R2)
        
        sig4 += 0.33333333333333333*einsum('IM,M,JKL->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('IM,M,JLK->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('IM,M,KLJ->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('IM,M,LKJ->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('IM,JKM,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('IM,JLM,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('IM,JMK,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('IM,JML,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('IM,KLM,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,KMJ,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('IM,KML,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,LMJ,K->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,LMK,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,MKJ,L->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,MLJ,K->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('IM,MLK,J->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,M,IKL->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('JM,M,ILK->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('JM,M,KLI->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('JM,M,LKI->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('JM,IKM,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,ILM,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,IMK,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,IML,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,KLM,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,KMI,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('JM,KML,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,LMI,K->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,LMK,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,MKI,L->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,MLI,K->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('JM,MLK,I->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,M,IJL->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('KM,M,ILJ->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('KM,M,JLI->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('KM,M,LJI->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('KM,IJM,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,ILM,J->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,IMJ,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,IML,J->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,JLM,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,JMI,L->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('KM,JML,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,LMI,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,LMJ,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,MJI,L->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,MLI,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('KM,MLJ,I->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,M,IJK->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('LM,M,IKJ->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('LM,M,JKI->IJKL', w2d, S1, R3)
        sig4 += 0.16666666666666666*einsum('LM,M,KJI->IJKL', w2d, S1, R3)
        sig4 += 0.33333333333333333*einsum('LM,IJM,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,IKM,J->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,IMJ,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,IMK,J->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,JKM,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,JMI,K->IJKL', w2d, S3, R1)
        sig4 += 0.33333333333333333*einsum('LM,JMK,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,KMI,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,KMJ,I->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,MJI,K->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,MKI,J->IJKL', w2d, S3, R1)
        sig4 += 0.16666666666666666*einsum('LM,MKJ,I->IJKL', w2d, S3, R1)

        sig4 += 0.33333333333333333*einsum('I,JKL->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('I,JLK->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('I,KLJ->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('I,LKJ->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('J,IKL->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('J,ILK->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('J,KLI->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('J,LKI->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('K,IJL->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('K,ILJ->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('K,JLI->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('K,LJI->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('L,IJK->IJKL', FUov, R3)
        sig4 += 0.33333333333333333*einsum('L,IKJ->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('L,JKI->IJKL', FUov, R3)
        sig4 += 0.16666666666666666*einsum('L,KJI->IJKL', FUov, R3)
        
        sig4 += 0.33333333333333333*einsum('I,JKL->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('I,JLK->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('I,KLJ->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('I,LKJ->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('J,IKL->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('J,ILK->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('J,KLI->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('J,LKI->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('K,IJL->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('K,ILJ->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('K,JLI->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('K,LJI->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('L,IJK->IJKL', gTov, R3)
        sig4 += 0.33333333333333333*einsum('L,IKJ->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('L,JKI->IJKL', gTov, R3)
        sig4 += 0.16666666666666666*einsum('L,KJI->IJKL', gTov, R3)
        
        sig4 += 1.0*einsum('IJ,KL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('IK,JL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('IL,JK->IJKL', gUov, R2)
        sig4 += 1.0*einsum('JI,KL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('JK,IL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('JL,IK->IJKL', gUov, R2)
        sig4 += 1.0*einsum('KI,JL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('KJ,IL->IJKL', gUov, R2)
        sig4 += 1.0*einsum('KL,IJ->IJKL', gUov, R2)
        sig4 += 1.0*einsum('LI,JK->IJKL', gUov, R2)
        sig4 += 1.0*einsum('LJ,IK->IJKL', gUov, R2)
        sig4 += 1.0*einsum('LK,IJ->IJKL', gUov, R2)

        sig4 += 0.083333333333333333*einsum('MI,JKLM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,JKML->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,JLKM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,JLMK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,JMKL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,JMLK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,KLJM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,KMJL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,LKJM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,LMJK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,MKJL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MI,MLJK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,IKLM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,IKML->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,ILKM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,ILMK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,IMKL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,IMLK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,KLIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,KMIL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,LKIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,LMIK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,MKIL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MJ,MLIK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,IJLM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,IJML->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,ILJM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,ILMJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,IMJL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,IMLJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,JLIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,JMIL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,LJIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,LMIJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,MJIL->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('MK,MLIJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IJKM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IJMK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IKJM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IKMJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IMJK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,IMKJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,JKIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,JMIK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,KJIM->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,KMIJ->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,MJIK->IJKL', gUov, R4)
        sig4 += 0.083333333333333333*einsum('ML,MKIJ->IJKL', gUov, R4)
        
        sig4 += 0.083333333333333333*einsum('ML,IJKM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,IJLM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,IJMK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,IJML->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,IKJM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,IKLM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,IKMJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,IKML->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,ILJM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,ILKM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,ILMJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,ILMK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,IMJK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,IMJL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,IMKJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,IMKL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,IMLJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,IMLK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,JKIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JKLM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JKML->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,JLIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JLKM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JLMK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,JMIK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,JMIL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JMKL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,JMLK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,KJIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,KLIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,KLJM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,KMIJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,KMIL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,KMJL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,LJIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,LKIM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,LKJM->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,LMIJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,LMIK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,LMJK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,MJIK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,MJIL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('ML,MKIJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,MKIL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,MKJL->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MK,MLIJ->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MJ,MLIK->IJKL', gS1ov, S4)
        sig4 += 0.083333333333333333*einsum('MI,MLJK->IJKL', gS1ov, S4)

        sig4 += 0.33333333333333333*einsum('M,IM,JKL->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,IM,JLK->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,IM,KLJ->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,IM,LKJ->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,JM,IKL->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,JM,ILK->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,JM,KLI->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,JM,LKI->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,KM,IJL->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,KM,ILJ->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,KM,JLI->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,KM,LJI->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,LM,IJK->IJKL', gTov, S2, R3)
        sig4 += 0.33333333333333333*einsum('M,LM,IKJ->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,LM,JKI->IJKL', gTov, S2, R3)
        sig4 += 0.16666666666666666*einsum('M,LM,KJI->IJKL', gTov, S2, R3)

        sig4 += 0.33333333333333333*einsum('M,IJM,KL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,IKM,JL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,ILM,JK->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,IMJ,KL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,IMK,JL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,IML,JK->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,JKM,IL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,JLM,IK->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,JMI,KL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,JMK,IL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,JML,IK->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,KLM,IJ->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,KMI,JL->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,KMJ,IL->IJKL', gTov, S3, R2)
        sig4 += 0.33333333333333333*einsum('M,KML,IJ->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,LMI,JK->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,LMJ,IK->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,LMK,IJ->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MJI,KL->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MKI,JL->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MKJ,IL->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MLI,JK->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MLJ,IK->IJKL', gTov, S3, R2)
        sig4 += 0.16666666666666666*einsum('M,MLK,IJ->IJKL', gTov, S3, R2)
        
        sig4 += 0.08333333333333333*einsum('M,IJKM,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IJLM,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IJMK,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IJML,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IKJM,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IKLM,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IKMJ,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IKML,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,ILJM,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,ILKM,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,ILMJ,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,ILMK,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMJK,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMJL,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMKJ,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMKL,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMLJ,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,IMLK,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JKIM,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JKLM,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JKML,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JLIM,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JLKM,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JLMK,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JMIK,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JMIL,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JMKL,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,JMLK,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KJIM,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KLIM,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KLJM,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KMIJ,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KMIL,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,KMJL,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LJIM,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LKIM,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LKJM,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LMIJ,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LMIK,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,LMJK,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MJIK,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MJIL,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MKIJ,L->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MKIL,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MKJL,I->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MLIJ,K->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MLIK,J->IJKL', gTov, S4, R1)
        sig4 += 0.08333333333333333*einsum('M,MLJK,I->IJKL', gTov, S4, R1)
        
        sig4 += 0.33333333333333333*einsum('MI,M,JKL->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('MI,M,JLK->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MI,M,KLJ->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MI,M,LKJ->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('MJ,M,IKL->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('MJ,M,ILK->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MJ,M,KLI->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MJ,M,LKI->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('ML,M,IJK->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('ML,M,IKJ->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('ML,M,JKI->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('ML,M,KJI->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('MK,M,IJL->IJKL', gUov, S1, R3)
        sig4 += 0.33333333333333333*einsum('MK,M,ILJ->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MK,M,JLI->IJKL', gUov, S1, R3)
        sig4 += 0.16666666666666666*einsum('MK,M,LJI->IJKL', gUov, S1, R3)
         
        sig4 += 0.33333333333333333*einsum('MI,JKM,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MI,JLM,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MI,JMK,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MI,JML,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MI,KLM,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,KMJ,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MI,KML,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,LMJ,K->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,LMK,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,MKJ,L->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,MLJ,K->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MI,MLK,J->IJKL', gUov, S3, R1)
        
        
        sig4 += 1.0*einsum('MI,JM,KL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MI,KM,JL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MI,LM,JK->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MJ,IM,KL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MJ,KM,IL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MJ,LM,IK->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MK,IM,JL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MK,JM,IL->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('MK,LM,IJ->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('ML,IM,JK->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('ML,JM,IK->IJKL', gUov, S2, R2)
        sig4 += 1.0*einsum('ML,KM,IJ->IJKL', gUov, S2, R2)
        
        sig4 += 0.33333333333333333*einsum('MJ,IKM,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MJ,ILM,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MJ,IMK,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MJ,IML,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MJ,KLM,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,KMI,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MJ,KML,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,LMI,K->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,LMK,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,MKI,L->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,MLI,K->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MJ,MLK,I->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,IJM,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,ILM,J->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,IMJ,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,IML,J->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,JLM,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,JMI,L->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('MK,JML,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,LMI,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,LMJ,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,MJI,L->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,MLI,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('MK,MLJ,I->IJKL', gUov, S3, R1) 
        sig4 += 0.33333333333333333*einsum('ML,IJM,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('ML,IKM,J->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('ML,IMJ,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('ML,IMK,J->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('ML,JKM,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,JMI,K->IJKL', gUov, S3, R1)
        sig4 += 0.33333333333333333*einsum('ML,JMK,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,KMI,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,KMJ,I->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,MJI,K->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,MKI,J->IJKL', gUov, S3, R1)
        sig4 += 0.16666666666666666*einsum('ML,MKJ,I->IJKL', gUov, S3, R1)
        
        sig4 += -0.33333333333333333*einsum('I,JKL->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('I,JLK->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('I,KLJ->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('I,LKJ->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('J,IKL->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('J,ILK->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('J,KLI->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('J,LKI->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('K,IJL->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('K,ILJ->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('K,JLI->IJKL', ITUov, R3)
        sig4 += -0.16666666666666666*einsum('K,LJI->IJKL', ITUov, R3)
        sig4 += -0.33333333333333333*einsum('L,IJK->IJKL', ITUov, R3)

    if nfock2 > 1:
        # add nfock == 2 term
        ### restrict nfock1 >> nfock2 ?

        print('add sigS2 components (TBD)')
        # (2,1), (3,1) (4,1) are listed above
        # add (2,2) (3,2) (4,2)
        
        #a) nfock1 = 2, nfock2 = 2 --> diff22_21.log

        #b) nfock1 = 3, nfock2 = 2 --> diff32_22.log

        #c) nfock1 = 4, nfock2 = 2 --> diff43_22.log

    if nfock2 > 2:
        # add nfock == 3 terms
        print('nfock >2 is not supported!!!')

        # (3,3) term, since (3,2) is listed above
        # only diff33_32.log needed

    XS1oo = None

    return sigS, sigD, sign, sigSn


def eom_ee_epccsd_n_sn_sigma_opt_equal(nfock, RS, RD,
        Rn, RSn, amps, F, I, w, g, h, G, H, imds=None):


    if nfock > 1:

        print('add nfock1=nfock2=2 erms')

    if nfock > 2:

        print('add nfock1=nfock2=3 erms')

    if nfock > 3:
        # nfock = 4 terms: S4 and U14
        raise ValueError('nfock > 3 is not supported at this moment!!!')

    return sigS, sigD, sign, sigSn

