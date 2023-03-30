import numpy
from pyscf import gto, scf
from pyscf.eph.rhf import EPH

from cqcpy import utils
from cqcpy.ov_blocks import one_e_blocks
from cqcpy.ov_blocks import two_e_blocks
from cqcpy.integrals import eri_blocks

class HFSystem(object):
    def __init__(self, mf, const=1.0):
        myeph = EPH(mf)
        grad = mf.nuc_grad_method().kernel()
        eph = myeph.kernel(mo_rep=True)
        self.mf = mf
        self.omegas = eph[1]
        self.gmat = const*eph[0]
        self.nelec = self.mf.mol.nelectron
        self.occ = [i for i,x in enumerate(self.mf.mo_occ) if x > 0]
        self.vir = [i for i,x in enumerate(self.mf.mo_occ) if x == 0]
        self.nmo = self.mf.mo_occ.shape[0]
        self.nmode = len(self.omegas)
        self.zpe = 0.5*self.omegas.sum()

    def hf_energy(self):
        return self.mf.energy_tot() + self.zpe

    def omega(self):
        return self.omegas

    def energies(self):
        eo = self.mf.mo_energy[numpy.ix_(self.occ)]
        ev = self.mf.mo_energy[numpy.ix_(self.vir)]
        return numpy.concatenate((eo,eo)), numpy.concatenate((ev,ev))

    def g_fock(self):
        f = self.mf.get_fock()
        mo = self.mf.mo_coeff
        fmo = numpy.einsum('mn,mp,nq->pq', f, mo.conj(), mo)
        gocc = self.occ + [i + self.nmo for i in self.occ]
        gvir = self.vir + [i + self.nmo for i in self.vir]
        fmo = utils.block_diag(fmo, fmo)
        oo = fmo[numpy.ix_(gocc, gocc)]
        ov = fmo[numpy.ix_(gocc, gvir)]
        vo = fmo[numpy.ix_(gvir, gocc)]
        vv = fmo[numpy.ix_(gvir, gvir)]
        return one_e_blocks(oo, ov, vo, vv)

    def g_aint(self):
        I = eri_blocks(self.mf)
        return two_e_blocks(
            vvvv=I.vvvv, vvvo=I.vvvo,
            vovv=I.vovv, vvoo=I.vvoo,
            vovo=I.vovo, oovv=I.oovv,
            vooo=I.vooo, ooov=I.ooov,
            oooo=I.oooo)

    def gint(self):
        gocc = self.occ + [i + self.nmo for i in self.occ]
        gvir = self.vir + [i + self.nmo for i in self.vir]
        mall = numpy.arange(self.nmode)
        g_gmat = numpy.zeros((self.nmode, 2*self.nmo, 2*self.nmo), self.gmat.dtype)
        for i in range(self.nmode):
            g_gmat[i,:,:] = utils.block_diag(self.gmat[i,:,:], self.gmat[i,:,:])
        oo = g_gmat[numpy.ix_(mall, gocc, gocc)]
        ov = g_gmat[numpy.ix_(mall, gocc, gvir)]
        vo = g_gmat[numpy.ix_(mall, gvir, gocc)]
        vv = g_gmat[numpy.ix_(mall, gvir, gvir)]
        g = one_e_blocks(oo,ov,vo,vv)
        return (g,g)

    def mfG(self):
        g,h = self.gint()
        # these contributions are zero since we compute only the
        # normal-ordered part of the operator
        G = 0.0*numpy.einsum('Iii->I',g.oo)
        H = 0.0*numpy.einsum('Iii->I',h.oo)
        return (G,H)

from pyscf.pbc.eph.eph_fd import kernel
import pyscf.pbc.scf as pbc_scf
from pyscf.pbc import tools
class PBCHFSystem(object):
    def __init__(self, mf, const=1.0):
        import copy
        if mf.exxdiv is not None:
            mftemp = copy.copy(mf)
            mftemp.exxdiv = None
            mftemp.kernel(mf.make_rdm1())
        else: mftemp = mf
        eph = kernel(mftemp, mo_rep=True)
        self.mf = mf
        self.omegas = eph[1]
        self.gmat = const*eph[0]
        self.nelec = self.mf.mol.nelectron
        assert(len(self.mf.kpts) == 1)
        self.occ = [i for i,x in enumerate(self.mf.mo_occ[0]) if x > 0]
        self.vir = [i for i,x in enumerate(self.mf.mo_occ[0]) if x == 0]
        self.nmo = self.mf.mo_occ[0].shape[0]
        self.nmode = len(self.omegas)

    def hf_energy(self):
        return self.mf.energy_tot()

    def omega(self):
        return self.omegas
    
    def energies(self):
        eo = self.mf.mo_energy[0][numpy.ix_(self.occ)]
        ev = self.mf.mo_energy[0][numpy.ix_(self.vir)]
        return numpy.concatenate((eo,eo)), numpy.concatenate((ev,ev))

    def g_fock(self):
        f = self.mf.get_fock()[0]
        mo = self.mf.mo_coeff[0]
        fmo = numpy.einsum('mn,mp,nq->pq', f, mo.conj(), mo)
        # remove Madelung contribution from Fock matrix if necessary
        if self.mf.exxdiv is not None:
            madelung = tools.madelung(self.mf.cell, self.mf.kpt)
            for i in self.occ: fmo[i,i] += madelung
        gocc = self.occ + [i + self.nmo for i in self.occ]
        gvir = self.vir + [i + self.nmo for i in self.vir]
        fmo = utils.block_diag(fmo, fmo)
        oo = fmo[numpy.ix_(gocc, gocc)]
        ov = fmo[numpy.ix_(gocc, gvir)]
        vo = fmo[numpy.ix_(gvir, gocc)]
        vv = fmo[numpy.ix_(gvir, gvir)]
        return one_e_blocks(oo, ov, vo, vv)

    def g_aint(self):
        mf_temp = pbc_scf.RHF(self.mf.mol,kpt=self.mf.kpt)
        mf_temp.mo_energy = self.mf.mo_energy[0]
        mf_temp.mo_coeff = self.mf.mo_coeff[0]
        mf_temp.mo_occ = self.mf.mo_occ[0]
        I = eri_blocks(mf_temp)
        return two_e_blocks(
            vvvv=I.vvvv, vvvo=I.vvvo,
            vovv=I.vovv, vvoo=I.vvoo,
            vovo=I.vovo, oovv=I.oovv,
            vooo=I.vooo, ooov=I.ooov,
            oooo=I.oooo)

    def gint(self):
        gocc = self.occ + [i + self.nmo for i in self.occ]
        gvir = self.vir + [i + self.nmo for i in self.vir]
        mall = numpy.arange(self.nmode)
        g_gmat = numpy.zeros((self.nmode, 2*self.nmo, 2*self.nmo), self.gmat.dtype)
        for i in range(self.nmode):
            g_gmat[i,:,:] = utils.block_diag(self.gmat[i,:,:], self.gmat[i,:,:])
        oo = g_gmat[numpy.ix_(mall, gocc, gocc)]
        ov = g_gmat[numpy.ix_(mall, gocc, gvir)]
        vo = g_gmat[numpy.ix_(mall, gvir, gocc)]
        vv = g_gmat[numpy.ix_(mall, gvir, gvir)]
        g = one_e_blocks(oo,ov,vo,vv)
        return (g,g)

    def mfG(self):
        g,h = self.gint()
        # these contributions are zero since we compute only the
        # normal-ordered part of the operator
        #G = numpy.einsum('Iii->I',g.oo)
        #H = numpy.einsum('Iii->I',h.oo)
        G = numpy.zeros((self.nmode), dtype=self.mf.mo_coeff[0].dtype)
        H = numpy.zeros((self.nmode), dtype=self.mf.mo_coeff[0].dtype)
        return (G,H)
