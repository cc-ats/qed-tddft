import unittest
import sys
sys.path.append("../..")

import numpy
import scipy
from   functools import reduce

from pyscf import gto, scf, tdscf
import qed

mol         = gto.Mole()
mol.verbose = 7
mol.output  = '/tmp/test-qed-rpa.out'
mol.atom    = '''
O         0.0000000000    0.0000000000    0.1596165326
H        -0.7615930851    0.0000000000   -0.4389594363
H         0.7615930851    0.0000000000   -0.4389594363
'''
mol.basis = 'cc-pvdz'
mol.build()

mf    = scf.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()

mo_coeff  = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ    = mf.mo_occ

nao, nmo  = mo_coeff.shape
nocc      = (mo_occ>0).sum()
nvir      = nmo - nocc

rpa        = tdscf.RPA(mf)
rpa.nroots = nocc * nvir
rpa.kernel()
dip0 = rpa.transition_dipole()
e0   = rpa.e

cavity_freq = [e0[0], e0[2]]
cavity_mode = [0.04*dip0[0], 0.04*dip0[2]]
cavity_freq = numpy.asarray(cavity_freq)
cavity_mode = numpy.asarray(cavity_mode)

ncav        = cavity_freq.size
cavity_mode = cavity_mode.T

nov       = nvir * nocc
num_state = ncav + nvir * nocc

def get_eig_values(e):
    e1 = e.real
    e1 = e1[e1>0]
    e1 = numpy.sort(e1)
    return e1

def tearDownModule():
    global mol, mf, cavity_mode, cavity_freq
    mol.stdout.close()
    del mol, mf, cavity_mode, cavity_freq

class KnownValues(unittest.TestCase):
    def test_rpa_jc(self):
        cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        td        = qed.RPA(mf, cav_obj=cav_model)
        td.nroots = num_state
        td.kernel()

        a,b   = td.get_ab_block()
        g     = td.get_g_block()
        omega = td.get_omega_block()
        a = a.reshape(nov, nov)
        b = b.reshape(nov, nov)
        g = g.reshape(ncav, nov)
        zero_block = numpy.zeros_like(omega)

        x   = numpy.asarray([xy[0] for xy in td.xy])
        y   = numpy.asarray([xy[1] for xy in td.xy])
        m   = numpy.asarray([mn[0] for mn in td.mn])

        norm2  = 2*numpy.einsum("px,qx->pq", x, x)
        norm2 -= 2*numpy.einsum("px,qx->pq", y, y)
        norm2 +=   numpy.einsum("px,qx->pq", m, m)
        err    =   numpy.linalg.norm(norm2 - numpy.eye(num_state))
        self.assertAlmostEqual(err, 0, 6)

        e  = 2 * numpy.einsum("xy,px,qy->pq", a, x, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", a, y, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b, x, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b, y, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", g/numpy.sqrt(2), m, x+y) 
        e += 2 * numpy.einsum("xy,px,qy->qp", g/numpy.sqrt(2), m, x+y) 
        e +=     numpy.einsum("xy,px,qy->pq", omega, m, m)

        err = numpy.linalg.norm(e-numpy.diag(td.e))
        self.assertAlmostEqual(err, 0, 6)

        qed_mat  = numpy.block([[a, b, g.T], [-b, -a, -g.T], [g, g, omega]])
        e, w     = scipy.linalg.eig(qed_mat)
        e        = get_eig_values(e)
        err      = numpy.linalg.norm(e-td.e)
        self.assertAlmostEqual(err, 0, 6)

    def test_rpa_rabi(self):
        cav_model = qed.Rabi(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        td        = qed.RPA(mf, cav_obj=cav_model)
        td.nroots = num_state
        td.kernel()

        a,b   = td.get_ab_block()
        g     = td.get_g_block()
        omega = td.get_omega_block()
        a = a.reshape(nov, nov)
        b = b.reshape(nov, nov)
        g = g.reshape(ncav, nov)
        zero_block = numpy.zeros_like(omega)

        x   = numpy.asarray([xy[0] for xy in td.xy])
        y   = numpy.asarray([xy[1] for xy in td.xy])
        m   = numpy.asarray([mn[0] for mn in td.mn])
        n   = numpy.asarray([mn[1] for mn in td.mn])

        norm2  = 2*numpy.einsum("px,qx->pq", x, x)
        norm2 -= 2*numpy.einsum("px,qx->pq", y, y)
        norm2 +=   numpy.einsum("px,qx->pq", m, m)
        norm2 -=   numpy.einsum("px,qx->pq", n, n)
        err    =   numpy.linalg.norm(norm2 - numpy.eye(num_state))
        self.assertAlmostEqual(err, 0, 6)

        e  = 2 * numpy.einsum("xy,px,qy->pq", a, x, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", a, y, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b, x, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b, y, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", g/numpy.sqrt(2), m+n, x+y) 
        e += 2 * numpy.einsum("xy,px,qy->qp", g/numpy.sqrt(2), m+n, x+y) 
        e +=     numpy.einsum("xy,px,qy->pq", omega, m, m)
        e +=     numpy.einsum("xy,px,qy->pq", omega, n, n)

        err = numpy.linalg.norm(e-numpy.diag(td.e))
        self.assertAlmostEqual(err, 0, 6)

        qed_mat  = numpy.block([[a, b, g.T, g.T], [-b, -a, -g.T, -g.T], [g, g, omega, zero_block], [-g, -g, zero_block, -omega]])
        e, w     = scipy.linalg.eig(qed_mat)
        e        = get_eig_values(e)
        err      = numpy.linalg.norm(e-td.e)
        self.assertAlmostEqual(err, 0, 6)

    def test_rpa_rwa(self):
        cav_model = qed.RWA(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        td        = qed.RPA(mf, cav_obj=cav_model)
        td.nroots = num_state
        td.kernel()

        a,b   = td.get_ab_block()
        dse   = td.get_dse_block()
        g     = td.get_g_block()
        omega = td.get_omega_block()
        a = a.reshape(nov, nov)
        b = b.reshape(nov, nov)
        dse = dse.reshape(nov,  nov)
        g = g.reshape(ncav, nov)
        zero_block = numpy.zeros_like(omega)

        x   = numpy.asarray([xy[0] for xy in td.xy])
        y   = numpy.asarray([xy[1] for xy in td.xy])
        m   = numpy.asarray([mn[0] for mn in td.mn])

        norm2  = 2*numpy.einsum("px,qx->pq", x, x)
        norm2 -= 2*numpy.einsum("px,qx->pq", y, y)
        norm2 +=   numpy.einsum("px,qx->pq", m, m)
        err    =   numpy.linalg.norm(norm2 - numpy.eye(num_state))
        self.assertAlmostEqual(err, 0, 6)

        e  = 2 * numpy.einsum("xy,px,qy->pq", a+dse, x, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", a+dse, y, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b+dse, x, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b+dse, y, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", g/numpy.sqrt(2), m, x+y) 
        e += 2 * numpy.einsum("xy,px,qy->qp", g/numpy.sqrt(2), m, x+y) 
        e +=     numpy.einsum("xy,px,qy->pq", omega, m, m)

        err = numpy.linalg.norm(e-numpy.diag(td.e))
        self.assertAlmostEqual(err, 0, 6)

        qed_mat  = numpy.block([[a+dse, b+dse, g.T], [-b-dse, -a-dse, -g.T], [g, g, omega]])
        e, w     = scipy.linalg.eig(qed_mat)
        e        = get_eig_values(e)
        err      = numpy.linalg.norm(e-td.e)
        self.assertAlmostEqual(err, 0, 6)

    def test_rpa_pf(self):
        cav_model = qed.PF(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        td        = qed.RPA(mf, cav_obj=cav_model)
        td.nroots = num_state
        td.kernel()

        a,b   = td.get_ab_block()
        dse   = td.get_dse_block()
        g     = td.get_g_block()
        omega = td.get_omega_block()
        a = a.reshape(nov, nov)
        b = b.reshape(nov, nov)
        dse = dse.reshape(nov,  nov)
        g = g.reshape(ncav, nov)
        zero_block = numpy.zeros_like(omega)

        x   = numpy.asarray([xy[0] for xy in td.xy])
        y   = numpy.asarray([xy[1] for xy in td.xy])
        m   = numpy.asarray([mn[0] for mn in td.mn])
        n   = numpy.asarray([mn[1] for mn in td.mn])

        norm2  = 2*numpy.einsum("px,qx->pq", x, x)
        norm2 -= 2*numpy.einsum("px,qx->pq", y, y)
        norm2 +=   numpy.einsum("px,qx->pq", m, m)
        norm2 -=   numpy.einsum("px,qx->pq", n, n)
        err    =   numpy.linalg.norm(norm2 - numpy.eye(num_state))
        self.assertAlmostEqual(err, 0, 6)

        e  = 2 * numpy.einsum("xy,px,qy->pq", a+dse, x, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", a+dse, y, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b+dse, x, y) 
        e += 2 * numpy.einsum("xy,px,qy->pq", b+dse, y, x) 
        e += 2 * numpy.einsum("xy,px,qy->pq", g/numpy.sqrt(2), m+n, x+y) 
        e += 2 * numpy.einsum("xy,px,qy->qp", g/numpy.sqrt(2), m+n, x+y) 
        e +=     numpy.einsum("xy,px,qy->pq", omega, m, m)
        e +=     numpy.einsum("xy,px,qy->pq", omega, n, n)

        err = numpy.linalg.norm(e-numpy.diag(td.e))
        self.assertAlmostEqual(err, 0, 6)

        qed_mat  = numpy.block([[a+dse, b+dse, g.T, g.T], [-b-dse, -a-dse, -g.T, -g.T], [g, g, omega, zero_block], [-g, -g, zero_block, -omega]])
        e, w     = scipy.linalg.eig(qed_mat)
        e        = get_eig_values(e)
        err      = numpy.linalg.norm(e-td.e)
        self.assertAlmostEqual(err, 0, 6)

if __name__ == "__main__":
    print("Full Tests for QED-TDA Methods.")
    unittest.main()
