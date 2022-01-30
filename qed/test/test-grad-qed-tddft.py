import unittest
import sys
sys.path.append("../..")

import numpy
import scipy

from pyscf import gto, scf, tdscf
import qed

mol = gto.Mole()
mol.verbose = 7
mol.output = '/tmp/test-grad-qed-tddft.out'
mol.atom = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''
mol.basis = 'cc-pvdz'
mol.build()

mf    = scf.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()

tdhf        = tdscf.RPA(mf)
tdhf.nroots = 10
tdhf.kernel()
dip0 = tdhf.transition_dipole()
e0   = tdhf.e

cavity_mode = [dip0[2]*0.02]
cavity_freq = [e0[2]]

cavity_mode = numpy.asarray(cavity_mode)
cavity_freq = numpy.asarray(cavity_freq)

num_cav     = cavity_freq.size
cavity_mode = cavity_mode.reshape(3, num_cav)

def tearDownModule():
    global mol, mf, cavity_mode, cavity_freq
    mol.stdout.close()
    del mol, mf, cavity_mode, cavity_freq

class KnownValues(unittest.TestCase):
    def test_cis_jc_grad(self):
        jc_model  = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        tdhf_jc   = qed.RPA(mf, cav_obj=jc_model)
        tdhf_jc.nroots = 10
        tdhf_jc.kernel()

        grad = qed.Gradients(tdhf_jc)
        g1 = grad.kernel(state=1)
        g2 = grad.kernel(state=2)
        g3 = grad.kernel(state=3)
        g4 = grad.kernel(state=4)

    def test_cis_rabi_grad(self):
        rabi_model = qed.Rabi(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        tdhf_rabi   = qed.RPA(mf, cav_obj=rabi_model)
        tdhf_rabi.nroots = 10
        tdhf_rabi.kernel()

        grad = qed.Gradients(tdhf_rabi)
        g1 = grad.kernel(state=1)
        g2 = grad.kernel(state=2)
        g3 = grad.kernel(state=3)
        g4 = grad.kernel(state=4)

    def test_cis_rwa_grad(self):
        rwa_model = qed.RWA(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        tdhf_rwa  = qed.RPA(mf, cav_obj=rwa_model)
        tdhf_rwa.nroots = 10
        tdhf_rwa.kernel()

        grad = qed.Gradients(tdhf_rwa)
        g1 = grad.kernel(state=1)
        g2 = grad.kernel(state=2)
        g3 = grad.kernel(state=3)
        g4 = grad.kernel(state=4)

    def test_cis_pf_grad(self):
        pf_model = qed.PF(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        cis_pf   = qed.TDA(mf, cav_obj=pf_model)
        cis_pf.nroots = 10
        cis_pf.kernel()

        grad = qed.Gradients(cis_pf)
        g1 = grad.kernel(state=1)
        g2 = grad.kernel(state=2)
        g3 = grad.kernel(state=3)
        g4 = grad.kernel(state=4)


if __name__ == "__main__":
    print("Full Tests for QED-TDRHF Methods.")
    unittest.main()
