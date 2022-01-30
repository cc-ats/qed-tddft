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
mol.output  = '/tmp/test-grad-qed-tdhf.out'
mol.atom    = '''
O         0.0000000000    0.0000000000    0.1596165326
H        -0.7615930851    0.0000000000   -0.4389594363
H         0.7615930851    0.0000000000   -0.4389594363
'''
mol.basis = 'cc-pVDZ'
mol.build()

mf    = scf.RHF(mol)
mf.kernel()

mo_coeff  = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ    = mf.mo_occ

nao, nmo  = mo_coeff.shape
nocc      = (mo_occ>0).sum()
nvir      = nmo - nocc

tda        = tdscf.TDA(mf)
tda.nroots = nocc * nvir
tda.kernel()
dip0 = tda.transition_dipole()
e0   = tda.e

cavity_freq = [e0[0], e0[2]]
cavity_mode = [0.04*dip0[0], 0.04*dip0[2]]
cavity_freq = numpy.asarray(cavity_freq)
cavity_mode = numpy.asarray(cavity_mode)

ncav        = cavity_freq.size
cavity_mode = cavity_mode.T

nov       = nvir * nocc
num_state = 10

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
    def test_cis_jc(self):
        cav_model = qed.PF(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
        td        = qed.RPA(mf, cav_obj=cav_model)
        td.nroots = num_state
        td.kernel()

        grad = qed.Gradients(td)
        g1 = grad.kernel(state=1)
        g2 = grad.kernel(state=2)
        g3 = grad.kernel(state=3)
        g4 = grad.kernel(state=4)

if __name__ == "__main__":
    print("Full Tests for QED-TDA Methods.")
    unittest.main()
