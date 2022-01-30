import numpy
from pyscf import gto, scf, tdscf

import qed

mol         = gto.Mole()
mol.verbose = 3
mol.atom    = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''
mol.basis = 'cc-pVDZ'
mol.build()

mf    = scf.RKS(mol)
mf.xc = "b3lyp"
mf.kernel()

cavity_freq = numpy.asarray([0.2940])
cavity_mode = numpy.asarray([[0.001, 0.0, 0.0]])

# TDDFT-PF
cav_model = qed.PF(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td        = qed.TDDFT(mf, cav_obj=cav_model)
td.nroots = 5
td.kernel()

# TDA-JC
cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td        = qed.TDA(mf, cav_obj=cav_model)
td.nroots = 5
td.kernel()