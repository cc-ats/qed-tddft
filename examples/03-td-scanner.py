import numpy
from pyscf import gto
from pyscf import scf
from pyscf import tdscf

import qed

xyz1 = '''
  C         0.6250000031    0.0000000000    0.0000000000
  C        -0.6250000031    0.0000000000    0.0000000000
  H         1.2100322321    0.9170150707    0.0000000000
  H         1.2100322321   -0.9170150707    0.0000000000
  H        -1.2100322321    0.9170150707    0.0000000000
  H        -1.2100322321   -0.9170150707    0.0000000000
'''

xyz2 = '''
  C         0.6304999999    0.0000000000    0.0000000000
  C        -0.6304999999    0.0000000000    0.0000000000
  H         1.2133016825    0.9181170425    0.0000000000
  H         1.2133016825   -0.9181170425    0.0000000000
  H        -1.2133016825    0.9181170425    0.0000000000
  H        -1.2133016825   -0.9181170425    0.0000000000
'''

mol         = gto.Mole()
mol.verbose = 0
mol.atom    = xyz1
mol.basis = '6-311++g**'
mol.build()

mf         = scf.RKS(mol)
mf.verbose = 0
mf.xc      = "pbe0"
mf.kernel()

# TDA-JC
cavity_freq = numpy.asarray([0.25684])
cavity_mode = numpy.asarray([[0.000, 0.000, 0.01]])
cav_model   = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td          = qed.TDA(mf, cav_obj=cav_model)
td.nroots   = 5
td.kernel()

grad = td.nuc_grad_method()
grad.verbose = 3
g    = grad.kernel(state=1)
grad_scanner = grad.as_scanner(state=1)
e1, g1 = grad_scanner(xyz2)

mol         = gto.Mole()
mol.verbose = 0
mol.atom    = xyz2
mol.basis   = '6-311++g**'
mol.build()

mf         = scf.RKS(mol)
mf.verbose = 0
mf.xc      = "pbe0"
mf.kernel()

# TDA-JC
cavity_freq = numpy.asarray([0.25684])
cavity_mode = numpy.asarray([[0.000, 0.000, 0.01]])
cav_model   = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td          = qed.TDA(mf, cav_obj=cav_model)
td.nroots   = 5
td.kernel()

grad = td.nuc_grad_method()
grad.verbose = 3
g2   = grad.kernel(state=1)

assert 1 == 2