import numpy
from pyscf import gto
from pyscf import scf
from pyscf import tdscf

import qed

mol         = gto.Mole()
mol.verbose = 3
mol.atom    = '''
C         0.7033642071    0.0000000000    0.0000000000
C        -0.7033642071    0.0000000000    0.0000000000
H         1.2464078727    0.9456793582    0.0000000000
H         1.2464078727   -0.9456793582    0.0000000000
H        -1.2464078727    0.9456793582    0.0000000000
H        -1.2464078727   -0.9456793582    0.0000000000
'''
mol.basis = '6-311++g**'
mol.build()

mf         = scf.RKS(mol)
mf.verbose = 4
mf.xc      = "pbe0"
mf.kernel()

grad = mf.nuc_grad_method()
g    = grad.kernel()

td = tdscf.TDA(mf)
td.verbose = 4
td.nroots  = 5
td.kernel()
td.analyze()

print("td.e = \n", td.e)

grad = td.nuc_grad_method()
g    = grad.kernel(state=1) # 2 for S2 (0 for ground state)

cavity_freq = numpy.asarray( [0.24392215])
cavity_mode = numpy.asarray([[0.000, 0.000, 0.001]])

# TDA-JC
cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td        = qed.TDA(mf, cav_obj=cav_model)
td.nroots = 5
td.kernel()
print("td.e = \n", td.e)

grad = td.nuc_grad_method()
g    = grad.kernel(state=1) # 1 for S1 (0 for ground state)

assert 1 == 2

# TODO: rewrite `as_scanner` and `reset`
# Make sure the optimizer is not using hessian anyway
# Package dependencies are: 
# NetworkX:  https://github.com/networkx/networkx
# geomeTRIC: https://github.com/leeping/geomeTRIC
grad_scanner = grad.as_scanner(state=1)
mol1         = grad_scanner.optimizer().kernel()
