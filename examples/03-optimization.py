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

cavity_freq = numpy.asarray([0.25684])
cavity_mode = numpy.asarray([[0.000, 0.000, 0.001]])

# TDA-JC
cav_model = qed.JC(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td        = qed.TDA(mf, cav_obj=cav_model)
td.nroots = 5
td.kernel()

grad = td.nuc_grad_method()
g    = grad.kernel(state=1) # 1 for S1 (0 for ground state)

grad_scanner = grad.as_scanner(state=1)
mol1         = grad_scanner.optimizer().kernel()
coords = mol1.atom_coords(unit='Angstrom')
r1     = numpy.linalg.norm(coords[0, :] - coords[1, :])

grad_scanner = grad.as_scanner(state=2)
mol2         = grad_scanner.optimizer().kernel()
coords = mol2.atom_coords(unit='Angstrom')
r2     = numpy.linalg.norm(coords[0, :] - coords[1, :])

print(f"r1 = {r1:6.4f}, r2 = {r2:6.4f}")
