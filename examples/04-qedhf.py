import numpy
from pyscf import gto, scf, tdscf


zshift = 2.0
mol         = gto.Mole()
mol.verbose = 4
mol.atom  = f"C                  0.00000000     0.00000000    {zshift};\
     O                  0.00000000     1.23456800     {zshift};\
     H                  0.97075033    -0.54577032     {zshift};\
     C                 -1.21509881    -0.80991169     {zshift};\
     H                 -1.15288176    -1.89931439     {zshift};\
     C                 -2.43440063    -0.19144555     {zshift};\
     H                 -3.37262777    -0.75937214     {zshift};\
     O                 -2.62194056     1.12501165     {zshift};\
     H                 -1.71446384     1.51627790     {zshift}"

#mol.basis = 'cc-pVDZ'
mol.basis = 'STO-3G',
mol.build()

mf    = scf.HF(mol)
mf.kernel()

dm = mf.make_rdm1()

nmode = 1
cavity_freq = numpy.zeros(nmode)
cavity_mode = numpy.zeros((nmode, 3))
cavity_freq[0] = 3.0 /27.211386245988
cavity_mode[0,:] = 0.1 * numpy.asarray([1, 1, 1])
cavity_mode[0,:] = 0.1 * numpy.asarray([0, 0, 1])

#
breakline = "*" * 50
print("\n"+breakline)
print("    QED HF calculation ")
print(breakline+"\n")

# DFT/HFT-PF
import qed

qedmf = qed.HF(mol, xc=None, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
qedmf.max_cycle = 500


qedmf.kernel(dm0=dm)


