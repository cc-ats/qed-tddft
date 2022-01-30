# Quantum-electrodynamical Time-dependent Density Functional Theory Within Gaussian Atomic Basis

## Installation
The PySCF program package can be installed with
```
pip install pyscf
```

To use QED-TDDFT,
```
git clone git@github.com:cc-ats/qed-tddft.git
export PYTHONPATH=$(pwd):$PYTHONPATH
cd examples
python 01-qed-tddft.py 
```

## Example
```
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

cavity_freq = numpy.asarray([0.200])
cavity_mode = numpy.asarray([[0.001, 0.0, 0.0]])

# TDDFT-PF
cav_model = qed.PF(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
td        = qed.TDDFT(mf, cav_obj=cav_model)
td.nroots = 5
td.kernel()
```

## Please Cite

[Quantum-electrodynamical time-dependent density functional theory within Gaussian atomic basis](https://aip.scitation.org/doi/full/10.1063/5.0057542),
Junjie Yang, Qi Ou, Zheng Pei, Hua Wang, Binbin Weng, Zhigang Shuai, Kieran Mullen, and Yihan Shao, *J. Chem. Phys.*. **155**, 064107 (2021). doi:[10.1063/5.0057542](https://aip.scitation.org/doi/full/10.1063/5.0057542)

The following paper should also be cited in publications utilizing the PySCF program package:

[PySCF: the Python‐based simulations of chemistry framework](https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340),
Q. Sun, T. C. Berkelbach, N. S. Blunt, G. H. Booth, S. Guo, Z. Li, J. Liu,
J. McClain, E. R. Sayfutyarova, S. Sharma, S. Wouters, G. K.-L. Chan (2018),
*WIREs Comput. Mol. Sci.*, **8**: e1340. doi:[10.1002/wcms.1340](https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340)

[Recent developments in the PySCF program package](https://aip.scitation.org/doi/10.1063/5.0006074),
Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan,
*J. Chem. Phys.*, **153**, 024109 (2020). doi:[10.1063/5.0006074](https://aip.scitation.org/doi/10.1063/5.0006074)