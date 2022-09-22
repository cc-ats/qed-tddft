import numpy
from pyscf import gto, scf, tdscf

import sys
sys.path.append('..')
import qed
from qed.tdscf.pyscf_parser import *

if __name__ == '__main__':
    field_type = ['homogeneous', 'heterogeneous']
    for i in range(2):
        parameters = parser('h2o_'+field_type[i]+'.in')
        mol, mf, td, qed_td, cav_obj, qed_model, cavity_model = \
                    run_pyscf_final(parameters)

    # we can also change the parameters here
    field_type = ['homogeneous', 'heterogeneous']
    for i in range(2):
        parameters = parser('h2o_'+field_type[i]+'.in')
        parameters['polariton']['cavity_model'] = 'pf'
        mol, mf, td, qed_td, cav_obj, qed_model, cavity_model = \
                    run_pyscf_final(parameters)
