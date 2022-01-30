from   functools import reduce
import numpy

from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import _response_functions  # noqa
from pyscf.data import nist
from pyscf import __config__

from pyscf.tdscf.rhf import get_ab
from pyscf.tdscf.rhf import analyze, get_nto, oscillator_strength
from pyscf.tdscf.rhf import _contract_multipole
from pyscf.tdscf.rhf import transition_dipole              
from pyscf.tdscf.rhf import transition_quadrupole          
from pyscf.tdscf.rhf import transition_octupole            
from pyscf.tdscf.rhf import transition_velocity_dipole     
from pyscf.tdscf.rhf import transition_velocity_quadrupole 
from pyscf.tdscf.rhf import transition_velocity_octupole   
from pyscf.tdscf.rhf import transition_magnetic_dipole 
from pyscf.tdscf.rhf import transition_magnetic_quadrupole

class UnrestrictedCavityModel(CavityModel):
    def build(self):
        if self.cavity_num > 0:
            mol      = self._mol
            dip_ao   = mol.intor("int1e_r", comp=3)

            mf       = self._scf
            mo_coeff = mf.mo_coeff
            assert(mo_coeff[0].dtype == numpy.double)
            mo_energy = mf.mo_energy
            mo_occ    = mf.mo_occ
            nao, nmo  = mo_coeff[0].shape
            occidxa   = numpy.where(mo_occ[0]>0)[0]
            occidxb   = numpy.where(mo_occ[1]>0)[0]
            viridxa   = numpy.where(mo_occ[0]==0)[0]
            viridxb   = numpy.where(mo_occ[1]==0)[0]
            nocca     = len(occidxa)
            noccb     = len(occidxb)
            nvira     = len(viridxa)
            nvirb     = len(viridxb)
            orboa     = mo_coeff[0][:,occidxa]
            orbob     = mo_coeff[1][:,occidxb]
            orbva     = mo_coeff[0][:,viridxa]
            orbvb     = mo_coeff[1][:,viridxb]
            assert(mo_coeff.dtype == numpy.double)

            dip_ov_a = lib.einsum('xmn,mi,na->xia', dip_ao, orboa.conj(), orbva)
            dip_ov_b = lib.einsum('xmn,mi,na->xia', dip_ao, orbob.conj(), orbvb)
            self.dip_ov = (dip_ov_a.reshape(3, nocca, nvira), dip_ov_b.reshape(3, noccb, nvirb))
        else:
            self.dip_ov = None
