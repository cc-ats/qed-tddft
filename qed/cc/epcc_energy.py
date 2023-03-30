import numpy
from pyscf import lib
from cqcpy import cc_energy

einsum = lib.einsum

def eph_energy(t1,s1,u11,g,G):
    Eeph = einsum('I,I->',G,s1)
    Eeph += einsum('Iia,Iai->',g,u11)
    Eeph += einsum('Iia,ai,I->',g,t1,s1)
    return Eeph

def energy(t1,t2,s1,u11,f,eri,w,g,G):
    Ecc = cc_energy.cc_energy(t1,t2,f,eri)
    Eeph = eph_energy(t1,s1,u11,g,G)
    return Ecc + Eeph



#### BRADEN WEIGHT ####

def QED_CCSD_S1_U11_energy(T1,T2,S1,U11,F,I,w,g,G):
    E_CCSD = 1.0*einsum('ia,ai->', F.ov, T1)
    E_CCSD += 1.0*einsum('Iia,Iai->', g.ov, U11)
    E_CCSD += 0.25*einsum('ijab,baji->', I.oovv, T2)
    E_CCSD += 1.0*einsum('Iia,ai,I->', g.ov, T1, S1)
    E_CCSD += -0.5*einsum('ijab,bi,aj->', I.oovv, T1, T1)

    return E_CCSD


def QED_CCSD_S1_S2_U11_energy(T1,T2,S1,S2,U11,F,I,w,g,G):
    E_CCSD = 1.0*einsum('ia,ai->', F.ov, T1)
    E_CCSD += 1.0*einsum('Iia,Iai->', g.ov, U11)
    E_CCSD += 0.25*einsum('ijab,baji->', I.oovv, T2)
    E_CCSD += 1.0*einsum('Iia,ai,I->', g.ov, T1, S1)
    E_CCSD += -0.5*einsum('ijab,bi,aj->', I.oovv, T1, T1)

    return E_CCSD


def QED_CCSD_S1_S2_S3_U11_energy(T1,T2,S1,S2,U11,F,I,w,g,G):
    E_CCSD = 1.0*einsum('ia,ai->', F.ov, T1)
    E_CCSD += 1.0*einsum('Iia,Iai->', g.ov, U11)
    E_CCSD += 0.25*einsum('ijab,baji->', I.oovv, T2)
    E_CCSD += 1.0*einsum('Iia,ai,I->', g.ov, T1, S1)
    E_CCSD += -0.5*einsum('ijab,bi,aj->', I.oovv, T1, T1)

    return E_CCSD

