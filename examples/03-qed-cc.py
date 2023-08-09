import sys
import numpy
from pyscf import gto, scf, cc
from time import time


NRoots = 10
nfock_max = 1
gfac = 0.005 #5

Rlist = numpy.arange(3.2, 3.21, 0.04)

for R in Rlist:

    print('\n==== calculating R=%f ==========\n' % R)

    mol = gto.M(
        atom = 'Li 0 0 0; H 0 0 ' + str(R),  # in au
        basis = '631g',
        unit = 'Bohr',
        symmetry = True,
    )
    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.e-8
    hf.diis_space = 10
    #hf.polariton = True
    #hf.gcoup = 0.01
    mf = hf.run()

    print('hf energy',mf.hf_energy)
    # Note that the line following these comments could be replaced by
    # mycc = cc.CCSD(mf)
    # mycc.kernel()
    mycc = cc.CCSD(mf).run()
    print('CCSD total energy', mycc.e_tot)

    gstot = '%6.3f  '%R   + '%20.9f  ' % mycc.e_tot

    print('\n=========== eom ccsd calculation  ======================\n')
    mycc.kernel()
    es, c_ee = mycc.eeccsd(nroots=NRoots)
    e_singlets, c_singlets = mycc.eomee_ccsd_singlet(nroots=NRoots)
    e_triplets, c_triplets = mycc.eomee_ccsd_triplet(nroots=NRoots)

    print('\npyscf excited state energies (all)      =\n', es)
    print('\npyscf excited state energies (singlets) =\n', e_singlets)
    print('\npyscf excited state energies (triplets) =\n', e_triplets)
    print('\n=========== end of eom ccsd calculation  ================\n')

    print('\n=========== eom pccsd calculation  ======================\n')
    method = 'uhf'
    mol.build()
    xc = 'b3lyp'
    nmode = 1
    omega = numpy.zeros(nmode)
    vec = numpy.zeros((nmode,3))
    omega[0] = 0.1 #2.7/27.211
    vec[0,:] = [1.0, 1.0, 1.0]
    print('\n')

    from qed.cc.eom_epcc import eom_ee_epccsd_1_s1
    from qed.cc.eom_epcc import eom_ee_epccsd_n_sn
    from qed.cc.polaritoncc import epcc, epcc_nfock
    from qed.cc.abinit import Model

    mod = Model(mol, method, xc, omega, vec, gfac, shift=False)
    print('omega=', mod.omega())

    options = {"ethresh":1e-7, 'tthresh':1e-6, 'max_iter':500, 'damp':0.4, 'nfock': 1}
    ecc, e_cor, (T1,T2,S1,U11) = epcc(mod, options,ret=True)
    print('correlation energy (epcc)=', ecc, e_cor)
    print('converged S1=', S1)

    T1save = T1.copy()
    T2save = T2.copy()
    S0save = S1.copy()
    U1save = U11.copy()

    print('\n========== eom_ee_epccsd_1_s1 calculaiton ======')
    options = {"nroots":NRoots, 'conv_tol':1e-7, 'max_cycle':500, 'max_space':1000}
    es = eom_ee_epccsd_1_s1(mod, options, amps=(T1,T2,S1,U11), analysis=True)

    print('excited state energies =', es)

    k = 1
    useslow = False
    options = {"ethresh":1e-7, 'tthresh':1e-7, 'max_iter':500, 'damp':0.4, 'nfock': k, 'nfock2':1, 'slow': useslow}
    ecc, e_cor, (T1,T2,Sn,U1n) = epcc_nfock(mod, options,ret=True)

    print('\n  ----excited states---- \n')
    options = {"nroots":NRoots, 'conv_tol':1e-7, 'max_cycle':500, 'max_space':1000, 'nfock1': k, 'nfock2':1, 'slow': useslow}
    es = eom_ee_epccsd_n_sn(mod, options, amps=(T1,T2,Sn,U1n), analysis=True)
    print('excited state energies', es)
