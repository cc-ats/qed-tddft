import os
import sys
import numpy
from pyscf import gto, scf, cc
from time import time
from qed.cc.abinit import Model

au2ev = 27.211386245988
NRoots = 6
nfock_max = 3

# use os.listdir() to get the list of files and folders in the specified directory
fd = "geo_trans"
filelist = os.listdir(fd)

# to get only files, you can filter the list using os.path.isfile()
filelist = [f for f in filelist if os.path.isfile(os.path.join(fd, f)) and ".xyz" in f]
#filelist.sort()
print(filelist)

def get_dangle(fname):
    start_pos = fname.find('D')
    end_pos = fname.find(".xyz") 
    return float(fname[start_pos+1:end_pos])

#sort file according to dihedral angle
Ngeo = len(filelist)
Dangle = numpy.zeros(Ngeo)
for i, fname in enumerate(filelist):
    Dangle[i] = get_dangle(fname)

idx = numpy.argsort(Dangle)
filelist = [filelist[i] for i in idx]

'''
#boson operator
 two_p is: \sum_{01}w_{01}[b^{\dag}_0(nm)b_1(nm)]
 one_p is: \sum_{0} G_{0} [b^{\dag}_0(nm) + b_0(nm)]
  Fock matrix and matrix of normal modes are not assumed to be diagonal,
 even though they will usually be diagonal
 i./e, w_{01} = 0, w_{00} is the vibrational mode (freq)

 H = f_{pq} a^\dag_p a_q  : one electron part
   + I_{pqsr} a^\dag_p a^\dag_q a_s a_r : two electron part
   + g_{pq m} a^\dag_p a_q (b^\dag_m + b_m)
   + \sum_{01}w_{01}[b^{\dag}_0(nm)b_1(nm)]
'''

gen_integral = False

for fname in filelist:
    print("\n-------------------------------------------------")
    print("------------- %s -------------" %fname)
    print("-------------------------------------------------\n")
    Dangle[i] = get_dangle(fname)
    print('Dihedral angle is', Dangle[i])

    lines = open(os.path.join(fd,fname), 'r').readlines()
    atoms = ""
    for line in lines:
        if len(line.split()) == 4:
            atoms += line.strip() + "; "
    print('atoms in %s is:\n' %fname, atoms, '\n')

    mol = gto.M(
        atom = atoms,
        basis = 'sto3g',
        #basis = '631g(d)',
        unit = 'Angstrom',
        symmetry = True,
    )

    hf = scf.HF(mol)
    hf.max_cycle = 200
    hf.conv_tol = 1.e-8
    hf.diis_space = 10
    #hf.polariton = True
    #hf.gcoup = 0.01
    mf = hf.run()

    print('hf energy', Dangle[i],  mf.e_tot)

    '''
    print('\n=========== ccsd calculation  ======================\n')
    # Note that the line following these comments could be replaced by
    # mycc = cc.CCSD(mf)
    # mycc.kernel()
    mycc = cc.CCSD(mf).run()

    gstot = '%6.3f  '%Dangle[i]   + '%20.9f  ' % mycc.e_tot
    print('CCSD total energy', gstot)
    print('CCSD correlaiton energy', mycc.e_tot - mf.e_tot)
    '''

    method = 'rhf'
    mol.build()
    xc = 'b3lyp'
    nmode = 1
    omega = numpy.zeros(nmode)
    vec = numpy.zeros((nmode,3))
    omega[0] = 3.0/au2ev
    gfac = 1.e-2 # start with zero, test different strengths
    vec[0,:] = [0.0, 0.0, 1.0]

    from qed.cc.eom_epcc import eom_ee_epccsd_1_s1
    from qed.cc.eom_epcc import eom_ee_epccsd_n_sn
    from qed.cc.polaritoncc import epcc, epcc_nfock
    from qed.cc.abinit import Model

    mod = Model(mol, method, xc, omega, vec, gfac, shift=False)

    if gen_integral:
        #--------------------------
        # get Hamiltonian:
        #--------------------------
        # fock matrix
        F = mod.g_fock()
        eo = F.oo.diagonal()
        ev = F.vv.diagonal()

        D1 = eo[:,None] - ev[None,:]
        D2 = eo[:,None,None,None] + eo[None,:,None,None]\
                - ev[None,None,:,None] - ev[None,None,None,:]

        F.oo = F.oo - numpy.diag(eo)
        F.vv = F.vv - numpy.diag(ev)

        # get HF energy
        Ehf = mod.hf_energy()

        # get ERIs
        I = mod.g_aint()

        # get normal mode energies
        w = mod.omega()
        np = w.shape[0]
        D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]

        # get elec-phon matrix elements
        g,h = mod.gint()
        G,H = mod.mfG()

        del g, h, G, H, I, F

    else:
        # conventional CCSD calculation
        hf = scf.HF(mol)
        mf = hf.run()
        print('hf energy',mf.hf_energy)
        mycc = cc.CCSD(mf).run()
        print('CCSD total energy', mycc.e_tot)
        mycc.kernel()
        pyscf_es, c_ee = mycc.eeccsd(nroots=NRoots)
        e_singlets, c_singlets = mycc.eomee_ccsd_singlet(nroots=NRoots)

        print('pyscf excited state energies =', pyscf_es)
        print('\npyscf excited state energies (singlets) =', e_singlets)
        print('\n=========== end of eom ccsd calculation  ================\n')

        for nfock in range(1, nfock_max+1):
            print(f'\n===========  qed-ccsd calculation  nfock={nfock}====================\n')
            #options = {"ethresh":1e-7, 'tthresh':1e-6, 'max_iter':500, 'damp':0.4, 'nfock': nfock}
            #ecc, e_cor, (T1,T2,S1,U11) = epcc(mod, options,ret=True)
            options = {"ethresh":1e-7, 'tthresh':1e-7, 'max_iter':500, 'damp':0.4, 'nfock': nfock, 'nfock2':1, 'slow': False}
            ecc, e_cor, (T1,T2,Sn,U1n) = epcc_nfock(mod, options,ret=True)
            print(f'correlation energy {gfac:.4f} (epcc): {nfock} {ecc:.9f}  {e_cor:.9f}')

            # exciton-polariton
            options = {"nroots":NRoots, 'conv_tol':1e-7, 'max_cycle':500, 'max_space':1000, 'nfock': nfock, 'nfock2':1, 'slow': False}
            es = eom_ee_epccsd_n_sn(mod, options, amps=(T1,T2,Sn,U1n), analysis=True)
            print(f'excited state energies {nfock}  {", ".join(["%.9f" % value*au2ev for value in es[0]])}')

