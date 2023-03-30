import numpy
import sys
from cqcpy import cc_equations
from . import epcc_energy
from . import epcc_equations
from . import epcc_equations_gen

def guage(model, option='dipole'):
    """
    1) dipole 
    2) coulomb
    """

    return None

def representation(model, basis='fock'):
    """
    transform into different representation:
    basis:
      1) fock
      2) GCS
    """

    return None

def getDsn(w, nfock):

    Dsn = [None] * nfock
    Dsn[0] = - w
    if nfock> 1:
        Dsn[1] = Dsn[0][...,None] - w[None,:]
    if nfock> 2:
        Dsn[2] = Dsn[1][...,None] - w[None,None,:]
    if nfock> 3:
        Dsn[3] = Dsn[2][...,None] - w[None,None,None,:]
    if nfock> 4:
        Dsn[4] = Dsn[3][...,None] - w[None,None,None,None,:]
    if nfock> 5:
        Dsn[5] = Dsn[4][...,None] - w[None,None,None,None,None,:]
    return Dsn 

def ccsd_pt2(model, options, iprint=0, ret=False):
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()
    eo,ev = model.energies()
    #eo = F.oo.diagonal()
    #ev = F.vv.diagonal()
    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]
    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]
    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))
    S1old = -H/w
    U11old = h.vo/D1p.transpose((0,2,1))
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]
    converged = False
    i = 0
    while i < max_iter and not converged:
        #T1,T2 = cc_equations.ccsd_simple(F, I, T1old, T2old)
        T1,T2 = cc_equations.ccsd_stanton(F, I, T1old, T2old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint > 0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1

    if not converged:
        print("WARNING: CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, (T1,T2,S1old,U11old))
    return (Ehf + Eold, Eold)

def epcc(model, options, porder=2, iprint=1, ret=False,theory='polariton'):
    """
    porder = order of photon operator (b^\dag) (not done yet)
    """
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()

    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print("Hartree-Fock energy: {}".format(Ehf))

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]

    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))
    S1old = -H/w
    U11old = h.vo/D1p.transpose((0,2,1))
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    su11 = numpy.sqrt(U11old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]

    converged = False
    i = 0
    while i < max_iter and not converged:
        #T1,T2,S1,U11 = epcc_equations.epcc_simple(
        #        F, I, w, g, h, G, H, T1old, T2old, S1old, U11old)
        T1,T2,S1,U11 = epcc_equations.epcc_opt(
                F, I, w, g, h, G, H, T1old, T2old, S1old, U11old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        U11 /= D1p.transpose((0,2,1))

        # using diis

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(U11old - U11)/su11

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        U11old = damp*U11old + (1.0 - damp)*U11

        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, (T1,T2,S1,U11))
    else: return (Ehf + Eold, Eold)


# for using diis
def vector_to_amps():

    return None

def amps_to_vector():

    return None


def epcc_nfock(model, options, iprint=1, ret=False,theory='polariton'):
    """
    nfock = order of photon operator (b^\dag) in pure photonic excitaiton (T_p)
    nfock2= order of photon operator in coupled excitaiton T_ep
    """
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()

    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print("Hartree-Fock energy: {}".format(Ehf))

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]

    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # DIIS

    # build MP2 T,S,U amplitudes
    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))
    S1old = -H/w
    U11old = h.vo/D1p.transpose((0,2,1))

    Eold = 0.0
    Eccsd = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eccsd))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    nfock = options['nfock']
    nfock2 = options['nfock2']
    useslow = False
    if 'slow' in options:
        useslow = options['slow']

    damp = options["damp"]
    if nfock < 1:
        print('nfock cannot be <1. It is changed to 1 instead!!!!')
        nfock = 1

    # ssn and su1n
    ssn = [None] * nfock
    su1n = [None] * nfock2

    print('test-zy: nfock/2=', nfock, nfock2)

    Snold = [None]*nfock
    U1nold = [None]*nfock2
    for k in range(nfock):
        if k == 0: 
            Snold[k] = -H/w
        else:
            shape  = [np for j in range(k+1)]
            #print('teset-zy: shape=', tuple(shape))
            Snold[k] = numpy.zeros(tuple(shape))
        ssn[k] = numpy.sqrt(Snold[k].size)

    for k in range(nfock2):
        if k == 0:
            U1nold[k] =  h.vo/D1p.transpose((0,2,1))
        else:
            shape  = [np for j in range(k+1)] + [nv, no]
            U1nold[k] =  numpy.zeros(tuple(shape))
        su1n[k] = numpy.sqrt(U1nold[k].size)

    #Ds1 = -w[:]
    #Ds2 = -w[:,None] - w[None,:]
    Dsn = getDsn(w, nfock)

    for k in range(nfock):
        print('Dsn[%d]'%k, Dsn[k])


    converged = False
    istep = 0
    while istep < max_iter and not converged:

        amps = (T1old, T2old, Snold, U1nold)
        if useslow:
            if nfock == nfock2 and nfock > 1:
                # support up to order 3
                T1,T2,Sn,U1n = epcc_equations_gen.qed_ccsd_sn_u1n_gen_equal(
                        F, I, w, g, h, G, H, nfock, amps)
            else:
                T1,T2,Sn,U1n = epcc_equations_gen.qed_ccsd_sn_u1n_gen(
                        F, I, w, g, h, G, H, nfock, nfock2, amps)

        else:
            #if nfock == nfock2 and nfock > 1:
            #    # not done yet! (todo)
            #    T1,T2,Sn,U1n = epcc_equations.qed_ccsd_sn_u1n_opt_equal(
            #            F, I, w, g, h, G, H, nfock, amps)
            #else:
            T1,T2,Sn,U1n = epcc_equations.qed_ccsd_sn_u1n_opt(
                    F, I, w, g, h, G, H, nfock, nfock2, amps)

        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        for k in range(nfock):
            #print('test-zy: sn', k, Sn[k])
            #Sn[k] /= -w 
            Sn[k] /= Dsn[k] # double check of higher order 

        for k in range(nfock2):
            U1n[k] /= D1p.transpose((0,2,1)) # currently it only works for order 1

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2

        # res of sn
        for k in range(nfock):
            res += numpy.linalg.norm(Snold[k] - Sn[k])/ssn[k]

        # res of U1n
        for k in range(nfock2):
            res += numpy.linalg.norm(U1nold[k] - U1n[k])/su1n[k] # 

        # for diis
        #tmpvec = amplitudes_to_vector(T1, T2, Sn, U1n)
        #tmpvec -+ amplitudes_to_vector(T1old, T2old, Snold, U1nold)
        #normt = numpy.linalg.norm(tmpvec)

        # Linear mixer
        if damp < 1.0:
            T1old = damp*T1old + (1.0 - damp)*T1
            T2old = damp*T2old + (1.0 - damp)*T2
            for k in range(nfock):
                Snold[k] = damp*Snold[k] + (1.0 - damp)*Sn[k]
            for k in range(nfock2):
                U1nold[k] = damp*U1nold[k] + (1.0 - damp)*U1n[k]
        
        # diis mixer (todo)
        #T1old, T2old, Snold, U1nold = diis([T1old,T2old,Snold,U1nold], 
        #        nfock1, nfock2, istep, normt, Eccsd - Eold, adiis)

        Eccsd = epcc_energy.energy(T1old,T2old,Snold[0],U1nold[0],F.ov,I.oovv,w,g.ov,G)

        Ediff = abs(Eccsd - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(istep+1, Eccsd, res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = Eccsd
        istep = istep + 1
    if iprint>0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, (T1,T2,Sn,U1n))
    else: return (Ehf + Eold, Eold)

def epccsd_2_s1(model, options, iprint=1, guess=None, ret=False):
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]
    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print("Hartree-Fock energy: {}".format(Ehf))

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]
    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    Ds2 = -w[:,None] - w[None,:]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    if guess is None:
        T1old = F.vo/D1.transpose((1,0))
        T2old = I.vvoo/D2.transpose((2,3,0,1))
        S1old = -H/w
        S2old = numpy.zeros((np,np))
        S3old = numpy.zeros((np,np,np))
        U11old = h.vo/D1p.transpose((0,2,1)) #+ 0.01*numpy.random.rand(np,nv,no)
    else:
        T1old = guess[0]
        T2old = guess[1]
        S1old = guess[2]
        S2old = guess[3]
        S3old = guess[4]
        U11old = guess[5]

    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)

    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    ss2 = numpy.sqrt(S2old.size)
    su11 = numpy.sqrt(U11old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]
    converged = False
    i = 0
    while i < max_iter and not converged:
        T1,T2,S1,S2,U11 = epcc_equations_gen.epcc2_11_gen(
                F, I, w, g, h, G, H, T1old, T2old, S1old, S2old, U11old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        S2 /= Ds2
        U11 /= D1p.transpose((0,2,1))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(S2old - S2)/ss2
        res += numpy.linalg.norm(U11old - U11)/su11

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        S2old = damp*S2old + (1.0 - damp)*S2
        U11old = damp*U11old + (1.0 - damp)*U11
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, T1, T2, S1, S2, U11)
    else:
        return (Ehf + Eold, Eold)

def epccsd_2_s2(model, options, iprint=1, guess=None, ret=False):
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]
    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print("Hartree-Fock energy: {}".format(Ehf))

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]
    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    D2p = eo[None,None,:,None] - ev[None,None,None,:] - w[:,None,None,None] - w[None,:,None,None]
    Ds2 = -w[:,None] - w[None,:]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    if guess is None:
        T1old = F.vo/D1.transpose((1,0))
        T2old = I.vvoo/D2.transpose((2,3,0,1))
        S1old = -H/w
        S2old = numpy.zeros((np,np))
        U11old = h.vo/D1p.transpose((0,2,1))
        U12old = numpy.zeros((np,np,nv,no))
    else:
        T1old = guess[0]
        T2old = guess[1]
        S1old = guess[2]
        S2old = guess[3]
        U11old = guess[4]
        U12old = guess[5]
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    ss2 = numpy.sqrt(S2old.size)
    su11 = numpy.sqrt(U11old.size)
    su12 = numpy.sqrt(U12old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]
    converged = False
    i = 0
    while i < max_iter and not converged:
        T1,T2,S1,S2,U11,U12 = epcc_equations_gen.epcc2_12_gen(
                F, I, w, g, h, G, H, T1old, T2old, S1old, S2old, U11old, U12old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        S2 /= Ds2
        U11 /= D1p.transpose((0,2,1))
        U12 /= D2p.transpose((0,1,3,2))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(S2old - S2)/ss2
        res += numpy.linalg.norm(U11old - U11)/su11
        res += numpy.linalg.norm(U12old - U12)/su12

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        S2old = damp*S2old + (1.0 - damp)*S2
        U11old = damp*U11old + (1.0 - damp)*U11
        U12old = damp*U12old + (1.0 - damp)*U12
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, T1, T2, S1, S2, U11, U12)
    else:
        return (Ehf + Eold, Eold)


def epccsd_3_s3(model, options, iprint=1, guess=None, ret=False):
    """
    P3 and braP3 have been added into the code generator
    TODO: fnish this funciton, generate the code for ccsd3_3_s3 (put into epcc_equations.py)
    """
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]
    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print("Hartree-Fock energy: {}".format(Ehf))

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]
    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    D2p = eo[None,None,:,None] - ev[None,None,None,:] - w[:,None,None,None] - w[None,:,None,None]
    Ds2 = -w[:,None] - w[None,:]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    if guess is None:
        T1old = F.vo/D1.transpose((1,0))
        T2old = I.vvoo/D2.transpose((2,3,0,1))
        S1old = -H/w
        S2old = numpy.zeros((np,np))
        U11old = h.vo/D1p.transpose((0,2,1))
        U12old = numpy.zeros((np,np,nv,no))
    else:
        T1old = guess[0]
        T2old = guess[1]
        S1old = guess[2]
        S2old = guess[3]
        U11old = guess[4]
        U12old = guess[5]
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    ss2 = numpy.sqrt(S2old.size)
    su11 = numpy.sqrt(U11old.size)
    su12 = numpy.sqrt(U12old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]
    converged = False
    i = 0
    while i < max_iter and not converged:
        T1,T2,S1,S2,U11,U12 = epcc_equations_gen.epcc2_12_gen(
                F, I, w, g, h, G, H, T1old, T2old, S1old, S2old, U11old, U12old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        S2 /= Ds2
        U11 /= D1p.transpose((0,2,1))
        U12 /= D2p.transpose((0,1,3,2))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(S2old - S2)/ss2
        res += numpy.linalg.norm(U11old - U11)/su11
        res += numpy.linalg.norm(U12old - U12)/su12

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        S2old = damp*S2old + (1.0 - damp)*S2
        U11old = damp*U11old + (1.0 - damp)*U11
        U12old = damp*U12old + (1.0 - damp)*U12
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: ep-CCSD did not converge!")
    if ret:
        return (Ehf + Eold, Eold, T1, T2, S1, S2, U11, U12)
    else:
        return (Ehf + Eold, Eold)



    return None



### BRADEN WEIGHT ###

def QED_CC_T1_T2_S1_U11__OLD(model, options, iprint=0, return_AMPS=False, use_Guess=False, guess_AMPS=None):

    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()

    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print(f"HF Energy: {round(Ehf,4)} a.u.")

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]

    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    if ( use_Guess == True ):
        T1old  = guess_AMPS[0]
        T2old  = guess_AMPS[1]
        S1old  = guess_AMPS[2]
        U11old = guess_AMPS[3]

    else:
        # Build MP2 T,S,U amplitudes
        T1old  = F.vo/D1.transpose((1,0))
        T2old  = I.vvoo/D2.transpose((2,3,0,1))
        S1old  = -H/w
        U11old = h.vo/D1p.transpose((0,2,1))
    
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    #if iprint>0:
    #    print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    su11 = numpy.sqrt(U11old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]

    converged = False
    issue_flag = False
    i = 0
    Estart = Eold * 1
    T1_OLD_save  = T1old  * 1
    T2_OLD_save  = T2old  * 1
    S1_OLD_save  = S1old  * 1
    U11_OLD_save = U11old * 1
    while i < max_iter and not converged:
        T1,T2,S1,U11 = epcc_equations_gen.QED_CC_T1_T2_S1_U11_gen(
                F, I, w, g, h, G, H, T1old, T2old, S1old, U11old)
        
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        U11 /= D1p.transpose((0,2,1))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(U11old - U11)/su11

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        U11old = damp*U11old + (1.0 - damp)*U11
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        
        if ( iprint > 0 or issue_flag == True ):
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        
        if ( math.isnan(Ediff) ):
            #assert(False), f"Convergence encountered 'Nan' during SCF cycle. Quitting."
            print(f"\t\t!!!!!! WARNING !!!!!!")
            print(f"\t!!!!!! Convergence encountered 'Nan' during SCF cycle.")
            print(f"\t!!!!!! Starting again with stronger damping.")
            damp = damp * 1.01
            if ( damp > 0.999 ):
                print("damp too strong. Quitting.")
                exit()
            Eold = Estart * 1
            E    = Estart * 1
            issue_flag = True
            T1old, T2old, S1old, U11old  = T1_OLD_save, T2_OLD_save, S1_OLD_save, U11_OLD_save
            if ( issue_flag == True and damp > 0.99 ):
                print(f"\t\t!!!!!! WARNING !!!!!!")
                print(f"\t!!!!!! Convergence encountered 'Nan' during SCF cycle.")
                print(f"\t!!!!!! Stronger damping was not effective. Starting with new guess.")
                # Start from scratch, as maybe the guess functions are bad
                T1old  = F.vo/D1.transpose((1,0))
                T2old  = I.vvoo/D2.transpose((2,3,0,1))
                S1old  = -H/w
                U11old = h.vo/D1p.transpose((0,2,1))
            continue
            
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: Did not converge!")
    
    Ediff_old = Ediff * 1
    
    if ( return_AMPS == True):
        return Ehf + Eold, Eold, (T1,T2,S1,U11)
    else: 
        return Ehf + Eold, Eold


def QED_CC_T1_T2_S1_U11(model, options, iprint=1, return_AMPS=False, use_Guess=False, guess_AMPS=None):
    """
    porder = order of photon operator (b^\dag) (not done yet)
    """
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()

    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print(f"HF Energy: {round(Ehf,4)} a.u.")

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]

    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))
    S1old = -H/w
    U11old = h.vo/D1p.transpose((0,2,1))
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    su11 = numpy.sqrt(U11old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]

    converged = False
    i = 0
    while i < max_iter and not converged:
        T1,T2,S1,U11 = epcc_equations_gen.QED_CC_T1_T2_S1_U11_gen(
                F, I, w, g, h, G, H, T1old, T2old, S1old, U11old)

        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        U11 /= D1p.transpose((0,2,1))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(U11old - U11)/su11

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        U11old = damp*U11old + (1.0 - damp)*U11
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: Did not converge!")
    if return_AMPS:
        return Ehf + Eold, Eold, (T1,T2,S1,U11)
    else: 
        return (Ehf + Eold, Eold)

def QED_CC_T1_T2_S1_U11__original_Yu_EQNS(model, options, iprint=1, return_AMPS=False, use_Guess=False, guess_AMPS=None):
    """
    porder = order of photon operator (b^\dag) (not done yet)
    """
    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    F = model.g_fock()

    eo = F.oo.diagonal()
    ev = F.vv.diagonal()

    D1 = eo[:,None] - ev[None,:]
    D2 = eo[:,None,None,None] + eo[None,:,None,None]\
            - ev[None,None,:,None] - ev[None,None,None,:]

    F.oo = F.oo - numpy.diag(eo)
    F.vv = F.vv - numpy.diag(ev)

    no = eo.shape[0]
    nv = ev.shape[0]

    # get HF energy
    Ehf = model.hf_energy()
    if iprint>0:
        print(f"HF Energy: {round(Ehf,4)} a.u.")

    # get ERIs
    I = model.g_aint()

    # get normal mode energies
    w = model.omega()
    np = w.shape[0]

    D1p = eo[None,:,None] - ev[None,None,:] - w[:,None,None]
    np,no,nv = D1p.shape

    # get elec-phon matrix elements
    g,h = model.gint()
    G,H = model.mfG()

    # build MP2 T,S,U amplitudes
    T1old = F.vo/D1.transpose((1,0))
    T2old = I.vvoo/D2.transpose((2,3,0,1))
    S1old = -H/w
    U11old = h.vo/D1p.transpose((0,2,1))
    Eold = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
    if iprint>0:
        print("Guess energy: {:.8f}".format(Eold))

    # sqare root of numbers of elements
    st1 = numpy.sqrt(T1old.size)
    st2 = numpy.sqrt(T2old.size)
    ss1 = numpy.sqrt(S1old.size)
    su11 = numpy.sqrt(U11old.size)

    # coupled cluster iterations
    ethresh = options["ethresh"]
    tthresh = options["tthresh"]
    max_iter = options["max_iter"]
    damp = options["damp"]

    converged = False
    i = 0
    while i < max_iter and not converged:
        T1,T2,S1,U11 = epcc_equations_gen.epcc1_11_gen_OLD(
                F, I, w, g, h, G, H, T1old, T2old, S1old, U11old)
        T1 /= D1.transpose((1,0))
        T2 /= D2.transpose((2,3,0,1))
        S1 /= -w
        U11 /= D1p.transpose((0,2,1))

        res = numpy.linalg.norm(T1old - T1)/st1
        res += numpy.linalg.norm(T2old - T2)/st2
        res += numpy.linalg.norm(S1old - S1)/ss1
        res += numpy.linalg.norm(U11old - U11)/su11

        T1old = damp*T1old + (1.0 - damp)*T1
        T2old = damp*T2old + (1.0 - damp)*T2
        S1old = damp*S1old + (1.0 - damp)*S1
        U11old = damp*U11old + (1.0 - damp)*U11
        E = epcc_energy.energy(T1old,T2old,S1old,U11old,F.ov,I.oovv,w,g.ov,G)
        Ediff = abs(E - Eold)
        if iprint>0:
            print(' {:2d}  {:.10f}   {:.4E}'.format(i+1,E,res))
        if Ediff < ethresh and res < tthresh:
            converged = True
        Eold = E
        i = i + 1
    if iprint>0:
        if not converged:
            print("WARNING: Did not converge!")
    if return_AMPS:
        return Ehf + Eold, Eold, (T1,T2,S1,U11)
    else: 
        return (Ehf + Eold, Eold)


def save_diis_CC_T1_T2_S1_U11( T1, T2, S1, U11 ):

    """
    Use only for serial jobs, else read-write issues may occur.
    """

    # Make scratch folder if it does not exist
    SCRATCH_DIR = "SCRATCH_CC/"
    sp.call(f"mkdir -p {SCRATCH_DIR}",shell=True)

    numpy.save(f"{SCRATCH_DIR}/T1.npy", T1)
    numpy.save(f"{SCRATCH_DIR}/T2.npy", T2)
    numpy.save(f"{SCRATCH_DIR}/S1.npy", S1)
    numpy.save(f"{SCRATCH_DIR}/U11.npy", U11)

    # For de-bugging
    """
    print(f"T1 Shape = {numpy.shape(T1)}")
    print(f"T2 Shape = {numpy.shape(T2)}")
    print(f"S1 Shape = {numpy.shape(S1)}")
    print(f"U11 Shape = {numpy.shape(U11)}")

    T1_OLD = numpy.load(f"{SCRATCH_DIR}/T1.npy")
    T2_OLD = numpy.load(f"{SCRATCH_DIR}/T2.npy")
    S1_OLD = numpy.load(f"{SCRATCH_DIR}/S1.npy")
    U11_OLD = numpy.load(f"{SCRATCH_DIR}/U11.npy")

    print( (T1_OLD == T1).all() )
    print( (T2_OLD == T2).all() )
    print( (S1_OLD == S1).all() )
    print( (U11_OLD == U11).all() )
    """
    
def load_diis_CC_T1_T2_S1_U11():

    SCRATCH_DIR = "SCRATCH_CC/"

    if ( os.path.isfile(f"{SCRATCH_DIR}/T1.npy") and \
         os.path.isfile(f"{SCRATCH_DIR}/T2.npy") and \
         os.path.isfile(f"{SCRATCH_DIR}/S1.npy") and \
         os.path.isfile(f"{SCRATCH_DIR}/U11.npy") ):

        T1  = numpy.load(f"{SCRATCH_DIR}/T1.npy")
        T2  = numpy.load(f"{SCRATCH_DIR}/T2.npy")
        S1  = numpy.load(f"{SCRATCH_DIR}/S1.npy")
        U11 = numpy.load(f"{SCRATCH_DIR}/U11.npy")

        return T1, T2, S1, U11

    else:
        print("No scratch files found. Skipping.")
        return None, None, None, None



