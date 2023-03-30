
import sys
import numpy
from pyscf import lib
from pyscf.pbc.lib import linalg_helper as pbc_linalg
from cqcpy import cc_equations
from . import epcc_energy
from .eom_epcc_equations import eom_ee_epccsd_1_s1_sigma_slow
from .eom_epcc_equations import eom_ip_epccsd_1_s1_sigma_slow
from .eom_epcc_equations import eom_ea_epccsd_1_s1_sigma_slow
from .eom_epcc_equations import eom_ee_epccsd_1_s1_sigma_opt
from .eom_epcc_equations import eom_ip_epccsd_1_s1_sigma_opt
from .eom_epcc_equations import eom_ea_epccsd_1_s1_sigma_opt

from .eom_epcc_equations import eom_ee_epccsd_n_sn_sigma_opt
from .eom_epcc_equations_slow import eom_ee_epccsd_n_sn_sigma_slow

from .eom_epcc_equations import EEImds
from .eom_epcc_equations import IPImds
from .eom_epcc_equations import EAImds

def vec_size(dims,NFock=1):
    np,nv,no = dims
    ss = nv*no
    sd = nv*(nv - 1)//2*no*(no - 1)//2
    sn = numpy.sum(np**numpy.arange(1,NFock+1) ) # np*NFock
    ss1 = np*no*nv
    #print( ss,sd,sn,ss1,nv )
    return (ss + sd + sn + ss1)

def vec_size_ip(dims):
    np,nv,no = dims
    ss = no
    sd = nv*no*(no - 1)//2
    ss1 = np*no
    return (ss + sd + ss1)

def vec_size_ea(dims):
    np,nv,no = dims
    ss = nv
    sd = nv*(nv - 1)//2*no
    ss1 = np*nv
    return (ss + sd + ss1)

def vector_to_amplitudes(dims, vector):
    np,nv,no = dims

    nrs = no*nv
    nrd = no*(no - 1)//2*nv*(nv - 1)//2
    nrs1 = nrs*np

    rs = vector[:nrs].copy()
    rdt = vector[nrs:nrs+nrd]
    r1 = vector[nrs+nrd:nrs+nrd+np].copy()
    rs1 = vector[nrs+nrd+np:].copy()
    rd = numpy.zeros((nv,nv,no,no),dtype=vector.dtype)
    count = 0
    for a in range(nv):
        for i in range(no):
             for b in range(a+1,nv):
                 for j in range(i+1,no):
                     rd[a,b,i,j] = rdt[count]
                     rd[a,b,j,i] = -1.0*rdt[count]
                     rd[b,a,i,j] = -1.0*rdt[count]
                     rd[b,a,j,i] = rdt[count]
                     count += 1

    rs = rs.reshape(nv,no)
    rs1 = rs1.reshape(np,nv,no)
    return (rs, rd, r1, rs1)

def vector_to_amplitudes_Sn_U11(dims, vector, nfock1=1):
    np,nv,no = dims

    nrs = no*nv
    nrd = no*(no - 1)//2*nv*(nv - 1)//2
    nrs1 = nrs*np

    nrn = numpy.sum( np**numpy.arange(1,nfock1+1) )
    assert (nrn == len(vector) - nrs - nrd - nrs1), f"Print 'nrn' and 'subtraction' do not match! {nrn} != {len(vector) - nrs - nrd - nrs1}"

    rs = vector[:nrs].copy()
    rdt = vector[nrs:nrs+nrd]
    
    rn = []  #[None] * nfock1
    for j_fock in range( nfock1 ):
        start = nrs+nrd + numpy.sum( np**numpy.arange(j_fock) )
        end   = nrs+nrd + numpy.sum( np**numpy.arange(j_fock+1) )
        rn.append( [vector[start:end].copy().reshape( [np for j in range(j_fock)] ) ] )
        #rn[j_fock] = vector[start:end].copy().reshape( [np for j in range(j_fock)] )

    

    rs1 = vector[nrs+nrd+nrn:].copy()
    rd = numpy.zeros((nv,nv,no,no),dtype=vector.dtype)
    count = 0
    for a in range(nv):
        for i in range(no):
             for b in range(a+1,nv):
                 for j in range(i+1,no):
                     rd[a,b,i,j] = rdt[count]
                     rd[a,b,j,i] = -1.0*rdt[count]
                     rd[b,a,i,j] = -1.0*rdt[count]
                     rd[b,a,j,i] = rdt[count]
                     count += 1

    rs = rs.reshape(nv,no)
    rs1 = rs1.reshape(np,nv,no)

    #print( "Shape rn [rn]", numpy.shape(rn), numpy.shape([rn]))
    #print( "Shape rs1 [rs1]", numpy.shape(rs1), numpy.shape([rs1]))


    return rs, rd, rn, [rs1]


def amplitudes_to_vector(dims, rs, rd, r1, rs1):
    np,nv,no = dims

    nrs = no*nv
    nrd = no*(no - 1)//2*nv*(nv - 1)//2
    nrs1 = nrs*np
    vector = numpy.zeros((nrs + nrd + np + nrs1), dtype=rs.dtype)
    rdt = numpy.zeros((nrd))
    count = 0
    for a in range(nv):
        for i in range(no):
             for b in range(a+1,nv):
                 for j in range(i+1,no):
                     rdt[count] = rd[a,b,i,j]
                     count += 1

    vector[:nrs] = rs.reshape(-1)
    vector[nrs:nrs + nrd] = rdt.reshape(-1)
    vector[nrs+nrd:nrs+nrd+np] = r1.reshape(-1)
    vector[nrs+nrd+np:] = rs1.reshape(-1)
    return vector

def amplitudes_to_vector_Sn_U11(dims, rs, rd, rn, rsn):
    # Rn = list of Sn amplitudes
    np,nv,no = dims

    rs1 = rsn[0]

    NFock = len(rn)

    nrs  = no*nv
    nrd  = no*(no - 1)//2*nv*(nv - 1)//2
    nrs1 = nrs*np
    nrn  = numpy.sum( np**numpy.arange(1,NFock+1) )
    vector = numpy.zeros((nrs + nrd + nrn + nrs1), dtype=rs.dtype)
    rdt = numpy.zeros((nrd))
    count = 0
    for a in range(nv):
        for i in range(no):
             for b in range(a+1,nv):
                 for j in range(i+1,no):
                     rdt[count] = rd[a,b,i,j]
                     count += 1

    vector[:nrs] = rs.reshape(-1)
    vector[nrs:nrs + nrd] = rdt.reshape(-1)

    """
    1 = np    --> nrs+nrd,              nrs+nrd + np
    2 = np**2 --> nrs+nrd + np,         nrs+nrd + np + np**2
    3 = np**3 --> nrs+nrd + np + np**2, nrs+nrd + np + np**2 + np**3
    n = np**n --> nrs+nrd + numpy.sum( np**np.arange(n) ), nrs+nrd + numpy.sum( np**np.arange(n+1) )
    """

    for j_fock in range( NFock ):
        #start = nrs+nrd + np * j_fock
        #end   = nrs+nrd + np * (j_fock+1)
        #vector[start:end] = rn[j_fock].reshape(-1)

        start = nrs+nrd + numpy.sum( np**numpy.arange(j_fock) )
        end   = nrs+nrd + numpy.sum( np**numpy.arange(j_fock+1) )
        vector[start:end] = rn[j_fock].reshape(-1)


    vector[nrs+nrd+nrn:] = rs1.reshape(-1)

    #print( f"Length of vector: {len(vector)}, {nrs + nrd + nrn + nrs1}" )

    return vector

def vector_to_amplitudes_ip(dims, vector):
    np,nv,no = dims

    nrs = no
    nrd = no*(no - 1)//2*nv
    nrs1 = no*np

    rs = vector[:nrs].copy()
    rdt = vector[nrs:nrs+nrd]
    rs1 = vector[nrs+nrd:].copy()
    rd = numpy.zeros((nv,no,no),dtype=vector.dtype)
    count = 0
    for a in range(nv):
        for i in range(no):
             for j in range(i+1,no):
                 rd[a,i,j] = rdt[count]
                 rd[a,j,i] = -1.0*rdt[count]
                 count += 1

    rs = rs.reshape(no)
    rs1 = rs1.reshape(np,no)
    return (rs, rd, rs1)

def amplitudes_to_vector_ip(dims, rs, rd, rs1):
    np,nv,no = dims

    nrs = no
    nrd = no*(no - 1)//2*nv
    nrs1 = no*np
    vector = numpy.zeros((nrs + nrd + nrs1), dtype=rs.dtype)
    rdt = numpy.zeros((nrd), dtype=rd.dtype)
    count = 0
    for a in range(nv):
        for i in range(no):
             for j in range(i+1,no):
                 rdt[count] = rd[a,i,j]
                 count += 1

    vector[:nrs] = rs.reshape(-1)
    vector[nrs:nrs + nrd] = rdt.reshape(-1)
    vector[nrs+nrd:] = rs1.reshape(-1)
    return vector

def vector_to_amplitudes_ea(dims, vector):
    np,nv,no = dims

    nrs = nv
    nrd = nv*(nv - 1)//2*no
    nrs1 = nv*np

    rs = vector[:nrs].copy()
    rdt = vector[nrs:nrs+nrd]
    rs1 = vector[nrs+nrd:].copy()
    rd = numpy.zeros((nv,nv,no), dtype=vector.dtype)
    count = 0
    for a in range(nv):
        for b in range(a+1,nv):
            for i in range(no):
                 rd[a,b,i] = rdt[count]
                 rd[b,a,i] = -1.0*rdt[count]
                 count += 1

    rs = rs.reshape(nv)
    rs1 = rs1.reshape(np,nv)
    return (rs, rd, rs1)

def amplitudes_to_vector_ea(dims, rs, rd, rs1):
    np,nv,no = dims

    nrs = nv
    nrd = nv*(nv - 1)//2*no
    nrs1 = nv*np
    vector = numpy.zeros((nrs + nrd + nrs1), dtype=rs.dtype)
    rdt = numpy.zeros((nrd), dtype=rs.dtype)
    count = 0
    for a in range(nv):
        for b in range(a+1,nv):
            for i in range(no):
                 rdt[count] = rd[a,b,i]
                 count += 1

    vector[:nrs] = rs.reshape(-1)
    vector[nrs:nrs + nrd] = rdt.reshape(-1)
    vector[nrs+nrd:] = rs1.reshape(-1)
    return vector

# for left eigenstates 
def lefteom_ee_epccsd_1_s1_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):

    rs, rd, r1, rs1 = vector_to_amplitudes(dims, vector)

    sigs, sigd, sig1, sigs1 = lefteom_ee_epccsd_1_s1_sigma_opt(rs, rd, r1, rs1, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector(dims, sigs, sigd, sig1, sigs1)

# for right eigenstate
def eom_ee_epccsd_1_s1_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):

    rs, rd, r1, rs1 = vector_to_amplitudes(dims, vector)

    #sigs, sigd, sig1, sigs1 = eom_ee_epccsd_1_s1_sigma_slow(rs, rd, r1, rs1, amps, F, I, w, g, h, G, H)
    sigs, sigd, sig1, sigs1 = eom_ee_epccsd_1_s1_sigma_opt(rs, rd, r1, rs1, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector(dims, sigs, sigd, sig1, sigs1)


def eom_ee_epccsd_1_s1(model, options, amps, verbose=0, analysis=False, calc_nac=False):
    T1, T2, S1, U11 = amps
    nm, nv, no = U11.shape
    nroots = options["nroots"]
    conv_tol = options["conv_tol"]
    max_space = options["max_space"]
    max_cycle = options["max_cycle"]

    F = model.g_fock()
    I = model.g_aint()
    w = model.omega()
    g,h = model.gint()
    G,H = model.mfG()

    # preconditioner
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    d1 = w.copy()
    ds = (ev[:,None] - eo[None,:])
    dd = (ev[:,None,None,None] + ev[None,:,None,None] - eo[None,None,:,None] - eo[None,None,None,:])
    ds1 = (w[:,None,None] + ev[None,:,None] - eo[None,None,:])
    diag = amplitudes_to_vector((nm, nv, no), ds, dd, d1, ds1)
    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)
        #return r/(e0 - Hp + 1.e-12) 

    gidx = numpy.argsort(diag)

    # generate guess
    guess = []
    for i in range(nroots):
        # noise first
        R1 = 1e-5*numpy.random.rand(nm)/numpy.sqrt(nm)
        RS = 1e-5*numpy.random.rand(nv, no)/numpy.sqrt(nv*no)
        RD = 1e-5*numpy.random.rand(nv, nv, no, no)/numpy.sqrt(no*no*nv*nv)
        RS1 = 1e-5*numpy.random.rand(nm, nv, no)/numpy.sqrt(nv*no*nm)
        Rnoise = amplitudes_to_vector((nm, nv, no), RS, RD, R1, RS1)
        Rguess = numpy.zeros(vec_size((nm, nv, no)))
        Rguess[gidx[i]] = 1.0
        Rguess += Rnoise
        Rguess = Rguess/numpy.linalg.norm(Rguess)
        guess.append(Rguess)

    # matvec
    imds = EEImds()
    imds.build(amps, F, I, w, g, h, G, H)

    # H|x_i>;  x_j | H|x_i> ==> H_ij  
    # for x in vec:
    #   yi = matvec(x) 
    #   for z in vec:
    #      dot_proct(z, yi) --> H_ij

    matvec = lambda vec: [eom_ee_epccsd_1_s1_matvec((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]
    #matvec = lambda vec: [eom_ee_epccsd_1_s1_matvec((no, nv, nm), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]

    # davidson iterations
    eig = lib.davidson_nosym1
    conv, es, vs = eig(matvec, guess, precond,
                       tol=conv_tol, max_cycle=max_cycle,
                       max_space=max_space, nroots=nroots,verbose=verbose)

    # cleanup
    vs = [x/numpy.linalg.norm(x) for x in vs]

    if analysis:
        szs = numpy.zeros((nv, no))
        szd = numpy.zeros((nv, nv, no, no))
        noa = no//2
        nva = nv//2
        ones = numpy.ones((nva,noa))
        nones = -1.0*numpy.ones((nva,noa))
        szs[:nva,noa:] = ones
        szs[nva:,:noa] = nones
        ones = numpy.ones((nva,nva,noa,noa))
        nones = -1.0*ones
        twos = 2.0*ones
        ntwos = -1*twos
        szd[:nva,:nva,:noa,noa:] = ones
        szd[:nva,:nva,noa:,:noa] = ones
        szd[nva:,nva:,:noa,noa:] = nones
        szd[nva:,nva:,noa:,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = nones
        szd[nva:,:nva,:noa,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = ones
        szd[nva:,:nva,:noa,:noa] = ones
        szd[nva:,nva:,:noa,:noa] = ntwos
        szd[:nva,:nva,noa:,noa:] = twos

        print("State    Energy        Sz        |RS|      |RSa+b|     |RD|       |R1|       |RS1|   Singlet?")
        for i,x in enumerate(vs):
            cs = vector_to_amplitudes((nm, nv, no), x)
            rs,rd,r1,rs1 = cs
            rsa = rs[:nv//2, :no//2]
            rsb = rs[nv//2:, no//2:]
            rsab = rs[:nv//2, no//2:]
            ds = numpy.linalg.norm(rs)
            dsab = numpy.linalg.norm(rsa+rsb)
            dd = numpy.sqrt(0.25*numpy.einsum('abij,abij->', rd, rd.conj()))
            d1 = numpy.sqrt(numpy.einsum('I,I->', r1, r1.conj()))
            ds1 = numpy.sqrt(numpy.einsum('Iai,Iai->',rs1,rs1.conj()))
            sz = numpy.einsum('ai,ai->', szs, rs)
            for I in range(nm): sz += numpy.einsum('ai,ai->', szs, rs1[I])
            sz += 0.25*numpy.einsum('abij,abij', szd, rd)
            singlet = (dsab > 1.e-2*ds)
            if d1 > 0.99: singlet = "Photon" # pure photon state
            print("{:3d}:   {:10.3E} {:10.3E} {:10.3E} {:10.3E} {:10.3E} {:10.3E} {:10.3E}  {:5s}".format(i,es[i]*27.2114,sz,ds,dsab,dd,d1,ds1,str(singlet)))

    if calc_nac:
        nac = eom_cc_epccsd_1_s1_nac()  # TODO
        return es, nac
    return es

# general nfock case; right eigenstate
def eom_ee_epccsd_n_sn_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):
    
    T1, T2, Sn, U1n = amps
    nfock1 = len(Sn)
    nfock2 = len(U1n)

    rs, rd, rn, rsn = vector_to_amplitudes_Sn_U11(dims, vector, nfock1=nfock1) # extend it into nfock case

    #sigs, sigd, sign, sigsn = eom_ee_epccsd_n_sn_sigma_slow(nfock1, nfock2, rs, rd, rn, rsn, amps, F, I, w, g, h, G, H, imds=imds)
    sigs, sigd, sign, sigsn = eom_ee_epccsd_n_sn_sigma_opt(nfock1, nfock2, rs, rd, rn, rsn, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector_Sn_U11(dims, sigs, sigd, sign, sigsn) # extend it into nfock case

def eom_ee_epccsd_n_sn_matvec_slow(dims, vector, amps, F, I, w, g, h, G, H, imds=None):
    
    T1, T2, Sn, U1n = amps
    nfock1 = len(Sn)
    nfock2 = len(U1n)

    rs, rd, rn, rsn = vector_to_amplitudes_Sn_U11(dims, vector, nfock1=nfock1) # extend it into nfock case

    sigs, sigd, sign, sigsn = eom_ee_epccsd_n_sn_sigma_slow(nfock1, nfock2, rs, rd, rn, rsn, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector_Sn_U11(dims, sigs, sigd, sign, sigsn) # extend it into nfock case

def eom_ee_s2(amps, imds=None):
    # return the expectation value of s^2 (TODO)

    s2 = 0.0
    return s2

def eom_ee_epccsd_n_sn_singlet(model, options, amps, verbose=0, analysis=False, calc_nac=False):

    return None

def eom_ee_epccsd_n_sn_triplet(model, options, amps, verbose=0, analysis=False, calc_nac=False):

    return None

cqed_eomccsd_singlet = eom_ee_epccsd_n_sn_singlet
cqed_eomccsd_triplet = eom_ee_epccsd_n_sn_triplet

def eom_ee_epccsd_n_sn(model, options, amps, verbose=0, analysis=False, calc_nac=False):
    T1, T2, Sn, U1n = amps
    nm, nv, no = U1n[0].shape
    nroots = options["nroots"]
    conv_tol = options["conv_tol"]
    max_space = options["max_space"]
    max_cycle = options["max_cycle"]
    nfock1 = len(Sn)
    nfock2 = len(U1n)
    useslow = False
    if 'slow' in options:
        useslow = options['slow']
    
    F = model.g_fock()
    I = model.g_aint()
    w = model.omega()
    g,h = model.gint()
    G,H = model.mfG()

    # preconditioner
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    ds = (ev[:,None] - eo[None,:])
    dd = (ev[:,None,None,None] + ev[None,:,None,None] - eo[None,None,:,None] - eo[None,None,None,:])
    dn = [None] * nfock1
    for fock in range( nfock1 ):
        dn[fock] = numpy.zeros([nm for j in range(1,fock+1)])
        if fock == 0: 
            dn[fock] = w.copy()
        elif fock == 1:
            dn[fock] = dn[fock-1][...,None] + w[None,:]
        elif fock == 2:
            dn[fock] = dn[fock-1][...,None] + w[None,None,:]
        elif fock == 3:
            dn[fock] = dn[fock-1][...,None] + w[None,None,None:]
        elif fock == 4:
            dn[fock] = dn[fock-1][...,None] + w[None,None,None,None:]
        elif fock == 5:
            dn[fock] = dn[fock-1][...,None] + w[None,None,None,None,None:]
            

    ds1 = (w[:,None,None] + ev[None,:,None] - eo[None,None,:])
    diag = amplitudes_to_vector_Sn_U11((nm, nv, no), ds, dd, dn, ds1)

    def precond(r, e0, x0):
        return r /(e0-diag+1e-12)

    gidx = numpy.argsort(diag)

    # generate guess
    Rn = [None] * nfock1
    RSn = [None] * nfock2

    guess = []
    for i in range(nroots):
        # noise first
        RS = 1e-5*numpy.random.rand(nv, no)/numpy.sqrt(nv*no)
        RD = 1e-5*numpy.random.rand(nv, nv, no, no)/numpy.sqrt(no*no*nv*nv)
        for k in range(nfock1):
            shape  = [nm for j in range(k+1)]
            tmp = 1e-5*numpy.random.rand(nm**(k+1))/numpy.sqrt(nm**(k+1))
            Rn[k] = tmp.reshape(shape)

        RS1 = 1e-5*numpy.random.rand(nm, nv, no)/numpy.sqrt(nv*no*nm)
        RSn[0] = RS1

        Rnoise = amplitudes_to_vector_Sn_U11((nm, nv, no), RS, RD, Rn, RSn) 
        Rguess = numpy.zeros(vec_size((nm, nv, no), NFock=nfock1))
        Rguess[gidx[i]] = 1.0
        Rguess += Rnoise
        Rguess = Rguess/numpy.linalg.norm(Rguess)
        guess.append(Rguess)

    # matvec
    amps_s1_U11 = (T1, T2, Sn[0], U1n[0]) # Create small amps
    imds = EEImds()
    imds.build(amps_s1_U11, F, I, w, g, h, G, H)

    if useslow:
        matvec = lambda vec: [eom_ee_epccsd_n_sn_matvec_slow((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]
    else:
        matvec = lambda vec: [eom_ee_epccsd_n_sn_matvec((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]

    # davidson iterations
    eig = lib.davidson_nosym1
    conv, es, vs = eig(matvec, guess, precond,
                       tol=conv_tol, max_cycle=max_cycle,
                       max_space=max_space, nroots=nroots,verbose=verbose)

    # cleanup
    vs = [x/numpy.linalg.norm(x) for x in vs]

    if analysis:
        szs = numpy.zeros((nv, no))
        s2s = numpy.zeros((nv, no)) # BMW
        szd = numpy.zeros((nv, nv, no, no))
        s2d = numpy.zeros((nv, nv, no, no)) # BMW
        noa = no//2
        nva = nv//2
        ones = numpy.ones((nva,noa))
        nones = -1.0*numpy.ones((nva,noa))

        szs[:nva,noa:] = ones
        szs[nva:,:noa] = nones

        s2s[:nva,noa:] = ones # BMW
        s2s[nva:,:noa] = ones # BMW

        ones = numpy.ones((nva,nva,noa,noa))
        nones = -1.0*ones
        twos = 2.0*ones
        ntwos = -1*twos

        szd[:nva,:nva,:noa,noa:] = ones
        szd[:nva,:nva,noa:,:noa] = ones
        szd[nva:,nva:,:noa,noa:] = nones
        szd[nva:,nva:,noa:,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = nones
        szd[nva:,:nva,:noa,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = ones
        szd[nva:,:nva,:noa,:noa] = ones
        szd[nva:,nva:,:noa,:noa] = ntwos
        szd[:nva,:nva,noa:,noa:] = twos

        szd[:nva,:nva,:noa,noa:] = ones # BMW
        szd[:nva,:nva,noa:,:noa] = ones # BMW
        szd[nva:,nva:,:noa,noa:] = ones # BMW
        szd[nva:,nva:,noa:,:noa] = ones # BMW
        szd[:nva,nva:,:noa,:noa] = ones # BMW
        szd[nva:,:nva,:noa,:noa] = ones # BMW
        szd[:nva,nva:,:noa,:noa] = ones # BMW
        szd[nva:,:nva,:noa,:noa] = ones # BMW
        szd[nva:,nva:,:noa,:noa] = twos # BMW
        szd[:nva,:nva,noa:,noa:] = twos # BMW

        sz = numpy.zeros(( len(vs) ))
        s2 = numpy.zeros(( len(vs) ))
        
        header = "State      E(eV)       S2        Sz        |RS|      |RSab|      |RD|  Singlet?"
        for fock in range( nfock1 ):
            header += f"       |R{fock+1}|"
        for fock in range( nfock2 ):
            header += f"       |RS{fock+1}|"
        print(header)
        for i,x in enumerate(vs):
            rs,rd,rn,rsn = vector_to_amplitudes_Sn_U11((nm, nv, no), x, nfock1=nfock1 )
            rs1 = numpy.array(rsn[0])
            ds = numpy.sqrt(numpy.einsum('ai,ai->', rs, rs.conj()))
            rsa = rs[:nv//2, :no//2]
            rsb = rs[nv//2:, no//2:]
            rsab = rs[:nv//2, no//2:]
            ds = numpy.linalg.norm(rs)
            dsab = numpy.linalg.norm(rsa+rsb)
            dd = numpy.sqrt(0.25*numpy.einsum('abij,abij->', rd, rd.conj()))
            dn = []
            for fock in range( nfock1 ):
                rfock = numpy.array(rn[fock])
                dn.append( numpy.sqrt(numpy.einsum('I,I->', rfock.reshape(-1), rfock.conj().reshape(-1))) )
            ds1 = numpy.sqrt(numpy.einsum('Iai,Iai->',rs1, rs1.conj()))
            sz[i] = numpy.einsum('ai,ai->', szs, rs)
            s2[i] = numpy.einsum('ai,ai->', s2s, rs) # BMW
            for I in range(nm): # Loop over photon modes
                sz[i] += numpy.einsum('ai,ai->', szs, rs1[I])
                s2[i] += 0.5*numpy.einsum('ai,ai->', s2s, rs1[I])
            sz[i] += 0.25*numpy.einsum('abij,abij', szd, rd)
            s2[i] += 0.25/2*numpy.einsum('abij,abij', s2d, rd) # TODO -- CHECK THIS BMW

            singlet = (dsab > 1.e-2*ds)
            properties = "{:3d}: {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}  {:8s}".format(i+1,es[i]*27.2114,s2[i],sz[i],ds,dsab,dd,str(singlet))
            for fock in range( nfock1 ):
                properties += " {:10.3f}".format(dn[fock])
            properties += " {:10.3f}".format(ds1)
            print(properties)

        # calclate <S^2> (TODO)

        properties_out = ( sz, s2, ds, dd, dn, ds1 )

        return es, properties_out

    if calc_nac:
        nac = eom_cc_epccsd_n_sn_nac()  # TODO
        return es, nac
    return es




def eom_ip_epccsd_1_s1_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):

    rs, rd, rs1 = vector_to_amplitudes_ip(dims, vector)

    sigs, sigd, sigs1 = eom_ip_epccsd_1_s1_sigma_opt(rs, rd, rs1, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector_ip(dims, sigs, sigd, sigs1)

def eom_ip_epccsd_1_s1(model, options, amps, verbose=0):
    T1, T2, S1, U11 = amps
    nm, nv, no = U11.shape
    nroots = options["nroots"]
    conv_tol = options["conv_tol"]
    max_space = options["max_space"]
    max_cycle = options["max_cycle"]

    F = model.g_fock()
    I = model.g_aint()
    w = model.omega()
    g,h = model.gint()
    G,H = model.mfG()
    dtype = T1.dtype

    # preconditioner
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    ds = (-eo.copy())
    dd = (ev[:,None,None] - eo[None,:,None] - eo[None,None,:])
    ds1 = (w[:,None] - eo[None,:])
    diag = amplitudes_to_vector_ip((nm, nv, no), ds, dd, ds1)
    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    gidx = numpy.argsort(diag)

    # generate guess
    guess = []
    for i in range(nroots):
        RS = 1e-5*numpy.random.rand(no)/numpy.sqrt(no)
        RD = 1e-5*numpy.random.rand(nv, no, no)/numpy.sqrt(nv*no*no)
        RS1 = 1e-5*numpy.random.rand(nm, no)/numpy.sqrt(nm*no)
        Rnoise = amplitudes_to_vector_ip((nm, nv, no), RS, RD, RS1)
        Rguess = numpy.zeros((vec_size_ip((nm, nv, no))), dtype=dtype)
        Rguess[gidx[i]] = 1.0
        Rguess += Rnoise
        Rguess = Rguess/numpy.linalg.norm(Rguess)
        guess.append(Rguess)

    # matvec
    imds = IPImds()
    imds.build(amps, F, I, w, g, h, G, H)
    matvec = lambda vec: [eom_ip_epccsd_1_s1_matvec((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]
    #matvec = lambda vec: eom_ip_epccsd_1_s1_matvec((nm, nv, no), vec, amps, F, I, w, g, h, G, H, imds=imds)

    # davidson iterations
    eig = lib.davidson_nosym1
    #eigs = pbc_linalg.eigs
    conv, es, vs = eig(matvec, guess, precond,
                       tol=conv_tol, max_cycle=max_cycle,
                       max_space=max_space, nroots=nroots,verbose=verbose)
    #conv, es, vs = eigs(matvec, vec_size_ea((nm, nv, no)), nroots, Adiag=diag)
    #vs = numpy.transpose(vs,(1,0))

    # cleanup
    vs = [x/numpy.linalg.norm(x) for x in vs]

    for x in vs:
        cs = vector_to_amplitudes_ip((nm, nv, no), x)
        rs,rd,rs1 = cs
    return es

def eom_ea_epccsd_1_s1_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):

    rs, rd, rs1 = vector_to_amplitudes_ea(dims, vector)

    sigs, sigd, sigs1 = eom_ea_epccsd_1_s1_sigma_opt(rs, rd, rs1, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector_ea(dims, sigs, sigd, sigs1)

def eom_ea_epccsd_1_s1(model, options, amps, verbose=0):
    T1, T2, S1, U11 = amps
    nm, nv, no = U11.shape
    nroots = options["nroots"]
    conv_tol = options["conv_tol"]
    max_space = options["max_space"]
    max_cycle = options["max_cycle"]

    F = model.g_fock()
    I = model.g_aint()
    w = model.omega()
    g,h = model.gint()
    G,H = model.mfG()
    dtype = T1.dtype

    # preconditioner
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    ds = (ev.copy())
    dd = (ev[:,None,None] + ev[None,:,None] - eo[None,None,:])
    ds1 = (w[:,None] + ev[None,:])
    diag = amplitudes_to_vector_ea((nm, nv, no), ds, dd, ds1)
    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    gidx = numpy.argsort(diag)

    # generate guess
    guess = []
    for i in range(nroots):
        RS = 1e-5*numpy.random.rand(nv)/numpy.sqrt(nv)
        RD = 1e-5*numpy.random.rand(nv, nv, no)/numpy.sqrt(nv*nv*no)
        RS1 = 1e-5*numpy.random.rand(nm, nv)/numpy.sqrt(nm*nv)
        Rnoise = amplitudes_to_vector_ea((nm, nv, no), RS, RD, RS1)
        Rguess = numpy.zeros((vec_size_ea((nm, nv, no))),dtype=dtype)
        Rguess[gidx[i]] = 1.0
        Rguess += Rnoise
        Rguess = Rguess/numpy.linalg.norm(Rguess)
        guess.append(Rguess)

    # matvec
    imds = EAImds()
    imds.build(amps, F, I, w, g, h, G, H)
    matvec = lambda vec: [eom_ea_epccsd_1_s1_matvec((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]
    #matvec = lambda vec: eom_ea_epccsd_1_s1_matvec((nm, nv, no), vec, amps, F, I, w, g, h, G, H, imds=imds)

    # davidson iterations
    eig = lib.davidson_nosym1
    #eigs = pbc_linalg.eigs
    log = lib.logger.Logger(sys.stdout, 4)
    conv, es, vs = eig(matvec, guess, precond,
                       tol=conv_tol, max_cycle=max_cycle,
                       max_space=max_space, nroots=nroots,verbose=verbose)
    #conv, es, vs = eigs(matvec, vec_size_ea((nm, nv, no)), nroots, Adiag=diag)
    #vs = numpy.transpose(vs,(1,0))

    # cleanup
    vs = [x/numpy.linalg.norm(x) for x in vs]

    for x in vs:
        cs = vector_to_amplitudes_ea((nm, nv, no), x)
        rs,rd,rs1 = cs
    return es

def eom_ee_epccsd_1_s1_nac(model, options, amps, verbose=0):
    return None


def eom_ee_epccsd_1_s2_matvec(dims, vector, amps, F, I, w, g, h, G, H, imds=None):

    # TODO: ad rs2
    rs, rd, r1, rs1 = vector_to_amplitudes(dims, vector)

    sigs, sigd, sig1, sigs1 = eom_ee_epccsd_1_s2_sigma_opt(rs, rd, r1, rs1, amps, F, I, w, g, h, G, H, imds=imds)
    return amplitudes_to_vector(dims, sigs, sigd, sig1, sigs1)

def eom_ee_epccsd_1_s2(model, options, amps, verbose=0, analysis=False, calc_nac=False):
    T1, T2, S1, U11 = amps
    nm, nv, no = U11.shape
    nroots = options["nroots"]
    conv_tol = options["conv_tol"]
    max_space = options["max_space"]
    max_cycle = options["max_cycle"]

    F = model.g_fock()
    I = model.g_aint()
    w = model.omega()
    g,h = model.gint()
    G,H = model.mfG()

    # preconditioner
    eo = F.oo.diagonal()
    ev = F.vv.diagonal()
    d1 = w.copy()
    ds = (ev[:,None] - eo[None,:])
    dd = (ev[:,None,None,None] + ev[None,:,None,None] - eo[None,None,:,None] - eo[None,None,None,:])
    ds1 = (w[:,None,None] + ev[None,:,None] - eo[None,None,:])
    diag = amplitudes_to_vector((nm, nv, no), ds, dd, d1, ds1)
    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    gidx = numpy.argsort(diag)

    # generate guess
    guess = []
    for i in range(nroots):
        # noise first
        R1 = 1e-5*numpy.random.rand(nm)/numpy.sqrt(nm)
        RS = 1e-5*numpy.random.rand(nv, no)/numpy.sqrt(nv*no)
        RD = 1e-5*numpy.random.rand(nv, nv, no, no)/numpy.sqrt(no*no*nv*nv)
        RS1 = 1e-5*numpy.random.rand(nm, nv, no)/numpy.sqrt(nv*no*nm)
        Rnoise = amplitudes_to_vector((nm, nv, no), RS, RD, R1, RS1)
        Rguess = numpy.zeros(vec_size((nm, nv, no)))
        Rguess[gidx[i]] = 1.0
        Rguess += Rnoise
        Rguess = Rguess/numpy.linalg.norm(Rguess)
        guess.append(Rguess)

    # matvec
    imds = EEImds()
    imds.build(amps, F, I, w, g, h, G, H)
    matvec = lambda vec: [eom_ee_epccsd_1_s2_matvec((nm, nv, no), x, amps, F, I, w, g, h, G, H, imds=imds) for x in vec]

    # davidson iterations
    eig = lib.davidson_nosym1
    conv, es, vs = eig(matvec, guess, precond,
                       tol=conv_tol, max_cycle=max_cycle,
                       max_space=max_space, nroots=nroots,verbose=verbose)

    # cleanup
    vs = [x/numpy.linalg.norm(x) for x in vs]

    if analysis:
        szs = numpy.zeros((nv, no))
        szd = numpy.zeros((nv, nv, no, no))
        noa = no//2
        nva = nv//2
        ones = numpy.ones((nva,noa))
        nones = -1.0*numpy.ones((nva,noa))
        szs[:nva,noa:] = ones
        szs[nva:,:noa] = nones
        ones = numpy.ones((nva,nva,noa,noa))
        nones = -1.0*ones
        twos = 2.0*ones
        ntwos = -1*twos
        szd[:nva,:nva,:noa,noa:] = ones
        szd[:nva,:nva,noa:,:noa] = ones
        szd[nva:,nva:,:noa,noa:] = nones
        szd[nva:,nva:,noa:,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = nones
        szd[nva:,:nva,:noa,:noa] = nones
        szd[:nva,nva:,:noa,:noa] = ones
        szd[nva:,:nva,:noa,:noa] = ones
        szd[nva:,nva:,:noa,:noa] = ntwos
        szd[:nva,:nva,noa:,noa:] = twos
        print("State    Sz        |RS|       |RD|       |R1|       |RS1|")
        for i,x in enumerate(vs):
            cs = vector_to_amplitudes((nm, nv, no), x)
            rs,rd,r1,rs1 = cs
            ds = numpy.sqrt(numpy.einsum('ai,ai->', rs, rs.conj()))
            dd = numpy.sqrt(0.25*numpy.einsum('abij,abij->', rd, rd.conj()))
            d1 = numpy.sqrt(numpy.einsum('I,I->', r1, r1.conj()))
            ds1 = numpy.sqrt(numpy.einsum('Iai,Iai->',rs1,rs1.conj()))
            sz = numpy.einsum('ai,ai->', szs, rs)
            for I in range(nm): sz += numpy.einsum('ai,ai->', szs, rs1[I])
            sz += 0.25*numpy.einsum('abij,abij', szd, rd)
            print("{:3d}: {:10.3E} {:10.3E} {:10.3E} {:10.3E} {:10.3E}".format(i,sz,ds,dd,d1,ds1))

    if calc_nac:
        nac = eom_cc_epccsd_1_s2_nac()
        return es, nac
    return es


def eom_ee_epccsd_properties(model, options, amps, verbose=0):

    """
    calculate the properties of polaritonic states:
    e.g., oscillator strenght
    """

    return None

def eom_ee_epccsd_1_s1_grad(model, options, amps, verbose=0):
    """
    file:///Users/zhy/Documents/papers/endnote-download_manualimport/1.1877072.pdf
    
    p == \partial

    dE    \p E     \p E \p T    \p E   \p C
    -- = ------ +  ---- ----- + ---- * -----
    dR    \p R     \p T \p R    \p C   \p R
        
    (\p E/ \p R and \p E/ \p L vanish)

    \eqiv dE_1 + dE_2 + dE_3

    """
    
    #1)
      
    return None

def eom_ee_epccsd_1_s2_nac(model, options, amps, verbose=0):
    return None
