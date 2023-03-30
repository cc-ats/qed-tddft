#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Yang Gao <ygao@caltech.edu>

'''
FCI for Electron boson coupling

H_ep = g(k_i,k_j,q) a^\dagger_{ki} a_{kj} (b^{\dagger}_{ki-kj}+b_{kj-ki})
ki = 0,1,2,...,N-1
q = 1,2,...,N-1(labeled as 0,1,2,..,N-2)
'''

import numpy
from pyscf import lib
from pyscf.fci import cistring
import pyscf.fci
einsum = lib.einsum

def contract_all(t, u, g, hpp, ci0, nsite, nelec, nmode, nphonon, e_only=False, space='r', xi=None):
    '''
    Turn on e_only to shift the phonon vacuum
    space: r for real space(Hubbard Holstein Model), k for reciprocal space (SSH Model)
    '''
    ci1  = contract_1e           (  t, ci0, nsite, nelec, nmode, nphonon, e_only, space)
    if space == 'r':
        ci1 += contract_2e_rspace(  u, ci0, nsite, nelec, nmode, nphonon, e_only)
    if space == 'k':
        ci1 += contract_2e_kspace(  u, ci0, nsite, nelec, nmode, nphonon, e_only)
    if e_only:
        return ci1
    if space == 'r':
        ci1 += contract_ep_rspace(  g, ci0, nsite, nelec, nmode, nphonon)
    if space == 'k':
        ci1 += contract_ep_kspace(  g, ci0, nsite, nelec, nmode, nphonon)
    ci1 += contract_pp           (hpp, ci0, nsite, nelec, nmode, nphonon, xi)
    return ci1

def make_shape(nsite, nelec, nmode, nphonon, e_only=False):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)
    if e_only:
        return (na, nb)
    return (na,nb)+(nphonon+1,)*nmode

def find_idx(t, nsite, nelec):
    link_index = cistring.gen_linkstr_index(range(nsite), nelec)
    idx = t.diagonal().argsort()[:nelec]
    occlist = idx[idx.argsort()]
    map_mat = numpy.zeros([len(link_index), nelec], dtype='int32')
    for str0, tab in enumerate(link_index):
        lst = []
        for  a, i, str1, sign in tab:
            if str0 == str1:
                lst.append(i)
        map_mat[str0] = lst
    delta = numpy.linalg.norm(map_mat - occlist, axis=1)
    idx = numpy.where(delta==0)
    return idx


def contract_1e(h1e, fcivec, nsite, nelec, nmode, nphonon, e_only=False, space='r'):
    '''
    Hopping matrix is diagonal in reciprocal space
    Apply to general
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
    if space=='r': dtype='float64'
    if space=='k': dtype = 'complex128'
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * ci0[str0] * h1e[a,i]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * ci0[:,str0] * h1e[a,i]
    return fcinew.reshape(fcivec.shape)

def contract_2e_rspace(u, fcivec, nsite, nelec, nmode, nphonon, e_only=False):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(nsite), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(nsite), nelecb))
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    for i in range(nsite):
        maska = (strsa & (1<<i)) > 0
        maskb = (strsb & (1<<i)) > 0
        fcinew[maska[:,None]&maskb] += u * ci0[maska[:,None]&maskb]
    return fcinew.reshape(fcivec.shape)

def contract_2e_kspace(u, fcivec, nsite, nelec, nmode, nphonon, e_only=False):
    '''
    H_2 = U\sum_{ki,kj,ka}a^{\dagger}_{ka}a_{ki}A^{\dagger}_{kb}A_{ka+kb-ki}
    '''
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=ci0.dtype)
    for str0a, taba in enumerate(link_indexa):
        for a, i, str1a, signa in taba:
            for str0b, tabb in enumerate(link_indexb):
                for b, j, str1b, signb in tabb:
                    if numpy.mod(a+b-i-j, nsite)==0:
                        # ka + kb = ki + kj
                        # can be modified to handle general 2 body operator with decreased efficiency
                        fcinew[str1a,str1b] += signa * signb * ci0[str0a, str0b] * u / nsite
    return fcinew.reshape(fcivec.shape)

def slices_for(mode_id, nmode, nphonon, idxab=None):
    slices = [slice(None,None,None)] * (2+nmode)  # +2 for electron indiceshpp * t1
    slices[2+mode_id] = nphonon
    if idxab is not None:
        ia, ib = idxab
        if ia is not None:
            slices[0] = ia
        if ib is not None:
            slices[1] = ib
    return tuple(slices)

def slices_for_cre(mode_id, nmode, nphonon, idxab=None):
    return slices_for(mode_id, nmode, nphonon+1, idxab)

def slices_for_des(mode_id, nmode, nphonon, idxab=None):
    return slices_for(mode_id, nmode, nphonon-1, idxab)

def contract_ep_kspace(g, fcivec, nsite, nelec, nmode, nphonon):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nmode, nphonon)
    refa = cistring.make_strings(range(nsite),neleca)
    refb = cistring.make_strings(range(nsite),nelecb)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=numpy.complex128)
    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    phonon_des = numpy.sqrt(numpy.arange(nphonon))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            x, y = bin(refa[str0])[2:], bin(refa[str1])[2:]
            for mode_des in range(nsite-1):
                mode_cre = nsite - 2 - mode_des
                for ip in range(nphonon):
                    slices0 = slices_for_cre(mode_des, nmode, ip, idxab=(str0,None))
                    slices1 = slices_for(mode_des, nmode, ip, idxab=(str1,None))
                    fcinew[slices1] += g[a,i,mode_des] * phonon_cre[ip] * sign * ci0[slices0]

                    slices0 = slices_for(mode_cre, nmode, ip, idxab=(str0,None))
                    slices1 = slices_for_cre(mode_cre, nmode, ip, idxab=(str1,None))
                    fcinew[slices1] += g[a,i,mode_des] * phonon_cre[ip] * sign * ci0[slices0]

    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            for mode_des in range(nsite-1):
                mode_cre = nsite - 2 - mode_des
                for ip in range(nphonon):
                    slices0 = slices_for_cre(mode_des, nmode, ip, idxab=(None,str0))
                    slices1 = slices_for(mode_des, nmode, ip, idxab=(None,str1))
                    fcinew[slices1] += g[a,i,mode_des] * phonon_cre[ip] * sign * ci0[slices0]

                    slices0 = slices_for(mode_cre, nmode, ip, idxab=(None,str0))
                    slices1 = slices_for_cre(mode_cre, nmode, ip, idxab=(None,str1))
                    fcinew[slices1] += g[a,i,mode_des] * phonon_cre[ip] * sign * ci0[slices0]

    return fcinew.reshape(fcivec.shape)

def contract_ep_rspace(g, fcivec, nsite, nelec, nmode, nphonon):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nmode, nphonon)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)
    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            for mode_id in range(nmode):
                for ip in range(nphonon):
                    slices1 = slices_for_cre(mode_id, nmode, ip,idxab=(str1, None))
                    slices0 = slices_for(mode_id, nmode, ip, idxab=(str0, None))
                    fcinew[slices1] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices0]
                    #fcinew[slices0] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices1]
                    slices1 = slices_for(mode_id, nmode, ip, idxab=(str1, None))
                    slices0 = slices_for_cre(mode_id, nmode, ip, idxab=(str0, None))
                    fcinew[slices1] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            for mode_id in range(nmode):
                for ip in range(nphonon):
                    slices1 = slices_for_cre(mode_id, nmode, ip, idxab=(None,str1))
                    slices0 = slices_for(mode_id, nmode, ip, idxab=(None, str0))
                    fcinew[slices1] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices0]
                    #fcinew[slices0] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices1]
                    slices1 = slices_for(mode_id, nmode, ip, idxab=(None, str1))
                    slices0 = slices_for_cre(mode_id, nmode, ip, idxab=(None, str0))
                    fcinew[slices1] += g[a,i,mode_id]*phonon_cre[ip]*sign*ci0[slices0]

    return fcinew.reshape(fcivec.shape)
# Contract to one phonon creation operator
def cre_phonon(fcivec, nsite, nelec, nmode, nphonon, mode_id):
    cishape = make_shape(nsite, nelec, nmode, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=ci0.dtype)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for ip in range(nphonon):
        slices1 = slices_for_cre(mode_id, nmode, ip)
        slices0 = slices_for    (mode_id, nmode, ip)
        fcinew[slices1] += phonon_cre[ip] * ci0[slices0]
    return fcinew.reshape(fcivec.shape)

# Contract to one phonon annihilation operator
def des_phonon(fcivec, nsite, nelec, nmode, nphonon, mode_id):
    cishape = make_shape(nsite, nelec, nmode, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=ci0.dtype)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for ip in range(nphonon):
        slices1 = slices_for_cre(mode_id, nmode, ip)
        slices0 = slices_for    (mode_id, nmode, ip)
        fcinew[slices0] += phonon_cre[ip] * ci0[slices1]
    return fcinew.reshape(fcivec.shape)

# Phonon-phonon coupling
def contract_pp(hpp, fcivec, nsite, nelec, nmode, nphonon, xi=None):
    if xi is not None: 
        dtype= xi.dtype
    else:
        dtype=fcivec.dtype
    
    cishape = make_shape(nsite, nelec, nmode, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=dtype)
    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    t1 = numpy.zeros((nmode,)+cishape, dtype=dtype)
    for mode_id in range(nmode):
        for i in range(nphonon):
            slices1 = slices_for_cre(mode_id, nmode, i)
            slices0 = slices_for    (mode_id, nmode, i)
            t1[(mode_id,)+slices0] += ci0[slices1] * phonon_cre[i]     # annihilation
    t1 = einsum('w, wxy...->wxy...', hpp, t1)

    for mode_id in range(nmode):
        for i in range(nphonon):
            slices1 = slices_for_cre(mode_id, nmode, i)
            slices0 = slices_for    (mode_id, nmode, i)
            fcinew[slices1] += t1[(mode_id,)+slices0] * phonon_cre[i]  # creation
    if xi is not None:
        for mode_id in range(nmode):
            for i in range(nphonon):
                slices1 = slices_for_cre(mode_id, nmode, i)
                slices0 = slices_for(mode_id, nmode, i)
                fcinew[slices1] -= xi[mode_id] * phonon_cre[i] * ci0[slices0] * hpp[mode_id]
                fcinew[slices0] -= xi[mode_id] * phonon_cre[i] * ci0[slices1] * hpp[mode_id]

    return fcinew.reshape(fcivec.shape)



def make_hdiag(t, u, g, hpp, nsite, nelec, nmode, nphonon, e_only=False, space='r'):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    occslista = [tab[:neleca,0] for tab in link_indexa]
    occslistb = [tab[:nelecb,0] for tab in link_indexb]
    if space=='r': dtype='float64'
    if space=='k': dtype='complex128'
    nelec_tot = neleca + nelecb
    # electron part
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
    hdiag = numpy.zeros(cishape, dtype=dtype)
    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = t[aocc,aocc].sum() + t[bocc,bocc].sum()
            if space == 'k': e2 = 1.0 * u * neleca * nelecb / nsite
            if space == 'r': 
                vlst = numpy.hstack([aocc, bocc])
                unique, counts = numpy.unique(vlst, return_counts=True)
                counts -= 1
                netot = counts.sum()
                e2 = u * netot
            hdiag[ia,ib] = e1 + e2
    if e_only:
        return hdiag.ravel()
    for mode_id in range(nmode):
        for i in range(nphonon+1):
            slices0 = slices_for(mode_id, nmode, i)
            hdiag[slices0] += i * hpp[mode_id]#(i+1) #* hpp[mode_id]
            #hdiag[slices0] += i+1
    return hdiag.ravel()

def rkernel(t,u,g,hpp,nsite,nelec,nmode,nphonon,tol=1e-9,max_cycle=100,verbose=0,ecore=0,nroots=1, **kwargs):
    e_only = True
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
    space = 'r'
    dtype = 'float64'
    ci0 = numpy.zeros(cishape, dtype=dtype)
    ci0.__setitem__((0,0),1)
    # Add noise for initial guess, remove it if problematic
    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6

    def hop(c):
        hc = contract_all(t, u, g, hpp, c, nsite, nelec, nmode, nphonon, e_only, space)
        return hc.reshape(-1)
    hdiag = make_hdiag(t, u, g, hpp, nsite, nelec, nmode, nphonon, e_only, space)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,
                        nroots=nroots)
    dm0 = make_rdm1e(c, nsite, nelec)
    tau = einsum('ijI,ij->I', g, dm0) / hpp
    const = einsum('I,I->',hpp,tau**2)
    tnew = t -  2*einsum('ijI,I->ij', g, tau)
    e_only = False
    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    ci0 = numpy.zeros(cishape)
    idxa = find_idx(t,nsite,neleca)
    idxb = find_idx(t,nsite,nelecb)
    ci0.__setitem__((idxa,idxb) + (0,)*nmode, 1)
    #ci0.__setitem__((0,0) + (0,)*nmode, 1)
    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6
    def hop(c):
        hc = contract_all(tnew, u, g, hpp, c, nsite, nelec, nmode, nphonon, e_only, space, xi=tau)
        return hc.reshape(-1)
    hdiag = make_hdiag(tnew, u, g, hpp, nsite, nelec, nmode, nphonon, e_only, space)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,
                        **kwargs)
    e = e.real + const.real
    return e+ecore, c

def kernel(t,u,g,hpp,nsite,nelec,nmode,nphonon,space='r',tol=1e-9,max_cycle=100,verbose=0,ecore=0,nroots=1,**kwargs):
    if space=='r':
        return rkernel(t,u,g,hpp,nsite,nelec,nmode,nphonon,tol,max_cycle,verbose,ecore,nroots=nroots,**kwargs)
    if space=='k':
        return kkernel(t,u,g,hpp,nsite,nelec,nmode,nphonon,tol,max_cycle,verbose,ecore,nroots=nroots,**kwargs)

#def kernel(t, u, g, hpp, nsite, nelec, nmode, nphonon, space='r', tol=1e-9, max_cycle=100, verbose=0, ecore=0, **kwargs):
#    e_only = True
#    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
#    if space=='r': dtype = 'float64'
#    if space=='k': dtype = 'complex128'
#    ci0 = numpy.zeros(cishape, dtype=dtype)
#    ci0.__setitem__((0,0),1)
    # Add noise for initial guess, remove it if problematic
#    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
#    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6

#    def hop(c):
#        hc = contract_all(t, u, g, hpp, c, nsite, nelec, nmode, nphonon, e_only, space)
#        return hc.reshape(-1)
#    hdiag = make_hdiag(t, u, g, hpp, nsite, nelec, nmode, nphonon, e_only, space)
#    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
#    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
#                        tol=tol, max_cycle=max_cycle, verbose=verbose,
#                        nroots=1)
#    dm0 = make_rdm1e(c, nsite, nelec)
#    tau = einsum('ijI,ij->I', g, dm0) / hpp
#    const = einsum('I,I->',hpp,tau**2)
#    if space =='r': tnew = t -  2*einsum('ijI,I->ij', g, tau) 
#    if space =='k':
#        tnew = numpy.asarray(t, dtype=dtype)
#        for i in range(nmode):
#            j = nmode - 1 - i
#            tnew -= (tau[i] + tau[j]) * g[:,:,i]
#    e_only = False
#    cishape = make_shape(nsite, nelec, nmode, nphonon, e_only)
#    neleca, nelecb = nelec
#    ci0 = numpy.zeros(cishape)
#    idxa = find_idx(t,nsite,neleca)
#    idxb = find_idx(t,nsite,nelecb)
#    ci0.__setitem__((idxa,idxb) + (0,)*nmode, 1)
    #ci0.__setitem__((0,0) + (0,)*nmode, 1)
#    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
#    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6
#    def hop(c):
#        hc = contract_all(tnew, u, g, hpp, c, nsite, nelec, nmode, nphonon, e_only, space, xi=tau)
#        return hc.reshape(-1)
#    hdiag = make_hdiag(tnew, u, g, hpp, nsite, nelec, nmode, nphonon, e_only, space)
#    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
#    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
#                        tol=tol, max_cycle=max_cycle, verbose=verbose,
#                        **kwargs)
#    e = e.real + const.real
#    return e+ecore, c

def kkernel(t, u, g, hpp, nsite, nelec, nmode, nphonon, tol=1e-9, max_cycle=100, verbose=0, ecore=0, nroots=1,**kwargs):
    cishape = make_shape(nsite, nelec, nmode, nphonon)
    dtype = 'complex128'
    ci0 = numpy.zeros(cishape, dtype=dtype)
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    idxa = find_idx(t,nsite,neleca)
    idxb = find_idx(t,nsite,nelecb)
    ci0.__setitem__((idxa,idxb) + (0,)*nmode, 1)
    # Add noise for initial guess, remove it if problematic
    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6
    xi = None
    def hop(c):
        hc = contract_all(t, u, g, hpp, c, nsite, nelec, nmode, nphonon, space='k')
        return hc.reshape(-1)
    hdiag = make_hdiag(t, u, g, hpp, nsite, nelec, nmode, nphonon, space='k')
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,nroots=nroots,
                        **kwargs)
    e = e.real
    return e+ecore, c 
    
def make_rdm1e(fcivec, nsite, nelec):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)
    x = numpy.asarray(fcivec)
    rdm1 = numpy.zeros((nsite,nsite), dtype=fcivec.dtype)
    ci0 = fcivec.reshape(na,-1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.dot(ci0[str1],ci0[str0])

    ci0 = fcivec.reshape(na,nb,-1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.einsum('ax,ax->', ci0[:,str1],ci0[:,str0])
    return rdm1


if __name__ == '__main__':
    from fci_real import kernel as rker
    nsite = 4
    nmode = nsite - 1
    nelecs = nsite
    nphonon = 3
    nelec = (nelecs//2, nelecs-nelecs//2)
    tconst = 2.5 / 27.211385
    K = 21.0 / 27.211385 * 0.52917721092**2
    alpha = 4.1 / 27.211385 * 0.52917721092
    u = 4 * tconst
    M = 1349.14 / 27.211385 * 0.52917721092**2 / (2.418884326505*1e-2)**2


    kvec = numpy.arange(nsite) * 2.0 * numpy.pi / nsite
    ene = -2 * numpy.cos(kvec)
    t = numpy.zeros((nsite,nsite))
    numpy.fill_diagonal(t, ene)
    qvec = numpy.arange(1,nsite) * 2 * numpy.pi / nsite
    hpp = 2.0 * numpy.sqrt(K/M) * numpy.sin(qvec/2.0)

    gmat = numpy.zeros([nsite, nsite, nmode], dtype=numpy.complex128)
    for k1 in range(nsite):
        for k2 in range(nsite):
            if k1 == k2: continue
            q = numpy.mod(nsite+k1-k2,nsite) - 1
            k1vec = 2.0 * numpy.pi / nsite * k1
            k2vec = 2.0 * numpy.pi / nsite * k2
            gmat[k1,k2,q] = 1.0j * alpha * numpy.sqrt(2.0/nsite/M/hpp[q]) * (numpy.sin(k1vec) - numpy.sin(k2vec))
    lst = []
    elst= []
    for nphonon in range(3,10):
        ef,c = kernel(t, u, gmat, hpp, nsite, nelec, nmode, nphonon,space='k', tol=1e-10, verbose=5, nroots=1, max_cycle=500)
        print(nphonon, ef)
        #elst.append(ef)
    #print(lst)
    #print(elst)
