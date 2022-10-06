import os, sys
import warnings
import numpy as np
from pyscf import scf, tdscf, gto, lib

import qed

try:
    from memory_profiler import profile
    from funcy import print_durations
except ModuleNotFoundError:
    pass

EV = 27.2116
section_names = ['molecule', 'rem', 'polariton']


def print_matrix(keyword, matrix, nwidth=0):
    if '\n' in keyword[-3:]: keyword = keyword[:-2]
    print(keyword)

    if len(matrix.shape)==1: # 1d array
        if nwidth==0: nwidth = 6
        for n in range(len(matrix)):
            print('%13.8f ' % matrix[n], end='')
            if (n+1)%nwidth==0: print('')
        print('\n')

    elif len(matrix.shape)==2: # 2d array
        nrow, ncol = matrix.shape
        if nwidth==0:
            nloop = 1
        else:
            nloop = ncol//nwidth
            if nloop*nwidth<ncol: nloop += 1

        for n in range(nloop):
            s0, s1 = n*nwidth, (n+1)*nwidth
            if s1>ncol or nwidth==0: s1 = ncol

            for r in range(nrow):
                for c in range(s0, s1):
                    print('%13.8f ' % matrix[r,c], end='')
                print('')
            print('')

    elif len(matrix.shape)==3: # 3d array
        for i in range(matrix.shape[0]):
            print_matrix(keyword+str(i+1), matrix[i], nwidth)
    else:
        warnings.warn('the matrix has higher dimension than this funciton can handle.')


def read_molecule(data):
    charge, spin = [], []
    coords, atmsym, xyz = [], [], []

    for line in data:
        info = line.split()
        if len(info) == 2:
            charge.append(int(info[0]))
            spin.append(int((int(info[1]) - 1) / 2))
            atmsym.append([])
            xyz.append([])
            coords.append('')
        elif len(info) == 4:
            coords[-1] += line + '\n'
            atmsym[-1].append(info[0])
            for x in range(3):
                xyz[-1].append(float(info[x+1]))

    nfrag = len(charge) - 1 if len(charge) > 1 else 1
    if len(charge) == 1:
        charge = charge[0]
        spin = spin[0]
        atmsym = atmsym[0]
        xyz = xyz[0]
        coords = coords[0]
    elif len(charge) > 1:
        # move the complex info to the end
        charge.append(charge.pop(0))
        spin.append(spin.pop(0))
        atmsym.append(atmsym.pop(0))
        xyz.append(xyz.pop(0))
        coords.append(coords.pop(0))
        # add complex coords
        for n in range(len(charge)-1):
            coords[-1] += coords[n]
            for i in range(len(atmsym[n])):
                atmsym[-1].append(atmsym[n][i])
                for x in range(3):
                    xyz[-1].append(xyz[n][i*3+x])

    #for n in range(len(charge)):
    #    print('charge and spin: ', charge[n], spin[n])
    #    print('coords:\n', coords[n])
    return nfrag, charge, spin, coords, atmsym, xyz


def read_keyword_block(data):
    rem_keys = {}
    for line in data:
        if '!' not in line:
            info = line.split()
        elif len(line.split('!')[0]) > 0:
            info = line.split('!')[0].split()
        else:
            info = []

        if len(info) == 2:
            rem_keys[info[0].lower()] = info[1]
        elif len(info) > 2:
            rem_keys[info[0].lower()] = [x for x in info[1:]]

    #print('rem_keys: ', rem_keys)
    return rem_keys


def parser(file_name):
    infile = open(file_name, 'r')
    lines = infile.read().split('$')
    #print('lines:\n', lines)

    parameters = {}
    for section in lines:
        data = section.split('\n')
        name = data[0].lower()
        #function = 'read_' + name
        #if function in globals():
        #    parameters[name] = eval('read_'+name)(data)
        if name == 'molecule':
            parameters[name] = read_molecule(data)
        else:
            parameters[name] = read_keyword_block(data)

    print('parameters:\n', parameters)
    return parameters


def build_single_molecule(charge, spin, atom, basis, verbose):
    mol = gto.M(
        atom    = atom,
        basis   = basis,
        spin    = spin,
        charge  = charge,
        verbose = verbose
    )

    return mol


#@print_durations()
def run_pyscf_dft(charge, spin, atom, basis, functional, nfrag=1, verbose=0):
    if nfrag == 1:
        mol = build_single_molecule(charge, spin, atom, basis, verbose)
        mf = scf.RKS(mol)
        mf.xc = functional
        mf.grids.prune = True
        mf.kernel()
    else:
        mol, mf = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            mol[n], mf[n] = run_pyscf_dft(charge[n], spin[n], atom[n],
                                          basis, functional, 1, verbose)

    return mol, mf


#@print_durations()
def run_pyscf_tddft(mf, td_model, nroots, nfrag=1, verbose=0, debug=0):
    if nfrag == 1:
        td = getattr(tdscf, td_model)(mf)
        td.max_cycle = 600
        td.max_space = 200

        td.kernel(nstates=nroots)
        try:
            td.converged.all()
            #print('TDDFT converged: ', td.converged)
            #print_matrix('Excited state energies (eV):\n', td.e * 27.2116, 6)
        except Warning:
            #print('the %d-th job for TDDFT is not converged.' % (n+1))
            print('the job for TDDFT is not converged.')
    else:
        td = [None]*nfrag
        for n in range(nfrag):
            td[n] = run_pyscf_tddft(mf[n], td_model, nroots, 1, verbose)

    if debug > 0:
        final_print_energy(td, nwidth=10)
        trans_dipole, argmax = find_transition_dipole(td, nroots, nfrag)

    return td


# mainly for parallel execute
#@print_durations()
def run_pyscf_dft_tddft(charge, spin, atom, basis, functional, td_model, nroots,
                        nfrag=1, verbose=0, debug=0):
    if nfrag == 1:
        mol, mf = run_pyscf_dft(charge, spin, atom, basis, functional, 1, verbose)
        td = run_pyscf_tddft(mf, td_model, nroots, 1, verbose)
    else:
        mol, mf, td = [None]*nfrag, [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            mol[n], mf[n] = run_pyscf_dft(charge[n], spin[n], atom[n],
                                          basis, functional, 1, verbose)
            td[n] = run_pyscf_tddft(mf[n], td_model, nroots, 1, verbose)

    if debug > 0:
        final_print_energy(td, nwidth=10)
        trans_dipole, argmax = find_transition_dipole(td, nroots, nfrag)

    return mol, mf, td


#@print_durations()
def run_pyscf_qed(mf, td, qed_model, cavity_model, cavity_mode, cavity_freq,
                  nfrag=1, uniform_field=True, external_field=None, verbose=0, debug=0):
    num_cav = cavity_freq.size
    if nfrag == 1:
        cav_obj = getattr(qed, cavity_model)(mf, cavity_mode=cavity_mode,
                                             cavity_freq=cavity_freq)
        cav_obj.uniform_field = uniform_field
        cav_obj.efield_file = external_field[0]
        cav_obj.field_strength = external_field[1]
        cav_obj.field_time = external_field[2]
        qed_td = getattr(qed, qed_model)(mf, cav_obj=cav_obj)

        qed_td.nroots = td.nroots + num_cav
        qed_td.kernel()
        try:
            qed_td.converged.all()
            #e_lp, e_up = qed_td.e[:2]
            #print('e_lp:', e_lp, '  e_up:', e_up)
            #print_matrix('qed state energies(H):\n', qed_td.e)
        except Warning:
            print('the job for qed-TDDFT is not converged.')
    else:
        qed_td, cav_obj = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            qed_td[n], cav_obj[n] = run_pyscf_qed(mf[n],
                                        td[n], qed_model,
                                        cavity_model, cavity_mode,
                                        cavity_freq, num_cav, 1, verbose)

        if debug > 0:
            final_print_energy(qed_td, 'qed-tddft', 10)

    return qed_td, cav_obj


def find_transition_dipole(td, nroots, nfrag=1):
    if nfrag == 1:
        trans_dipole = td.transition_dipole()
        argmax = np.unravel_index(np.argmax(np.abs(trans_dipole), axis=None),
                                  trans_dipole.shape)[0]
    else:
        trans_dipole, argmax = np.zeros((nfrag, nroots, 3)), np.zeros(nfrag, dtype=int)
        for n in range(nfrag):
            trans_dipole[n], argmax[n] = find_transition_dipole(td[n], 1)

        print_matrix('trans_dipole:', trans_dipole, 10)
        print('argmax of trans_dipole:\n', argmax+1)

    return trans_dipole, argmax


def final_print_energy(td, title='tddft', nwidth=6):
    if isinstance(td, list):
        if len(td) > 1:
            energy = []
            for n in range(len(td)):
                energy.append(td[n].e)
            energy = np.reshape(energy, (len(td), -1))
    else:
        energy = np.reshape(td.e, (1, -1))

    print_matrix(title+' energy:', energy, nwidth)
    return energy


def get_basis_info(mol):
    nbas = mol.nao_nr()
    nocc = mol.nelectron // 2 # assume closed-shell even electrons
    nvir = nbas - nocc
    nov  = nocc * nvir

    return [nbas, nocc, nvir, nov]


def get_rem_info(rem_keys):
    for key in rem_keys:
        print(key + ' = ', end=' ')
        key = rem_keys.get(key)
        print(key)

    functional = rem_keys.get('method')
    basis = rem_keys.get('basis')
    nroots = int(rem_keys.get('cis_n_roots'), 0)
    # 0 for rpa if it is ungiven. But it's working! Still None
    rpa = int(rem_keys.get('rpa', 0))
    td_model = 'TDDFT' if rpa == 2 else 'TDA'

    verbose = int(rem_keys.get('verbose', 1))
    debug   = int(rem_keys.get('debug', 0))

    return functional, basis, nroots, td_model, verbose, debug


def get_photon_info(photon_keys):
    # put default values first
    cavity_model = 'JC'
    cavity_mode, cavity_freq = None, None
    uniform_field, efield_file = True, 'efield'
    field_strength, field_time = 1, 0

    if 'cavity_model' in photon_keys: # support many models
        cavity_model = photon_keys.get('cavity_model')
        if isinstance(cavity_model, list):
            cavity_model = [x.upper() for x in cavity_model]
        else:
            cavity_model = [cavity_model.upper()]
        for i in range(len(cavity_model)):
            if cavity_model[i] == 'RABI': cavity_model[i] = 'Rabi'
    if 'cavity_mode' in photon_keys:
        cavity_mode = np.array([float(x) for x in photon_keys['cavity_mode']]).reshape(3, -1)
    if 'cavity_freq' in photon_keys:
        cavity_freq = np.array([float(photon_keys.get('cavity_freq'))])
    if 'uniform_field' in photon_keys:
        uniform_field = bool(int(photon_keys.get('uniform_field')))
    if 'efield_file' in photon_keys:
        efield_file = photon_keys.get('efield_file')
    if 'field_strength' in photon_keys:
        field_strength = float(photon_keys.get('field_strength'))
    if 'field_time' in photon_keys:
        field_time = float(photon_keys.get('field_time'))

    if cavity_mode is None:
        if uniform_field:
            raise TypeError('need cavity mode with uniform field')
        else:
            cavity_mode = np.ones((3, 1)) # artificial array


    rpa = int(photon_keys.get('rpa', 0))
    qed_model = 'RPA' if rpa == 2 else 'TDA'

    cavity_model = cavity_model

    print('qed_cavity_model: %s/%s' % (qed_model, cavity_model))
    print('cavity_mode: ', cavity_mode)
    print('cavity_freq: ', cavity_freq)

    return qed_model, cavity_model, cavity_mode, cavity_freq, uniform_field, [efield_file, field_strength, field_time]


def justify_photon_info(td, nroots, nfrag, func='average', nwidth=10):
    energy = final_print_energy(td, nwidth=nwidth)
    trans_dipole, argmax = find_transition_dipole(td, nroots, nfrag)
    argmax0 = argmax[0] if isinstance(argmax, list) else argmax
    print_matrix('max tddft energy:', energy[:, argmax0].T, nwidth=nwidth)
    freq = getattr(np, func)(energy[:, argmax0])
    print('applied photon energy is:', freq)

    return argmax0, np.asarray([freq])


#@print_durations()
def run_pyscf_final(parameters):
    nfrag, charge, spin, atom = parameters.get(section_names[0])[:4]
    functional, basis, nroots, td_model, verbose, debug = \
                get_rem_info(parameters.get(section_names[1]))

    mol, mf, td = run_pyscf_dft_tddft(charge, spin, atom, basis, functional,
                                      td_model, nroots, nfrag, verbose, debug)

    qed_td, cav_obj, qed_model, cavity_model = None, None, None, None
    if section_names[2] in parameters:
        mol0 = mol[0] if isinstance(mol, list) else mol
        nov = get_basis_info(mol0)[-1] # assume identical molecules
        if nroots > nov: # fix the nroots if necessary
            nroots = nov
            if isinstance(td, list):
                for n in range(nfrag): td[n].nroots = nov
            else: td.nroots = nov

        qed_model, cavity_model, cavity_mode, cavity_freq, uniform_field, external_field = \
                    get_photon_info(parameters.get(section_names[2]))

        if cavity_freq == None:
            target_state, cavity_freq = justify_photon_info(td, nroots, nfrag)

        n_model = len(cavity_model)
        qed_td, cav_obj = [None]*n_model, [None]*n_model
        for i in range(len(cavity_model)):
            qed_td[i], cav_obj[i] = run_pyscf_qed(mf, td, qed_model, cavity_model[i],
                                                  cavity_mode, cavity_freq, nfrag,
                                                  uniform_field, external_field,
                                                  verbose, debug)

    final_print_energy(td, nwidth=10)
    if qed_td:
        for i in range(n_model):
            final_print_energy(qed_td[i], cavity_model[i]+' qed-tddft', 10)

    return mol, mf, td, qed_td, cav_obj, qed_model, cavity_model


if __name__ == '__main__':
    parameters = parser('water.in')
    mol, mf, td, qed_td, cav_obj, qed_model, cavity_model = run_pyscf_final(parameters)
