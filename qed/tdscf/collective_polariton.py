import os, sys, time
import numpy
from pyscf import scf, tdscf, gto, lib

from wavefunction_analysis.utils import print_matrix
from wavefunction_analysis.utils.pyscf_parser import *
from qed.tdscf.ghf import FewLevel

import functools
# real-time printout
print = functools.partial(print, flush=True)

def print_energy_weight(energy, weight_p, weight_e, method, cavity_model):
    keyword = method+' '+cavity_model
    print_matrix(keyword+' collective polariton energy:', energy, 10)
    if isinstance(weight_p, numpy.ndarray):
        if weight_p.shape[0] == 1: weight_p = weight_p[0]
        print_matrix(keyword+' collective photon contribution:', weight_p, 10)
    if isinstance(weight_e, numpy.ndarray):
        print_matrix(keyword+' collective electron contribution:', weight_e, 10)


def print_energy_weight_dip_oscillator(energy, weight_p, trans_dip, f_oscillator, mag_dip, f_rotation):
    if weight_p.ndim == 2 and weight_p.shape[1] == 1: weight_p = weight_p[:,0]

    print('state     energy(au)  photon          x           y           z           f            mx          my          mz           r')
    for i, ei in enumerate(energy):
        dip, mag = trans_dip[i], mag_dip[i]
        print('%3d    %11.5f %9.4f  %11.4f %11.4f %11.4f %11.4f %13.4f %11.4f %11.4f %11.4f'
              % (i+1, energy[i], weight_p[i], dip[0], dip[1], dip[2], f_oscillator[i], mag[0], mag[1], mag[2], f_rotation[i]))
    print('')


def run_ab_initio_qed(qed_method, mf, td, cav_obj, cavity_model, key, amp0=None):
    qed_obj = qed_method(mf, td, cav_obj, key)
    energy = qed_obj.kernel(amp0=amp0)[0]
    if not qed_obj.converged.all():
        print('!!!!!WARNING qed_obj is not converged:', qed_obj.converged)
    weight_p = qed_obj.cav_obj.get_mns_weight(qed_obj.mn)
    weight_e = qed_obj.get_xys_weight()

    print_energy_weight(energy, weight_p.T, weight_e, 'ab_initio', cavity_model)

    trans_dip = qed_obj.trans_dip
    mag_dip = qed_obj.trans_mag_dip
    f_oscillator = qed_obj.oscillator_strength()
    f_rotation = qed_obj.rotation_strength()
    print_energy_weight_dip_oscillator(energy, weight_p, trans_dip, f_oscillator, mag_dip, f_rotation)

    if key.get('debug', 0) > 10:
        print_matrix('td-qed amplitudes', qed_obj.xy[0][0])

    return energy, weight_p


def collective_polariton(parameters, job_type=None):
    process_clock, perf_counter = time.process_time, time.perf_counter
    cpu0, wall0 = process_clock(), perf_counter()

    nfrag, charge, spin, atom = parameters.get(section_names[0])[:4]
    functional, basis, nroots, td_model, verbose, debug, scf_method \
                        = get_rem_info(parameters.get(section_names[1]))

    #if 'few_level' in job_type: nroots = 80
    #elif key.get('iguess', None) == 'qed': nroots = 0

    h = None
    mol, mf, etot, td = run_pyscf_dft_tddft(charge, spin, atom, basis, functional,
                                      td_model, nroots, nfrag, verbose, debug,
                                      h, scf_method)

    print('ground-state energy:', numpy.array(etot))
    if nroots > 0:
        final_print_energy(td, 'tddft', 10, 1)
        trans_dip, trans_mag_dip, _ = find_transition_dipole(td, nroots, nfrag)
        f_oscillator, f_rotation = find_oscillator_strength(td, nroots, nfrag)
        print_matrix('tddft oscillator strength', f_oscillator, 10)
        print_matrix('tddft rotation strength', f_rotation, 10)

    if 'polariton' not in parameters:
        return None, None

    key = get_photon_info(parameters.get(section_names[2]))

    if key.get('debug', 0) > 10:
        if type(td) is not list:
            td = [td]
        for n in range(len(td)):
            xys = []
            for (x, y) in td[n].xy:
                xys.append(x.ravel())
            print_matrix('tddft amplitudes', numpy.array(xys))

    if nroots > 0:
        if key.get('adjust_func', None):
            _, key['cavity_freq'] = justify_photon_info(td, nroots, key['resonance_state'], key['adjust_func'])

    qed_model, cavity_models = key['qed_model'], key['cavity_model']
    if not isinstance(cavity_models, list):
        cavity_models = [cavity_models]

    qed_method = getattr(qed, qed_model) # method calling function

    if nfrag > 1 or key.get('scale_coupling', 0) > 0:
        key['cavity_mode'] /= numpy.sqrt(nfrag/40) # scale coupling strength

    print('photon frequency:', key['cavity_freq'], 'strength:', key['cavity_mode'][:,0].T)

    #if key['qed_gs']:
    #    # gas-phase rdm1 as initial guess of the qed-hf
    #    dm0 = [None]*nfrag
    #    for n in range(nfrag):
    #        dm0[n] = mf[n].make_rdm1()
    #    # use qed-ks ground state reference
    #    mf1 = run_pyscf_dft(charge, spin, atom, basis, functional,
    #                        #td_model, nroots,
    #                        nfrag, verbose, #debug,
    #                        [key['qed_gs'], key['cavity_mode']], dm0)[1]
    #    # we cannot pass though mf1 since the response function is different
    #    # copy over the results
    #    for n in range(nfrag):
    #        print('n:', n, ' mf energy:', mf[n].e_tot, ' qed-mf energy:', mf1[n].e_tot)
    #        mf[n].mo_coeff = mf1[n].mo_coeff
    #        mf[n].mo_energy = mf1[n].mo_energy
    #        mf[n].e_tot = mf1[n].e_tot
    #    # another way is to change back the original get_jk function


    for cavity_model in cavity_models:
        qed_obj0 = None
        #if key.pop('iguess', False) == 'qed' and job_type is 'ab_initio': # individual qed states as initial guess
        #    print('start inidividual qed calculation')
        #    nstates = key['nstates']
        #    key['nstates'] = 2 # minimum polariton states
        #    qed_obj0, _ = run_pyscf_qed(mf, td, qed_model, cavity_model, key, nfrag)
        #    final_print_energy(qed_obj0, 'inidividual '+cavity_model+' polariton', 10, 1)
        #    key['nstates'] = nstates

        key['qed_obj0'] = qed_obj0

        cav_obj = getattr(qed, cavity_model)(mf, key)

        energy, weight_p = None, None

        if job_type == 'ab_initio':
            energy, weight_p = run_ab_initio_qed(qed_method, mf, td, cav_obj, cavity_model, key)

        elif job_type == 'few_level':
            qed_obj = FewLevel(td, cav_obj, key)
            qed_obj.cavity_model = cavity_model
            energy, vector, trans_dip, mag_dip = qed_obj.kernel()
            weight_p, weight_e = qed_obj.get_weights(vector)

            print_energy_weight(energy, weight_p, weight_e, 'few_level', cavity_model)

            f_oscillator = 2./3. * numpy.einsum('s,sx,sx->s', energy, trans_dip.conj(), trans_dip)
            f_rotation = numpy.einsum('sx,sx->s', trans_dip.conj(), mag_dip)
            print_energy_weight_dip_oscillator(energy, weight_p, trans_dip, f_oscillator, mag_dip, f_rotation)

        elif job_type == 'few_level2':
            amp0 = None
            few_nstate = key.get('few_nstate', [1, 2, 5, 10, 20, 40])
            for n in few_nstate:
                qed_obj = FewLevel(td, cav_obj, key)
                qed_obj.cavity_model = cavity_model

                qed_obj.save_amplitude = qed_obj.resonance_state
                if n<qed_obj.resonance_state: n = qed_obj.resonance_state
                if n>nroots: break

                energy, vector, trans_dip, mag_dip = qed_obj.kernel(nstates=n)
                if qed_obj.save_amplitude:
                    weight_p, weight_e = vector[:,-qed_obj.ng:].T**2, None
                    if qed_obj.ng == 1: weight_p = weight_p[0]
                    else: weight_p = weight_p[0] - weight_p[1]
                else:
                    weight_p, weight_e = qed_obj.get_weights(vector)

                print_energy_weight(energy, weight_p, weight_e, 'few_level states '+str(n), cavity_model)

                f_oscillator = 2./3. * numpy.einsum('s,sx,sx->s', energy, trans_dip.conj(), trans_dip)
                f_rotation = numpy.einsum('sx,sx->s', trans_dip.conj(), mag_dip)
                print_energy_weight_dip_oscillator(energy, weight_p, trans_dip, f_oscillator, mag_dip, f_rotation)

                if qed_obj.save_amplitude: amp0 = np.copy(vector)

            energy, weight_p = run_ab_initio_qed(qed_method, mf, td, cav_obj, cavity_model, key, amp0)

    cpu, wall = process_clock(), perf_counter()
    print('%s %s total wall time: %9.2f sec, and cpu time: %9.2f sec' % (sys.argv[0], sys.argv[1], wall-wall0, cpu-cpu0))
    return energy, weight_p


if __name__ == '__main__':
    print(sys.argv)

    cavity_model = 'rwa'
    coupling = 0.005
    func = None # average, fwhm_l, fwhm_p
    qed_gs = None
    job_type = 'few_level2' # 'ab_initio'

    qcint = sys.argv[1]

    xyz2num = {'x': 0, 'y': 1, 'z': 2}

    if len(sys.argv) > 2:
        cavity_model = sys.argv[2]
    if len(sys.argv) > 3:
        coupling = sys.argv[3]
        if len(coupling.split('-')) == 2:
            coupling = [xyz2num[coupling.split('-')[1]], float(coupling.split('-')[0])]
        else:
            coupling = [0, float(coupling)]
    if len(sys.argv) > 4:
        if sys.argv[4].lstrip('-').replace('.','',1).isdigit():
            func = float(sys.argv[4])
        else:
            func = sys.argv[4]
    if len(sys.argv) > 5:
        job_type = sys.argv[5]
    if len(sys.argv) > 6:
        qed_gs = sys.argv[6]
        #nroots = int(sys.argv[3])
        #qcout += '_'+str(nroots)

    qcout = '.'.join(qcint.split('.')[:-1])
    qcout += '_'+cavity_model
    qcout += '_'+str(coupling[1])
    if type(func) is float: qcout += '_{:.4f}'.format(func)
    elif type(func) is str: qcout += '_'+func
    #if job_type is not 'ab_initio': qcout += '_'+job_type
    qcout += '_'+job_type
    if qed_gs: qcout += '_'+qed_gs
    qcout += '.out'
    print(qcout)
    sys.stdout = open(qcout, 'w')

    parameters = parser(qcint)

    #if len(sys.argv) > 3:
    #    nroots = int(sys.argv[3])
    #    parameters['rem']['cis_n_roots'] = nroots

    if 'polariton' in parameters:
        if cavity_model == 'all':
            parameters['polariton']['cavity_model'] = ['jc', 'rwa', 'rabi', 'pf']
        else:
            parameters['polariton']['cavity_model'] = cavity_model
        parameters['polariton']['cavity_mode'][0] = 0.
        parameters['polariton']['cavity_mode'][coupling[0]] = coupling[1]
        if type(func) is float:
            parameters['polariton']['cavity_freq'] = func
        elif type(func) is str:
            parameters['polariton']['adjust_func'] = func
        parameters['polariton']['qed_gs'] = qed_gs

        #parameters['polariton']['has_k'] = False
        parameters['polariton']['solver_algorithm'] = 'direct'

    energy, weight = collective_polariton(parameters, job_type)
