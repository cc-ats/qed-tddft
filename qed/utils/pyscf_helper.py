from dataclasses import dataclass, field
from typing import Any

import sys
import numpy as np
from qed.utils import print_matrix
from qed.utils import parser, collect_lists

from pyscf import scf, tdscf, gto

#import qed

section_names = ['molecule', 'rem', 'polariton']

@dataclass
class MoleculeInput:
    r"""Class to hold molecule input data."""
    charge: int = 0
    spin: int = 1
    atom: list[str] = None
    symbols: list[str] = None
    coords: np.ndarray = None
    unit: str = 'angstrom'


@dataclass
class SystemInput:
    r"""Class to hold system input data."""
    molecules: list[MoleculeInput]

    @property
    def nfrag(self):
        return len(self.molecules)


@dataclass
class SCFInput:
    r"""Class to hold SCF input data."""
    functional: str
    basis: str
    method: str = "RKS"
    grids_prune: bool = True
    unrestricted: bool = False

    h: Any = None # core Hamiltonian
    e_field: np.ndarray = None

    max_cycle: int = 50
    convergence: float = 1e-6

    jobtype: str = 'energy'
    max_memory: int = 60000
    verbose: int = 0
    debug: int = 0


@dataclass
class TDInput:
    method: str = 'TDA'
    cis_n_roots: int = 0
    cis_singlets: bool = False
    cis_triplets: bool = False
    rpa: bool = False

    max_cycle: int = 50
    max_space: int = 200
    verbose: int = 0


@dataclass
class QEDInput:
    cavity_freq: np.ndarray
    cavity_mode: np.ndarray
    cavity_model: str = 'JC'
    uniform_field: bool = True

    freq_window: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    solver_algorithm: str = 'davidson_qr'
    solver_conv_prop: str = 'norm'
    target_states: str = 'polariton'
    nstates: int = 4
    solver_conv_thresh: float = 1e-8

    resonance_state: int = None
    verbose: int = 0
    debug: int = 0


def setup_molecules(parameters, unit='angstrom'):
    r"""Setup MoleculeInput class from input parameters."""
    if isinstance(parameters, dict):
        parameters = parameters.get(section_names[0], parameters)
    nfrag, charge, spin, atom, symbols, coords = parameters

    if nfrag >= 1:
        molecules = [
            MoleculeInput(
                charge=charge[i],
                spin=spin[i],
                atom=atom[i],
                symbols=symbols[i],
                coords=coords[i],
                unit=unit,
            )
            for i in range(nfrag)
        ]
    else:
        molecules = [
            MoleculeInput(
                charge=charge,
                spin=spin,
                atom=atom,
                symbols=symbols,
                coords=coords,
                unit=unit,
            )
        ]

    return SystemInput(molecules=molecules)


def setup_scf_input(parameters):
    r"""Setup SCFInput class from input parameters."""
    parameters = parameters.get(section_names[1], parameters)
    if parameters.get('debug', 0) > 0:
        print('SCF parameters:')
        for key, value in parameters.items():
            print(f' {key} = {value}')

    return SCFInput(
        method=parameters.get('scf_method', 'RKS'),
        functional=parameters.get('method'),
        basis=parameters.get('basis'),
        grids_prune=parameters.get('grids_prune', True),
        unrestricted=parameters.get('unrestricted', False),
        h=parameters.get('h', None),
        e_field=parameters.get('e_field', None),
        max_cycle=parameters.get('max_cycle', 50),
        convergence=pow(10, -parameters.get('convergence', 6)),
        jobtype=get_jobtype(parameters),
        max_memory=parameters.get('max_memory', 60000),
        verbose=parameters.get('verbose', 0),
        debug=parameters.get('debug', 0),
    )

def setup_td_input(parameters):
    r"""Setup TDInput class from input parameters."""
    parameters = parameters.get(section_names[1], parameters)
    rpa = parameters.get('rpa', 1)
    td_model = 'TDDFT' if rpa == 2 else 'TDA'
    return TDInput(
        method=td_model,
        cis_n_roots=parameters.get('cis_n_roots', 0),
        cis_singlets=parameters.get('cis_singlets', False),
        cis_triplets=parameters.get('cis_triplets', False),
        rpa=rpa,
    )


def build_atom(atmsym, coords):
    r"""Build atom string for PySCF from atomic symbols and coordinates."""
    atom = ''
    for i in range(len(atmsym)):
        atom += str(atmsym[i]) + ' '
        for x in range(3):
            atom += str(coords[i,x]) + ' '
        atom += ';  '

    return atom


def build_molecule(molecule: MoleculeInput, basis, **kwargs):
    r"""
    Build molecule object for PySCF from MoleculeInput data and other parameters.

    Args:
        molecule (MoleculeInput): An instance of MoleculeInput containing the molecule data.
        basis (str): Basis set name for the molecule.
        **kwargs: Extra options used to build the molecule.
            Supported options include ``max_memory`` in MB and ``verbose``.

    Returns:
        mol (pyscf.gto.Mole): PySCF molecule object.
    """
    max_memory = kwargs.get('max_memory', 60000)
    verbose = kwargs.get('verbose', 0)

    mol = gto.M(
        atom       = molecule.atom,
        unit       = molecule.unit,
        spin       = molecule.spin,
        charge     = molecule.charge,
        basis      = basis,
        max_memory = max_memory,
        verbose    = verbose
    )

    # print(f'Basis number: {mol.nao_nr()} spherical and {mol.nao_cart()} cartesian.')
    return mol


def get_jobtype(parameters):
    r"""Determine the job type based on the parameters provided."""
    jobtype = 'scf'
    if 'cis_n_roots' in parameters:
        jobtype = 'td'

    if 'force' in parameters:
        jobtype += 'force'
    elif 'freq' in parameters:
        jobtype += 'hess'

    return jobtype


def get_frgm_idx(parameters):
    r"""Get the fragment indices from the parameters provided
    for embedding or other purposes."""
    frgm_idx = parameters.get(section_names[1])['impurity']
    if isinstance(frgm_idx, list):
        for i in range(len(frgm_idx)):
            at = frgm_idx[i].split('-')
            frgm_idx[i] = list(range(int(at[0])-1, int(at[1])))
    else:
        at = frgm_idx.split('-')
        frgm_idx = [list(range(int(at[0])-1, int(at[1])))] # need the outer bracket

    natm = len(np.ravel(parameters.get(section_names[0])[4]))
    assigned = np.concatenate(frgm_idx).tolist()
    if len(assigned) < natm:
        frgm_idx.append(list(set(range(natm)) - set(assigned)))

    #print('frgm_idx:', frgm_idx)
    return frgm_idx


def _get_center_of_mass(mol):
    mass = mol.atom_mass_list(isotope_avg=True)
    atom_coords = mol.atom_coords()
    mass_center = np.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    return mass_center


def get_center_of_mass(mol, nfrag=1):
    if isinstance(mol, list):
        mass_center = [None]*nfrag
        for n in range(nfrag):
            mass_center[n] = _get_center_of_mass(mol[n])
        return np.array(mass_center)
    else:
        return _get_center_of_mass(mol)


def _run_pyscf_dft(molecule, scf_input):
    r"""
    Run PySCF DFT calculation based on the parameters provided.

    Args:
        molecule (MoleculeInput): An instance of MoleculeInput containing the molecule data.
        scf_input (SCFInput): An instance of SCFInput containing the SCF calculation parameters.

    Returns:
        mol (pyscf.gto.Mole): PySCF molecule object.
        mf (pyscf.scf.SCF): PySCF mean-field object after SCF convergence.
        etot (float): Total energy of the system after SCF convergence.

"""
    mol = build_molecule(molecule, scf_input.basis,
                         max_memory=scf_input.max_memory,
                         verbose=scf_input.verbose)
    mf = getattr(scf, scf_input.method)(mol)
    if scf_input.h:
        #h = h + mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
        mf.get_hcore = lambda *args: scf_input.h
    mf.xc = scf_input.functional
    mf.grids.prune = scf_input.grids_prune
    mf.max_cycle = scf_input.max_cycle
    mf.conv_tol = scf_input.convergence

    # return total energy so that we don't need to calculate it again
    etot = mf.kernel()

    return mol, mf, etot


def run_pyscf_dft(molecules, scf_input):
    r"""Run PySCF DFT calculation based on the parameters provided."""
    if len(molecules) == 1:
        return _run_pyscf_dft(molecules[0], scf_input)
    return collect_lists(_run_pyscf_dft, molecules, scf_input)


def _run_pyscf_tddft(mf, td_input):
    r"""Run a single PySCF TDDFT calculation based on the mean-field object
    and TDDFT model provided."""
    def rotation_strength(td, trans_dip=None, trans_mag_dip=None):
        if trans_dip is None: trans_dip = td.transition_dipole()
        if trans_mag_dip is None:
            #trans_mag_dip = td.trans_mag_dip
            trans_mag_dip = td.transition_magnetic_dipole()

        f = np.einsum('sx,sx->s', trans_dip, trans_mag_dip)
        return f

    td = getattr(tdscf, td_input.method)(mf)
    td.max_cycle = td_input.max_cycle
    td.max_space = td_input.max_space

    td.kernel(nstates=td_input.cis_n_roots)
    if not td.converged.all():
        print('tddft is not converged:', td.converged)
    #try:
    #    td.converged.all()
    #    #print('TDDFT converged: ', td.converged)
    #    #print_matrix('Excited state energies (eV):\n', td.e * 27.2116, 6)
    #except Warning:
    #    #print('the %d-th job for TDDFT is not converged.' % (n+1))
    #    print('the job for TDDFT is not converged.')

    td.f_rotation = rotation_strength(td)

    if td_input.verbose >= 5:
        td.analyze(td_input.verbose)

    return td


def run_pyscf_tddft(mf, td_input):
    r"""Run PySCF TDDFT calculation based on the mean-field object and TDDFT model provided."""
    if isinstance(mf, list):
        return [_run_pyscf_tddft(_mf, td_model) for _mf in mf]
    else:
        return _run_pyscf_tddft(mf, td_input)


def _run_pyscf_dft_tddft(molecule, scf_input, td_input):
    r"""Run PySCF DFT and TDDFT calculations based on the parameters provided."""
    mol, mf, etot = _run_pyscf_dft(molecule, scf_input)
    td = _run_pyscf_tddft(mf, td_input)
    return mol, mf, etot, td


# mainly for parallel execute
def run_pyscf_dft_tddft(molecules, scf_input, td_input):
    if len(molecules) == 1:
        return _run_pyscf_dft_tddft(molecules[0], scf_input, td_input)
    return collect_lists(_run_pyscf_dft_tddft, molecules, scf_input, td_input)


def _run_pyscf_tdqed(mf, td, td_input, qed_input, key):
    cav_obj = getattr(qed, qed_input.cavity_model)(mf, key)
    qed_td = getattr(qed, td_input.method)(mf, td, cav_obj, key)
    qed_td.kernel()
    if not qed_td.converged.all():
        print('tdqed is not converged:', td.converged)
    #try:
    #    qed_td.converged.all()
    #    #e_lp, e_up = qed_td.e[:2]
    #    #print('e_lp:', e_lp, '  e_up:', e_up)
    #    #print_matrix('qed state energies(H):\n', qed_td.e)
    #except Warning:
    #    print('the job for qed-TDDFT is not converged.')

    return qed_td, cav_obj


def run_pyscf_tdqed(mf, td, td_input, qed_input, key):
    if isinstance(mf, list):
        return collect_lists(_run_pyscf_tdqed, zip(mf, td), td_input, qed_input,
                             key, unpack=True)
    else:
        return _run_pyscf_tdqed(mf, td, td_input, qed_input, key)


def _find_transition_dipole(td, nroots):
    trans_dipole = td.transition_dipole()
    trans_mag_dip = td.transition_magnetic_dipole()
    argmax = np.unravel_index(np.argmax(np.abs(trans_dipole), axis=None),
                              trans_dipole.shape)[0]
    print_matrix('trans_dipole:', trans_dipole, 10)
    print_matrix('trans_mag_dip:', trans_mag_dip, 10)
    return trans_dipole, trans_mag_dip, argmax


def find_transition_dipole(td, nroots, nfrag=1):
    if isinstance(td, list):
        trans_dipole, trans_mag_dip, argmax = [None]*nfrag, [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            trans_dipole[n], trans_mag_dip[n], argmax[n] = _find_transition_dipole(td[n], nroots)

        trans_dipole = np.reshape(trans_dipole, (nfrag, -1, 3))
        trans_mag_dip = np.reshape(trans_mag_dip, (nfrag, -1, 3))
        return trans_dipole, trans_mag_dip, argmax
    else:
        return _find_transition_dipole(td, nroots)


def find_oscillator_strength(td, nroots, nfrag=1):
    if isinstance(td, list):
        f_oscillator, f_rotation = [None]*nfrag, [None]*nfrag
        for n in range(nfrag):
            f_oscillator[n] = td[n].oscillator_strength()
            f_rotation[n] = td[n].f_rotation
        return np.array(f_oscillator), np.array(f_rotation)
    else:
        return td.oscillator_strength(), td.f_rotation


def final_print_energy(td, title='tddft', nwidth=6, iprint=0):
    if not isinstance(td, list): td = [td]

    if not isinstance(td[0].e, np.ndarray): return

    energy = []
    for n in range(len(td)):
        energy.append(td[n].e)
    energy = np.reshape(energy, (len(td), -1))

    if iprint > 0:
        print_matrix(title+' energy:', energy, nwidth)

    return energy


def get_basis_info(mol):
    nbas = mol.nao_nr()
    nocc = mol.nelectron // 2 # assume closed-shell even electrons
    nvir = nbas - nocc
    nov  = nocc * nvir

    return [nbas, nocc, nvir, nov]


def get_photon_info(photon_key):
    key = photon_key.copy()

    if isinstance(key.get('cavity_model', 'JC'), list): # support many models
        key['cavity_model'] = [x.capitalize() if x.upper()=='RABI' else x.upper() for x in key['cavity_model']]
    else:
        x = key.get('cavity_model')
        key['cavity_model'] = x.capitalize() if x.upper()=='RABI' else x.upper()
    if key.get('cavity_freq', None):
        key['cavity_freq'] = np.array([key.get('cavity_freq')])
    else:
        raise TypeError('need cavity frequency')
    key['uniform_field'] = bool(key.get('uniform_field', True))
    key.setdefault('efield_file', 'efield')
    #if key.get('cavity_mode', None):
    cavity_mode = key.get('cavity_mode', None)
    if isinstance(cavity_mode, list) or isinstance(cavity_mode, np.ndarray):
        key['cavity_mode'] = np.array(key['cavity_mode']).reshape(3, -1)
    else:
        if key['uniform_field']:
            raise TypeError('need cavity mode with uniform field')
        else:
            key['cavity_mode'] = np.ones((3, 1)) # artificial array

    key.setdefault('freq_window', [-0.05, 0.05])
    key['solver_algorithm'] = key.get('solver_algorithm', 'davidson_qr').lower()
    key['solver_conv_prop'] = key.get('solver_conv_prop', 'norm').lower()
    key['target_states'] = key.get('target_states', 'polariton').lower()
    key.setdefault('nstates', 4)
    #key.setdefault('solver_nvecs', 4)
    key['solver_conv_thresh'] = pow(10, -key.get('solver_conv_thresh', 8))
    key.setdefault('rpa', 0)
    key['qed_model'] = 'RPA' if key['rpa'] == 2 else 'TDA'
    key.setdefault('resonance_state', None)
    key.setdefault('verbose', 0)
    key.setdefault('debug', 0)

    key.setdefault('save_data', 0)
    key.setdefault('max_cycle', 50)
    key.setdefault('tolerance', 1e-9)
    key.setdefault('level_shift', 1e-2)

    print('qed_cavity_model: %s/%s' % (key['qed_model'], key['cavity_model']))
    print('cavity_mode: ', key['cavity_mode'])
    print('cavity_freq: ', key['cavity_freq'])

    return key


def justify_photon_info(td, nroots, nstate='max_dipole', func='average',
                        nwidth=10):
    energy = final_print_energy(td, nwidth=nwidth)
    if nstate == 'max_dipole':
        trans_dipole, trans_mag_dip, argmax = find_transition_dipole(td, nroots)
        argmax0 = argmax[0] if isinstance(argmax, np.ndarray) else argmax
        print_matrix('max tddft energy:', energy[:, argmax0].T, nwidth=nwidth)
    elif isinstance(nstate, int):
        argmax0 = nstate - 1

    if func == 'average':
        freq = getattr(np, func)(energy[:, argmax0])
        print('change applied photon energy to a more suitable one as:', freq)
    elif 'fwhm' in func: # full width at half maximum
        data = func.split('-')
        if len(data) > 1:
            func, factor = data[0], float(data[1])
        else: factor = 1.
        std = np.std(energy[:, argmax0])
        ave = np.average(energy[:, argmax0])
        if func[-2:] == '_m' or func[-2:] == '_l':
            freq = ave - 2. /factor * np.sqrt(2.*np.log(2.)) * std
        else:
            freq = ave + 2. /factor * np.sqrt(2.*np.log(2.)) * std
        print('change applied photon energy to a more suitable one as:', freq)


    return argmax0, np.asarray([freq])


def run_pyscf_final(parameters):
    mol_param = parameters.get(section_names[0], {})
    rem_param = parameters.get(section_names[1], {})
    unit = rem_param.get('unit', 'angstrom')
    molecular_system = setup_molecules(mol_param, unit)
    molecules = molecular_system.molecules
    scf_input = setup_scf_input(rem_param)
    td_input = setup_td_input(rem_param)


    results = {}

    jobtype = scf_input.jobtype
    if jobtype == 'scf':
        mol, mf, etot = run_pyscf_dft(molecules, scf_input)
        results['mol'] = mol
        results['mf']  = mf
        results['etot'] = etot
    elif 'td' in jobtype:
        mol, mf, etot, td = run_pyscf_dft_tddft(molecules, scf_input, td_input)
        results['mol'] = mol
        results['mf']  = mf
        results['etot'] = etot
        results['td']  = td

    print_matrix('scf energy:', results['etot'])
    if 'td' in jobtype:
        final_print_energy(results['td'], nwidth=10, iprint=1)

    if 'td_qed' in jobtype:
        mol0 = mol[0] if isinstance(mol, list) else mol
        nov = get_basis_info(mol0)[-1] # assume identical molecules
        if nroots > nov: # fix the nroots if necessary
            nroots = nov
            if isinstance(td, list):
                for n in range(len(td)): td[n].nroots = nov
            else: td.nroots = nov

        key = get_photon_info(parameters.get(section_names[2]))

        cavity_model = key['cavity_model']
        if not isinstance(cavity_model, list): cavity_model = [cavity_model]
        n_model = len(cavity_model)
        qed_td, cav_obj = [None]*n_model, [None]*n_model
        for i in range(len(cavity_model)):
            qed_td[i], cav_obj[i] = run_pyscf_tdqed(mf, td, key['qed_model'],
                                                    cavity_model[i], key, nfrag)

        for i in range(n_model):
            final_print_energy(qed_td[i], cavity_model[i]+' qed-tddft', 10, iprint=1)

        results['qed_td'] = qed_td
        results['cav_obj'] = cav_obj

    return results



if __name__ == '__main__':
    infile = 'water.in'
    if len(sys.argv) >= 2: infile = sys.argv[1]
    parameters = parser(infile)
    results = run_pyscf_final(parameters)
