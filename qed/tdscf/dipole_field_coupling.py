import sys
import numpy as np
np.set_printoptions(precision=6)
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from pyscf import scf, gto, dft, df, lib
from pyscf import tdscf
from pyscf.dft import numint
from pyscf.tools import cubegen

from qed.tdscf.pyscf_parser import *

def creat_mesh_grids(mol, mf, grid_type=1, nxyz=[80, 80, 80, 0.1, 6.0]):
    if grid_type == 1:
        # default mesh grids and weights
        coords = mf.grids.coords
        weights = mf.grids.weights
        ngrids = weights.shape[0]
    elif grid_type == 2:
        cc = cubegen.Cube(mol, nx=nxyz[0], ny=nxyz[1], nz=nxyz[2],
                          resolution=nxyz[3], margin=nxyz[4])
        coords = cc.get_coords() # in Angstrom?
        ngrids = cc.get_ngrids()
        weights = ( np.ones(ngrids)
                * (cc.xs[1]-cc.xs[0])
                * (cc.ys[1]-cc.ys[0])
                * (cc.zs[1]-cc.zs[0])
                )
        print('nx: ', cc.nx, ' ny: ', cc.ny, ' nz: ', cc.nz)

    print('ngrids: ', ngrids)
    return ngrids, coords, weights


def get_electric_field(file_name='efield', strength=1, time=0, unit='nm', nline=2):
    from pyscf.data import nist
    # read in efield on cubic grids
    data = np.loadtxt(file_name+'.txt', skiprows=nline) # skip the header lines

    factor = 10 / lib.param.BOHR
    if unit == 'au': factor = 1.0 / lib.param.BOHR
    coords = data[:, :3] * factor

    efield = data[:, 3:8:2] # we can only handle real field now
    if time > 0: # calculate efield at time t
        efield = efield + data[:, 4:9:2] * 1j

        line = next(open(file_name+'.txt', 'r'))
        f_lambda = float(line.split('lambda=')[1].split('nm')[0])
        # E = hc/\lambda; 1fs = 41.341374575751 au
        #print(1e7/nist.HARTREE2WAVENUMBER/f_lambda, time /(nist.HBAR/nist.HARTREE2J*1e15) )
        factor = (1e7/nist.HARTREE2WAVENUMBER) / (nist.HBAR/nist.HARTREE2J*1e15)
        efield *= np.exp(1j*time/f_lambda*factor)
        efield = np.real(efield)

    # 5.14220674763 * 1e11
    factor = nist.HARTREE2J / nist.E_CHARGE / nist.BOHR_SI
    efield *= (strength / factor)
    #print_matrix('cubic grid efield:', efield0.T, 5)

    return coords, efield


def interpolate_electric_field(coords, coords0, efield0):
    # interpolate cubic efields on Lebedev coords
    efield = []
    for x in range(3):
        #interp = NearestNDInterpolator(coords0, efield0[:, x])
        interp = LinearNDInterpolator(coords0, efield0[:, x])
        efield.append(interp(coords))
    efield = np.reshape(efield, (3, -1)).T
    #print_matrix('Lebedev efield:', efield.T, 5)

    # temporarily add dimension of photon number for consistency
    efield = np.expand_dims(efield, axis=0)

    return efield


def build_ao(mol, ngrids, coords):
    # ao integral and its derivatives
    #ao_value = mol.eval_gto('GTOval_sph_deriv1', coords)
    # if we need to seperate batch
    #ao_value = np.zeros((4, ngrids, mol.nao_nr()))
    ao_value = np.zeros((ngrids, mol.nao_nr()))
    blksize = min(8000, ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao_value[ip0:ip1] = mol.eval_gto('GTOval_sph_deriv1', coords[ip0:ip1])[0]

    return ao_value


def dipole_dot_efield_on_grid(mol, mf, cav_obj):
    if cav_obj.uniform_field:
        dip_ao = mol.intor("int1e_r", comp=3)
        dip_dot_efield = lib.einsum('xmn,xp->pmn', dip_ao, cav_obj.cavity_mode)

    else:
        efield_file = cav_obj.efield_file
        coords0, efield0 = get_electric_field(efield_file, cav_obj.field_strength, cav_obj.field_time)

        # get Lebedev grids
        ngrids, coords, weights = creat_mesh_grids(mol, mf, grid_type=1)

        def extrema(m):
            a, b = np.min(m, axis=0), np.max(m, axis=0)
            return np.reshape([a, b], (2, -1))
        print_matrix('cubic grid range [xyz]:', extrema(coords0))
        print_matrix('Lebedev grid range [xyz]:', extrema(coords))

        efield = interpolate_electric_field(coords, coords0, efield0)

        re = np.einsum('ix,pix->pi', coords, efield)
        re = np.einsum('pi,i->pi', re, weights)

        ao_value = build_ao(mol, ngrids, coords)
        dip_dot_efield = np.einsum('im,in,pi->pmn', ao_value, ao_value, re)

    #print_matrix('dip_dot_efield', dip_dot_efield, 10)
    if np.any(np.isnan(dip_dot_efield)):
        raise ValueError('nan in interpolation. Need larger sample points.')

    return dip_dot_efield


def build_dip_ao_check2(mol, mf):
    strength = 0.01
    cavity_mode = np.ones((3, 1)) * strength

    cavity_size = [80, 80, 80, 1, 6]
    ngrids, coords0, _ = creat_mesh_grids(mol, mf, grid_type=2, nxyz=cavity_size)

    coords0 /= lib.param.BOHR
    efield0 = np.ones((ngrids, 3)) * strength

    # get Lebedev grids
    ngrids, coords, weights = creat_mesh_grids(mol, mf, grid_type=1)

    efield = interpolate_electric_field(coords, coords0, efield0)

    re = np.einsum('ix,pix->pi', coords, efield)
    re = np.einsum('pi,i->pi', re, weights)

    ao_value = build_ao(mol, ngrids, coords)
    dip_dot_efield = np.einsum('im,in,pi->pmn', ao_value, ao_value, re)

    dip_ao = mol.intor("int1e_r", comp=3)
    dip_dot_efield2 = lib.einsum('xmn,xp->pmn', dip_ao, cavity_mode)

    print_matrix('dip_dot_efield', dip_dot_efield, 10)
    print_matrix('dip_dot_efield2', dip_dot_efield2, 10)


if __name__ == '__main__':
    parameters = parser('water.in')
    nfrag, charge, spin, atom = parameters.get(section_names[0])[:4]
    functional, basis, nroots, td_model, verbose, debug = \
                get_rem_info(parameters.get(section_names[1]))

    mol, mf, td = run_pyscf_dft_tddft(charge, spin, atom, basis, functional,
                                      td_model, nroots, nfrag, verbose, debug)
    mol = mol[1]
    mf = mf[1]
    print(mol)

    build_dip_ao_check2(mol, mf)
