import os, sys
from functools import reduce
import numpy
from pyscf      import gto
from pyscf      import lib
from pyscf.lib  import logger
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.scf  import cphf
from pyscf      import __config__

sys.path.append('../..')
import qed
from qed.cavity.rhf import Rabi

def get_mol_iatm_icoord_dx(mol, iatm=0, icoord=0, dx=0.0, unit="bohr"):
    new_mol        = gto.copy(mol)
    natm           = mol.natm
    unit_converter = None
    if unit.lower() == "bohr" or unit.lower() == "au" or unit.lower() == "a.u.":
        unit_converter = 1.0 
    else:
        raise RuntimeError("wrong unit!")
    assert unit_converter is not None

    tmp_atm = [[mol.atom_symbol(ia), mol.atom_coord(ia, unit=unit)[0], mol.atom_coord(ia, unit=unit)[1], mol.atom_coord(ia, unit=unit)[2]] for ia in range(natm)]
    tmp_atm[iatm][1+icoord] += dx * unit_converter

    new_mol.atom = tmp_atm
    new_mol.unit = unit
    new_mol.build()

    return new_mol

def grad_fdiff(mf0, cav_method, td_method, cavity_mode, cavity_freq, nroots, state, dx,
                                with_e0=True, with_nuc=True):

    mol0      = mf0.mol
    mf_method = mf0.__class__

    grad_e = numpy.zeros([mol0.natm, 3]) 

    for iatm in range(mol0.natm):
        for icoord in range(3):
            mol1 = get_mol_iatm_icoord_dx(mol0, iatm=iatm, icoord=icoord, dx=dx, unit="bohr")
            mf1  = mf_method(mol1)
            if hasattr(mf0, "xc"): 
                mf1.xc = mf0.xc
            mf1.kernel()

            cav_obj1 = cav_method(mf1, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
            td_obj1  = td_method(mf1, cav_obj=cav_obj1)
            td_obj1.nroots  = nroots
            td_obj1.verbose = 0 
            td_obj1.kernel()

            e1 = with_e0 * mf1.energy_elec()[0] + with_nuc * mf1.energy_nuc() + td_obj1.e[state-1]

            mol2 = get_mol_iatm_icoord_dx(mol0, iatm=iatm, icoord=icoord, dx=-dx, unit="bohr")
            mf2  = mf_method(mol2)
            if hasattr(mf0, "xc"): 
                mf2.xc = mf0.xc
            mf2.kernel()

            cav_obj2 = cav_method(mf2, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
            td_obj2  = td_method(mf2, cav_obj=cav_obj2)
            td_obj2.nroots  = nroots
            td_obj2.verbose = 0 
            td_obj2.kernel()

            e2 = with_e0 * mf2.energy_elec()[0] + with_nuc * mf2.energy_nuc() + td_obj2.e[state-1]
            print("iatm:", iatm, "icoord:", icoord, "state:", state, "energy:",  td_obj1.e[state-1], td_obj2.e[state-1])

            grad_e[iatm, icoord] = (e1 - e2)/2/dx

    return grad_e

table_head = r'''
\begin{table}
\caption{%s results for state #%d (photon population: %.3f) of %s with %s functional and %s basis.}
\begin{center} 
\begin{tabular}{lccclccc} 
\toprule
& \multicolumn{3}{c}{Analytical Gradient (au)} & & \multicolumn{3}{c}{Numerical Gradient (au)}\\
\cline{2-4} \cline{6-8}
Atom & $x$ & $y$ & $z$ & & $x$ & $y$ & $z$\\
\midrule
'''

table_body = r'''%s(%d) & % 12.6f  & % 12.6f  & % 12.6f & & % 12.6f & % 12.6f & % 12.6f\\'''

table_end = r'''
 \bottomrule
 \multicolumn{4}{l}{Max Error  = $%s$ au} &
 \multicolumn{4}{l}{Mean Error = $%s$ au}\\
 \bottomrule
\end{tabular}
\end{center} 
\end{table}
'''

def sci_notation(number, sig_fig=2):
    assert isinstance(number, float)
    assert abs(number) > 1e-16
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    coef, exp = ret_string.split("e")
    exp  = int(exp)

    return r"%s \times 10^{%d}"%(coef, exp)

def print_two_grad_table(g1, g2, molname, functional, basis, method, state, pop, mol = None, file = None):
    atmlst = range(mol.natm)
    assert file is not None

    with open(file, "a") as f:
        f.write(table_head % (method, state, pop, molname, functional.upper(), basis))
        for k, ia in enumerate(atmlst):
            f.write((table_body%(mol.atom_symbol(ia), ia+1, g1[k,0], g1[k,1], g1[k,2], g2[k,0], g2[k,1], g2[k,2])+"\n"))
        err_grad = g1 - g2
        err_max  = numpy.max(numpy.abs(err_grad))
        err_mean = numpy.linalg.norm(err_grad)/mol.natm/mol.natm
        err_max_str  = sci_notation(err_max, sig_fig=1)
        err_mean_str = sci_notation(err_mean, sig_fig=1)
        f.write(table_end%(err_max_str, err_mean_str))

