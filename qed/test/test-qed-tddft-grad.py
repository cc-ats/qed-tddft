import sys
import numpy
from functools import reduce

from pyscf import scf, tdscf, gto, lib
from pyscf import grad
from pyscf.grad.rhf import grad_nuc

from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import cphf
from pyscf.tools.dump_mat import dump_rec
from pyscf import __config__

from finite_diff import grad_fdiff
sys.path.append("../..")
import qed


h2co = '''
H       -0.9450370725    -0.0000000000     1.1283908757
C       -0.0000000000     0.0000000000     0.5267587663
H        0.9450370725     0.0000000000     1.1283908757
O        0.0000000000    -0.0000000000    -0.6771667936
'''

table_head = r'''
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
'''

def print_grad(xx, mol = None, title = ""):
    log = lib.logger.Logger(mol.stdout, 5)
    print(title)
    grad.rhf._write(log, mol, xx, range(mol.natm))

def print_rec(xx, title = "", verbose=True):
    print(title)
    dump_rec(sys.stdout, xx)
    print("")

def sci_notation(number, sig_fig=2):
    assert isinstance(number, float)
    assert abs(number) > 1e-16
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    coef, exp = ret_string.split("e")
    exp  = int(exp)

    return r"%s \times 10^{%d}"%(coef, exp)

def print_two_grad_table(g1, g2, mol = None, file = None):
    atmlst = range(mol.natm)

    print(table_head)
    for k, ia in enumerate(atmlst):
        print((table_body%(mol.atom_symbol(ia), ia+1, g1[k,0], g1[k,1], g1[k,2], g2[k,0], g2[k,1], g2[k,2])))
    err_grad = g1 - g2
    err_max  = numpy.abs(numpy.max(err_grad))
    err_mean = numpy.linalg.norm(err_grad)/mol.natm/mol.natm
    err_max_str  = sci_notation(err_max, sig_fig=1)
    err_mean_str = sci_notation(err_mean, sig_fig=1)
    print(table_end%(err_max_str, err_mean_str))


if __name__ == "__main__":
    td_models = [tdscf.TDA, tdscf.TDDFT]
    qtd_models = [qed.TDA, qed.RPA]
    cav_models = [qed.JC, qed.Rabi, qed.RWA, qed.PF]

    a_td, b_qtd, c_cav = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    td_model  = td_models[a_td]
    qtd_model = qtd_models[b_qtd]
    cav_model = cav_models[c_cav]

    atom   = h2co
    basis  = "sto-3g"
    functional = str(sys.argv[4])
    nroots = 20

    mol = gto.M(
        atom    = atom,
        basis   = basis,
        spin    = 0,
        charge  = 0,
        verbose = 0
    )

    mf  = scf.RKS(mol)
    mf.xc = functional
    mf.kernel()


    td = td_model(mf)
    td.nroots  = nroots
    td.verbose = 0
    td.kernel()

    dipoles = numpy.zeros(nroots)
    for k in range(0, nroots):
    	dipoles[k] = numpy.linalg.norm(td.transition_dipole()[k])
    state = numpy.argsort(dipoles)[-1]
    #state = 4
    print("e0", td.e)
    print("dip0", td.transition_dipole())
    print('state=', state+1)

    if functional!='hf':
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=state)
        print('g1:\n', g1)


    cavity_mode = [td.transition_dipole()[state]*0.02]
    cavity_freq = [td.e[state]]

    cavity_freq = numpy.asarray(cavity_freq)
    cavity_mode = numpy.asarray(cavity_mode)
                                
    num_cav     = cavity_freq.size
    cavity_mode = cavity_mode.reshape(3, num_cav)

    print()
    print("############################")
    print("td_model  = ", qtd_model)
    print("cav_model = ", cav_model)
    cav_obj    = cav_model(mf, cavity_mode=cavity_mode, cavity_freq=cavity_freq)
    qed_td_obj = qtd_model(mf, cav_obj=cav_obj)

    qed_td_obj.nroots = nroots
    qed_td_obj.kernel()

    #grad = qed.Gradients(qed_td_obj)
    #Grad1 = grad.kernel(state=state)
    Grad1 = qed.grad.tdrks.Gradients(qed_td_obj).kernel(state=state)
    Grad2 = grad_fdiff(mf, cav_model, qtd_model, cavity_mode, cavity_freq, nroots, state, dx=1e-4)

    print_two_grad_table(Grad1, Grad2, mol=mol, file="test-%d.tex"%(state))

    if functional=='hf':
        Grad3 = qed.grad.tdrhf.Gradients(qed_td_obj).kernel(state=state)
        print_two_grad_table(Grad3, Grad2, mol=mol, file="test-%d.tex"%(state))

    
