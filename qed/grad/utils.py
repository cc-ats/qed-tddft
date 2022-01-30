#TODO: Finish it

def _ovlp_grad_ao_dot_w_ao(ipovlp_ao, wao, atmlst=None, offsetdic=None):
    ''' 
        `ipovlp_ao` the `(3, nao, nao)` matrix $\langle \mu | \partial_x | \nu rangle $
        with the properties of gaussian functions, could be converted into overlap 
        gradient matrix as,
        ```
            ovlp_deriv = numpy.zeros([natm, 3, nao, nao])
            ipovlp     = mol.intor("int1e_ipovlp", comp=3)
            ipovlp     = ipovlp.reshape(3, nao, nao)

            for k, ia in enumerate(atmlst):
                shl0, shl1, p0, p1 = offsetdic[ia]
                ovlp_deriv[k, :, p0:p1, :] += ipovlp[:, p0:p1, :]  
                ovlp_deriv[k, :, :, p0:p1] += ipovlp[:, p0:p1, :].transpose(0,2,1)
        ```

        `wao` the `(nao, nao)` energy weighted density matrix in CIS or RPA gradient.

        return numpy.einsum('kxpq,pq->kx', ovlp_deriv[:, :, :, :], wao)*2
    '''
    pass

def _x_ao_dot_dse_grad_ao_dot_x_ao(irp_ao, x_ao, atmlst=None, offsetdic=None):
    ''' `irp_ao` the `(3, 3, nao, nao)` matrix $\langle \mu | y \partial_x | \nu rangle $
        with the properties of gaussian functions, could be converted into dipole 
        gradient matrix as,
        ```
            dip_ao_grad = numpy.zeros([natm, 3, 3, nao, nao])
        ```

        `wao` the `(nao, nao)` energy weighted density matrix in CIS or RPA gradient.

        return numpy.einsum('kxulmn,ul,mn->kx', dse_grad_ao, x_ao, x_ao)*2
    '''
    pass

def _dse_ao_dot_x_ao(dip_ao, x_ao, atmlst=None, offsetdic=None):
    pass