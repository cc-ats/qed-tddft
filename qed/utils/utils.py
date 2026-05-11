import numpy as np
import itertools

def random_matrix(shape, mean=0., width=.1, distribution='uniform', sym=True,
                  has_diag=True, dtype=float, seed=None):
    r"""
    Generate a random matrix with specified properties.

    Args:
        shape (int or list): Size of the random array.
        mean (float): Mean value for the random numbers.
        width (float): Width (range) for the random numbers.
        distribution (str): Type of distribution to use ('uniform', 'gaussian', 'normal', 'standard').
        sym (bool): If True, generate a symmetric matrix. Default is True.
        has_diag (bool): If False, set diagonal elements to zero. Default is True.
        dtype: Data type of the matrix elements. Default is float.
        seed: Random seed for reproducibility. Default is None.

    Returns:
        np.ndarray: Generated random matrix.
    """
    if dtype == complex:
        real_part = random_matrix(n, mean, width, distribution, sym, has_diag,
                                  float, seed)
        imag_part = random_matrix(n, mean, width, distribution, sym, has_diag,
                                  float, seed)
        matrix = real_part + 1j * imag_part
        return matrix

    rng = np.random.default_rng(seed)

    if distribution == 'uniform':
        matrix = rng.random(shape)*width + mean
    elif distribution in {'gaussian', 'normal'}:
        matrix = np.random.normal(mean, width, n)

    if len(shape) == 2 and shape[0] == shape[1]:
        n = shape[0]
        if sym:
            matrix = .5 * (matrix + matrix.T)
        if has_diag is False: # remove diagonal elements
            mask = np.eye(n, dtype=bool)
            matrix[mask] = 0.

    return matrix


def ishermitian(keyword, matrix, digits=[2,13,8,'f'], debug=0):
    r"""Check if a matrix is Hermitian and optionally print its elements."""
    print(keyword+' is Hermitian? ', end='')
    print(np.allclose(matrix, matrix.conj().T))
    if debug > 0:
        n = matrix.shape[0]
        w, width, precision, notation = digits
        for (i, j) in itertools.product(range(n), range(n)):
            f0, f1 = matrix[i,j], matrix[j,i]
            print(f'{i:{w}{'d'}} ', end='')
            print(f'{j:{w}{'d'}}: ', end='')
            print(f'{f0.real:{width}.{precision}{notation}} ', end='')
            print(f'{f1.real:{width}.{precision}{notation}} ', end='')
            print(f'{f0.imag:{width}.{precision}{notation}} ', end='')
            print(f'{f0.imag:{width}.{precision}{notation}} ')
        print()


def swap_largest_to_diagonal(matrix):
    for i in range(matrix.shape[0]):
        idx = np.argmax(np.abs(matrix[i]))
        matrix[i,i], matrix[i, idx] = matrix[i, idx], matrix[i,i]

    return matrix


def collect_lists(fn, iterable, *args, upack=False):
    r"""
    Collect lists returned by a function into columns.

    Args:
        fn (callable): Function that takes an element of the iterable and
            returns a list of values.
        iterable (iterable): An iterable of elements to process with the function.
        *args: Additional arguments to pass to the function.
        upack (bool, optional): If True, the elements of the iterable are unpacked as arguments to the function. Default is False.

    Returns:
        list[list]: Collected columns of values returned by ``fn``.
    """
    cols = None
    for x in iterable:
        values = fn(*x, *args) if upack else fn(x, *args)
        if cols is None:
            cols = [[] for _ in range(len(values))]
        for i, v in enumerate(values):
            cols[i].append(v)
    return cols
