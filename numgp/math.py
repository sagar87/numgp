from functools import partial, reduce
from typing import List, Union

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def kronecker(*Ks):
    r"""Return the Kronecker product of arguments:
          :math:`K_1 \otimes K_2 \otimes ... \otimes K_D`
    Parameters
    ----------
    Ks : Iterable of 2D array-like
        Arrays of which to take the product.
    Returns
    -------
    np.ndarray :
        Block matrix Kroncker product of the argument matrices.
    """
    return reduce(jnp.kron, Ks)


def cartesian(arrays: Union[None, List[jnp.ndarray]]) -> Union[None, jnp.ndarray]:
    """Makes the Cartesian product of arrays.
    Parameters
    ----------
    arrays: 1D array-like
            1D arrays where earlier arrays loop more slowly than later ones
    """
    if arrays is None:
        return None

    N = len(arrays)
    arrays = [array.reshape(-1) for array in arrays]
    return jnp.stack(jnp.meshgrid(*arrays, indexing="ij"), -1).reshape(-1, N)


def kron_matrix_op(krons, m, op):
    r"""Apply op to krons and m in a way that reproduces ``op(kronecker(*krons), m)``
    Parameters
    -----------
    krons : list of square 2D array-like objects
        D square matrices :math:`[A_1, A_2, ..., A_D]` to be Kronecker'ed
        :math:`A = A_1 \otimes A_2 \otimes ... \otimes A_D`
        Product of column dimensions must be :math:`N`
    m : NxM array or 1D array (treated as Nx1)
        Object that krons act upon
    Returns
    -------
    numpy array
    """

    def flat_matrix_op(flat_mat, mat):
        Nmat = mat.shape[1]
        flat_shape = flat_mat.shape
        mat2 = flat_mat.reshape((Nmat, -1))
        return op(mat, mat2).T.reshape(flat_shape)

    def kron_vector_op(v):
        return reduce(flat_matrix_op, krons, v)

    if m.ndim == 1:
        m = m[:, None]  # Treat 1D array as Nx1 matrix
    if m.ndim != 2:  # Has not been tested otherwise
        raise ValueError(f"m must have ndim <= 2, not {m.ndim}")
    res = kron_vector_op(m)
    res_shape = res.shape
    return jnp.reshape(res, (res_shape[1], res_shape[0])).T


# # Define kronecker functions that work on 1D and 2D arrays
kron_dot = partial(kron_matrix_op, op=jnp.dot)
kron_solve_lower = partial(kron_matrix_op, op=partial(solve_triangular, lower=True))
kron_solve_upper = partial(kron_matrix_op, op=partial(solve_triangular, lower=False))
