from functools import partial

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

solve_lower = partial(solve_triangular, lower=True)
solve_upper = partial(solve_triangular, lower=False)


def infer_shape(X, n_points=None):
    if n_points is None:
        try:
            n_points = np.int(X.shape[0])
        except TypeError:
            raise TypeError("Cannot infer 'shape', provide as an argument")
    return n_points


def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * jnp.eye(K.shape[0])
