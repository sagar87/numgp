import numgp
import numpy as np
import numpy.testing as npt
import numpyro as npy
from jax.scipy.linalg import solve_triangular


def test_kronecker():
    np.random.seed(1)
    # Create random matrices
    [a, b, c] = [np.random.rand(3, 3 + i) for i in range(3)]

    custom = numgp.math.kronecker(a, b, c)  # Custom version
    nested = np.kron(a, np.kron(b, c))
    np.testing.assert_array_almost_equal(custom, nested)  # Standard nested version


def test_cartesian():
    np.random.seed(1)
    a = np.array([1, 2, 3])
    b = np.array([0, 2])
    c = np.array([5, 6])
    manual_cartesian = np.array(
        [
            [1, 0, 5],
            [1, 0, 6],
            [1, 2, 5],
            [1, 2, 6],
            [2, 0, 5],
            [2, 0, 6],
            [2, 2, 5],
            [2, 2, 6],
            [3, 0, 5],
            [3, 0, 6],
            [3, 2, 5],
            [3, 2, 6],
        ]
    )
    auto_cart = numgp.math.cartesian([a, b, c])
    np.testing.assert_array_almost_equal(manual_cartesian, auto_cart)


def test_kron_dot():
    np.random.seed(1)
    # Create random matrices
    Ks = [np.random.rand(3, 3) for i in range(3)]
    # Create random vector with correct shape
    tot_size = np.prod([k.shape[1] for k in Ks])
    x = np.random.rand(tot_size).reshape((tot_size, 1))
    # Construct entire kronecker product then multiply
    big = numgp.math.kronecker(*Ks)
    slow_ans = np.dot(big, x)
    # Use tricks to avoid construction of entire kronecker product
    fast_ans = numgp.math.kron_dot(Ks, x)
    np.testing.assert_array_almost_equal(slow_ans, fast_ans)


def test_kron_solve_lower():
    np.random.seed(1)
    # Create random matrices
    Ls = [np.tril(np.random.rand(3, 3)) for i in range(3)]
    # Create random vector with correct shape
    tot_size = np.prod([L.shape[1] for L in Ls])
    x = np.random.rand(tot_size).reshape((tot_size, 1))
    # Construct entire kronecker product then solve
    big = numgp.math.kronecker(*Ls)
    slow_ans = solve_triangular(big, x, lower=True)
    # Use tricks to avoid construction of entire kronecker product
    fast_ans = numgp.math.kron_solve_lower(Ls, x)
    np.testing.assert_allclose(slow_ans, fast_ans)
