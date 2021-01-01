import jax.numpy as jnp
import numgp
import numpy as np
import numpy.testing as npt
import numpyro as npy


def test_zero_mean(test_array):
    zero_mean = numgp.mean.Zero()
    M = zero_mean(test_array)
    assert np.all(M == 0)
    assert M.shape == (10,)
    M = zero_mean(None)
    assert np.isnan(M)


def test_constant(test_array):
    const_mean = numgp.mean.Constant(6)
    M = const_mean(test_array)
    assert np.all(M == 6)
    assert M.shape == (10,)
    M = const_mean(None)
    assert np.isnan(M)


def test_linear_mean(test_array):
    linear_mean = numgp.mean.Linear(2, 0.5)
    M = linear_mean(test_array)
    npt.assert_allclose(M[1], 0.7222, atol=1e-3)
    assert M.shape == (10,)
    M = linear_mean(None)
    assert np.isnan(M)
