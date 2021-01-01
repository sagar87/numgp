import numgp
import numpy as np
import numpy.testing as npt
import numpyro as npy
import pytest

#
# Implements the test from the original pymc3 test suite
# TestExpSquad
# TestWhiteNoise
# TestConstant
# TestCovAdd
# TestCovProd
# TestCovSliceDim
#


def test_cov_add_symadd_cov(test_array):
    cov1 = numgp.cov.ExpQuad(1, 0.1)
    cov2 = numgp.cov.ExpQuad(1, 0.1)
    cov = cov1 + cov2
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_add_rightadd_scalar(test_array):
    a = 1
    cov = numgp.cov.ExpQuad(1, 0.1) + a
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_add_leftadd_scalar(test_array):
    a = 1
    cov = a + numgp.cov.ExpQuad(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_add_rightadd_matrix(test_array):
    M = 2 * np.ones((10, 10))
    cov = numgp.cov.ExpQuad(1, 0.1) + M
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_add_leftadd_matrixt(test_array):
    M = 2 * np.ones((10, 10))
    cov = M + numgp.cov.ExpQuad(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_add_leftprod_matrix():
    X = np.linspace(0, 1, 3)[:, None]
    M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
    cov = M + numgp.cov.ExpQuad(1, 0.1)
    cov_true = numgp.cov.ExpQuad(1, 0.1) + M
    K = cov(X)
    K_true = cov_true(X)
    assert np.allclose(K, K_true)


def test_cov_add_inv_rightadd():
    M = np.random.randn(2, 2, 2)
    with pytest.raises(ValueError, match=r"cannot combine"):
        cov = M + numgp.cov.ExpQuad(1, 1.0)


def test_cov_prod_symprod_cov(test_array):
    cov1 = numgp.cov.ExpQuad(1, 0.1)
    cov2 = numgp.cov.ExpQuad(1, 0.1)
    cov = cov1 * cov2
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 0.53940 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_prod_rightprod_scalar(test_array):
    a = 2
    cov = numgp.cov.ExpQuad(1, 0.1) * a
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_prod_leftprod_scalar(test_array):
    a = 2
    cov = a * numgp.cov.ExpQuad(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_prod_rightprod_matrix(test_array):
    M = 2 * np.ones((10, 10))
    cov = numgp.cov.ExpQuad(1, 0.1) * M
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_prod_leftprod_matrix():
    X = np.linspace(0, 1, 3)[:, None]
    M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
    cov = M * numgp.cov.ExpQuad(1, 0.1)
    cov_true = numgp.cov.ExpQuad(1, 0.1) * M
    K = cov(X)
    K_true = cov_true(X)
    assert np.allclose(K, K_true)


def test_cov_prod_multiops():
    X = np.linspace(0, 1, 3)[:, None]
    M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
    cov1 = (
        3
        + numgp.cov.ExpQuad(1, 0.1)
        + M * numgp.cov.ExpQuad(1, 0.1) * M * numgp.cov.ExpQuad(1, 0.1)
    )
    cov2 = (
        numgp.cov.ExpQuad(1, 0.1) * M * numgp.cov.ExpQuad(1, 0.1) * M
        + numgp.cov.ExpQuad(1, 0.1)
        + 3
    )
    K1 = cov1(X)
    K2 = cov2(X)
    assert np.allclose(K1, K2)
    # check diagonal
    K1d = cov1(X, diag=True)
    K2d = cov2(X, diag=True)
    npt.assert_allclose(np.diag(K1), K2d, atol=1e-5)
    npt.assert_allclose(np.diag(K2), K1d, atol=1e-5)


def test_cov_prod_inv_rightprod():
    M = np.random.randn(2, 2, 2)
    with pytest.raises(ValueError, match=r"cannot combine"):
        cov = M + numgp.cov.ExpQuad(1, 1.0)


def test_slice_dim_slice1():
    X = np.linspace(0, 1, 30).reshape(10, 3)
    cov = numgp.cov.ExpQuad(3, 0.1, active_dims=[0, 0, 1])
    K = cov(X)
    npt.assert_allclose(K[0, 1], 0.20084298, atol=1e-3)
    # check diagonal
    Kd = cov(X, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=2e-5)


def test_slice_dim_slice2():
    X = np.linspace(0, 1, 30).reshape(10, 3)
    cov = numgp.cov.ExpQuad(3, ls=[0.1, 0.1], active_dims=[1, 2])
    K = cov(X)
    npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
    # check diagonal
    Kd = cov(X, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_slice_dim_slice3():
    X = np.linspace(0, 1, 30).reshape(10, 3)
    cov = numgp.cov.ExpQuad(3, ls=np.array([0.1, 0.1]), active_dims=[1, 2])
    K = cov(X)
    npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
    # check diagonal
    Kd = cov(X, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_slice_dim_diffslice():
    X = np.linspace(0, 1, 30).reshape(10, 3)
    cov = numgp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 0, 0]) + numgp.cov.ExpQuad(
        3, ls=[0.1, 0.2, 0.3]
    )
    K = cov(X)
    npt.assert_allclose(K[0, 1], 0.683572, atol=1e-3)
    # check diagonal
    Kd = cov(X, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=2e-5)


def test_slice_dim_raises():
    lengthscales = 2.0
    with pytest.raises(ValueError):
        numgp.cov.ExpQuad(1, lengthscales, [True, False])
        numgp.cov.ExpQuad(2, lengthscales, [True])


def test_stability():
    X = np.random.uniform(low=320.0, high=400.0, size=[2000, 2])
    cov = numgp.cov.ExpQuad(2, 0.1)
    dists = cov.square_dist(X, X)
    assert not np.any(dists < 0)


def test_exp_quad_1d(test_array):
    cov = numgp.cov.ExpQuad(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)

    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)

    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_exp_quad_2d(test_array_2d):
    cov = numgp.cov.ExpQuad(2, 0.5)
    K = cov(test_array_2d)
    npt.assert_allclose(K[0, 1], 0.820754, atol=1e-3)
    # diagonal
    Kd = cov(test_array_2d, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_exp_quad_2dard(test_array_2d):
    cov = numgp.cov.ExpQuad(2, np.array([1, 2]))
    K = cov(test_array_2d)
    npt.assert_allclose(K[0, 1], 0.969607, atol=1e-3)
    # check diagonal
    Kd = cov(test_array_2d, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_exp_quad_inv_lengthscale(test_array):
    cov = numgp.cov.ExpQuad(1, ls_inv=10)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_white_noise(test_array):
    # with npy.handlers.seed(rng_seed=45):
    cov = numgp.cov.WhiteNoise(sigma=0.5)
    K = cov(test_array)

    npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
    npt.assert_allclose(K[0, 0], 0.5 ** 2, atol=1e-3)

    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
    npt.assert_allclose(K[0, 0], 0.0, atol=1e-3)


def test_constant_1d(test_array):

    cov = numgp.cov.Constant(2.5)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
    npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
    npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cov_kron_symprod_cov():
    X1 = np.linspace(0, 1, 10)[:, None]
    X2 = np.linspace(0, 1, 10)[:, None]
    X = numgp.math.cartesian([X1.reshape(-1), X2.reshape(-1)])

    cov1 = numgp.cov.ExpQuad(1, 0.1)
    cov2 = numgp.cov.ExpQuad(1, 0.1)
    cov = numgp.cov.Kron([cov1, cov2])

    K = cov(X)
    npt.assert_allclose(K[0, 1], 1 * 0.53940, atol=1e-3)
    npt.assert_allclose(K[0, 11], 0.53940 * 0.53940, atol=1e-3)
    # check diagonal
    Kd = cov(X, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_multiops():
    X1 = np.linspace(0, 1, 3)[:, None]
    X21 = np.linspace(0, 1, 5)[:, None]
    X22 = np.linspace(0, 1, 4)[:, None]
    X2 = numgp.math.cartesian([X21.reshape(-1), X22.reshape(-1)])
    X = numgp.math.cartesian([X1.reshape(-1), X21.reshape(-1), X22.reshape(-1)])

    cov1 = (
        3
        + numgp.cov.ExpQuad(1, 0.1)
        + numgp.cov.ExpQuad(1, 0.1) * numgp.cov.ExpQuad(1, 0.1)
    )
    cov2 = numgp.cov.ExpQuad(1, 0.1) * numgp.cov.ExpQuad(2, 0.1)
    cov = numgp.cov.Kron([cov1, cov2])
    K_true = numgp.math.kronecker(cov1(X1), cov2(X2))
    K = cov(X)
    npt.assert_allclose(K_true, K)


def test_matern52_1d(test_array):

    cov = numgp.cov.Matern52(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


def test_cosine_1d(test_array):
    cov = numgp.cov.Cosine(1, 0.1)
    K = cov(test_array)
    npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
    K = cov(test_array, test_array)
    npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
    # check diagonal
    Kd = cov(test_array, diag=True)
    npt.assert_allclose(np.diag(K), Kd, atol=1e-5)
