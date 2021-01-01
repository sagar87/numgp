import warnings

warnings.simplefilter("error")
import numpy as np
import numpyro as npy
import pytest


@pytest.fixture(scope="module")
def test_array():
    X = np.linspace(0, 1, 10)[:, None]
    return X


@pytest.fixture(scope="module")
def test_array_2d():
    X = np.linspace(0, 1, 10).reshape(5, 2)
    return X
