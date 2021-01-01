import functools

import jax.numpy as jnp

__all__ = ["Zero"]


def check_input(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args[1] is None:
            return jnp.nan
        else:
            return func(*args, **kwargs)

    return wrapper


class Mean:
    R"""
    Base class for mean functions
    """

    def __call__(self, X):
        R"""
        Evaluate the mean function.

        Parameters
        ----------
        X: The training inputs to the mean function.
        """
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Prod(self, other)


class Zero(Mean):
    R"""
    Zero mean function for Gaussian process.

    """

    @check_input
    def __call__(self, X):
        return jnp.zeros(X.shape[0])


class Add(Mean):
    def __init__(self, first_mean, second_mean):
        Mean.__init__(self)
        self.m1 = first_mean
        self.m2 = second_mean

    @check_input
    def __call__(self, X):
        return self.m1(X) + self.m2(X)


class Prod(Mean):
    def __init__(self, first_mean, second_mean):
        Mean.__init__(self)
        self.m1 = first_mean
        self.m2 = second_mean

    def __call__(self, X):
        return self.m1(X) * self.m2(X)


class Constant(Mean):
    R"""
    Constant mean function for Gaussian process.

    Parameters
    ----------
    c: variable, array or integer
        Constant mean value
    """

    def __init__(self, c=0):
        Mean.__init__(self)
        self.c = c

    @check_input
    def __call__(self, X):
        return jnp.ones(X.shape[0]) * self.c


class Linear(Mean):
    R"""
    Linear mean function for Gaussian process.
    Parameters
    ----------
    coeffs: variables
        Linear coefficients
    intercept: variable, array or integer
        Intercept for linear function (Defaults to zero)
    """

    def __init__(self, coeffs, intercept=0):
        Mean.__init__(self)
        self.b = intercept
        self.A = coeffs

    @check_input
    def __call__(self, X):
        return jnp.squeeze(jnp.dot(X, self.A) + self.b)
