import warnings
from functools import reduce
from operator import add, mul

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

# from jax.scipy.spatial.distance import pdist, cdist, squareform


class Covariance:
    r"""
    Base class for all kernels/covariance functions.

    Parameters
    ----------
    input_dim: integer
        The number of input dimensions, or columns of X (or Xs)
        the kernel will operate on.
    active_dims: List of integers
        Indicate which dimension or column of X the covariance
        function operates on.
    """

    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = jnp.arange(input_dim)
        else:
            self.active_dims = jnp.asarray(active_dims, np.int)

    def __call__(self, X, Xs=None, diag=False):
        r"""
        Evaluate the kernel/covariance function.

        Parameters
        ----------
        X: The training inputs to the kernel.
        Xs: The optional prediction set of inputs the kernel.
            If Xs is None, Xs = X.
        diag: bool
            Return only the diagonal of the covariance function.
            Default is False.
        """
        if X is None:
            return jnp.nan
        if diag:
            return self.diag(X)
        else:
            return self.full(X, Xs)

    def diag(self, X):
        raise NotImplementedError

    def full(self, X, Xs):
        raise NotImplementedError

    def _slice(self, X, Xs):
        if self.input_dim != X.shape[-1]:
            warnings.warn(
                f"Only {self.input_dim} column(s) out of {X.shape[-1]} are"
                " being used to compute the covariance function. If this"
                " is not intended, increase 'input_dim' parameter to"
                " the number of columns to use. Ignore otherwise.",
                UserWarning,
            )
        # print(self.active_dims)
        X = X[:, self.active_dims]
        if Xs is not None:
            Xs = Xs[:, self.active_dims]
        return X, Xs

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __array_wrap__(self, result):
        """
        Required to allow radd/rmul by numpy arrays.
        """
        result = np.squeeze(result)
        if len(result.shape) <= 1:
            result = result.reshape(1, 1)
        elif len(result.shape) > 2:
            raise ValueError(
                f"cannot combine a covariance function with array of shape {result.shape}"
            )
        r, c = result.shape
        A = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                A[i, j] = result[i, j].factor_list[1]
        if isinstance(result[0][0], Add):
            return result[0][0].factor_list[0] + A
        elif isinstance(result[0][0], Prod):
            return result[0][0].factor_list[0] * A
        else:
            raise RuntimeError


class Combination(Covariance):
    def __init__(self, factor_list):
        input_dim = max(
            [
                factor.input_dim
                for factor in factor_list
                if isinstance(factor, Covariance)
            ]
        )
        super().__init__(input_dim=input_dim)
        self.factor_list = []
        for factor in factor_list:
            if isinstance(factor, self.__class__):
                self.factor_list.extend(factor.factor_list)
            else:
                self.factor_list.append(factor)

    def merge_factors(self, X, Xs=None, diag=False):
        factor_list = []
        for factor in self.factor_list:
            # make sure diag=True is handled properly
            if isinstance(factor, Covariance):
                factor_list.append(factor(X, Xs, diag))
            elif isinstance(factor, np.ndarray):
                if np.ndim(factor) == 2 and diag:
                    factor_list.append(np.diag(factor))
                else:
                    factor_list.append(factor)
            elif isinstance(factor, jnp.DeviceArray):
                if factor.ndim == 2 and diag:
                    factor_list.append(jnp.diag(factor))
                else:
                    factor_list.append(factor)
            else:
                factor_list.append(factor)
        return factor_list


class Add(Combination):
    def __call__(self, X, Xs=None, diag=False):
        return reduce(add, self.merge_factors(X, Xs, diag))


class Prod(Combination):
    def __call__(self, X, Xs=None, diag=False):
        return reduce(mul, self.merge_factors(X, Xs, diag))


class Stationary(Covariance):
    r"""
    Base class for stationary kernels/covariance functions.

    Parameters
    ----------
    ls: Lengthscale.  If input_dim > 1, a list or array of scalars or PyMC3 random
    variables.  If input_dim == 1, a scalar or PyMC3 random variable.
    ls_inv: Inverse lengthscale.  1 / ls.  One of ls or ls_inv must be provided.
    """

    def __init__(self, input_dim, ls=None, ls_inv=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if (ls is None and ls_inv is None) or (ls is not None and ls_inv is not None):
            raise ValueError("Only one of 'ls' or 'ls_inv' must be provided")
        elif ls_inv is not None:
            if isinstance(ls_inv, (list, tuple)):
                self.ls = 1.0 / jnp.asarray(ls_inv)
            else:
                self.ls = 1.0 / ls_inv
        else:
            self.ls = jnp.asarray(ls)

    def cov_map(self, cov_func, xs, xs2):
        """Compute a covariance matrix from a covariance function and data points.
        Args:
        cov_func: callable function, maps pairs of data points to scalars.
        xs: array of data points, stacked along the leading dimension.
        Returns:
        A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
        """
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

    def square_dist(self, X, Xs):
        def _square_dist(x1, x2):
            return jnp.sum((x1 - x2) ** 2)

        X = X * (1.0 / self.ls)
        if Xs is None:
            sqd = self.cov_map(_square_dist, X, X)
        else:
            Xs = Xs * (1.0 / self.ls)
            sqd = self.cov_map(_square_dist, X, Xs)

        return sqd

    def euclidean_dist(self, X, Xs):
        r2 = self.square_dist(X, Xs)
        return jnp.sqrt(r2 + 1e-12)

    def diag(self, X):
        return jnp.ones(X.shape[0])

    def full(self, X, Xs=None):
        raise NotImplementedError


class Constant(Covariance):
    r"""
    Constant valued covariance function.

    .. math::

       k(x, x') = c
    """

    def __init__(self, c):
        super().__init__(1, None)
        self.c = c

    def diag(self, X):
        return self.c * jnp.ones(X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return self.c * jnp.ones((X.shape[0], X.shape[0]))
        else:
            return self.c * jnp.ones((X.shape[0], Xs.shape[0]))


class ExpQuad(Stationary):
    r"""
    The Exponentiated Quadratic kernel.  Also refered to as the Squared
    Exponential, or Radial Basis Function kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return jnp.exp(-0.5 * self.square_dist(X, Xs))


class WhiteNoise(Covariance):
    r"""
    White noise covariance function.

    .. math::

       k(x, x') = \sigma^2 \mathrm{I}
    """

    def __init__(self, sigma):
        super().__init__(1, None)
        self.sigma = sigma

    def diag(self, X):
        return self.sigma ** 2 * jnp.ones(X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return jnp.diag(self.diag(X))
        else:
            return jnp.zeros((X.shape[0], Xs.shape[0]))


class Matern52(Stationary):
    r"""
    The Matern kernel with nu = 5/2.
    .. math::
       k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} +
                   \frac{5(x-x')^2}{3\ell^2}\right)
                   \mathrm{exp}\left[ - \frac{\sqrt{5(x - x')^2}}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return (1.0 + jnp.sqrt(5.0) * r + 5.0 / 3.0 * (r ** 2)) * jnp.exp(
            -1.0 * jnp.sqrt(5.0) * r
        )


class Matern32(Stationary):
    r"""
    The Matern kernel with nu = 3/2.
    .. math::
       k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
                  \mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return (1.0 + jnp.sqrt(3.0) * r) * jnp.exp(-np.sqrt(3.0) * r)


class Matern12(Stationary):
    r"""
    The Matern kernel with nu = 1/2
    k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return jnp.exp(-r)


class Kron(Covariance):
    r"""Form a covariance object that is the kronecker product of other covariances.
    In contrast to standard multiplication, where each covariance is given the
    same inputs X and Xs, kronecker product covariances first split the inputs
    into their respective spaces (inferred from the input_dim of each object)
    before forming their product. Kronecker covariances have a larger
    input dimension than any of its factors since the inputs are the
    concatenated columns of its components.
    Factors must be covariances or their combinations, arrays will not work.
    Generally utilized by the `gp.MarginalKron` and gp.LatentKron`
    implementations.
    """

    def __init__(self, factor_list):
        self.input_dims = [factor.input_dim for factor in factor_list]
        input_dim = sum(self.input_dims)
        super().__init__(input_dim=input_dim)
        self.factor_list = factor_list

    def _split(self, X, Xs):
        indices = np.cumsum(self.input_dims)  # jnp.cumsum(jnp.asarray(self.input_dims))
        X_split = jnp.hsplit(X, indices)
        # X_split = jit(jnp.split, static_argnums=(1,))(X, indices)
        if Xs is not None:
            Xs_split = jnp.hsplit(Xs, indices)
        else:
            Xs_split = [None] * len(X_split)
        return X_split, Xs_split

    def __call__(self, X, Xs=None, diag=False):
        if X is None:
            return jnp.nan
        X_split, Xs_split = self._split(X, Xs)
        covs = [
            cov(x, xs, diag) for cov, x, xs in zip(self.factor_list, X_split, Xs_split)
        ]
        return reduce(mul, covs)


class Cosine(Stationary):
    r"""
    The Cosine kernel.
    .. math::
       k(x, x') = \mathrm{cos}\left( 2 \pi \frac{||x - x'||}{ \ell^2} \right)
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return jnp.cos(2.0 * np.pi * self.euclidean_dist(X, Xs))
