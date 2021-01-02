import inspect

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax import vmap
from jax.scipy.linalg import cholesky
from numgp.cov import Constant, Covariance, Kron, WhiteNoise
from numgp.math import cartesian, kron_dot, kron_solve_lower, kron_solve_upper
from numgp.mean import Zero
from numgp.util import infer_shape, solve_lower, solve_upper, stabilize


class Base:
    R"""
    Base class.
    """

    def __init__(self, mean_func=Zero(), cov_func=Constant(0.0)):
        self.mean_func = mean_func
        self.cov_func = cov_func

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) or not same_attrs:
            raise TypeError("Cannot add different GP types")
        mean_total = self.mean_func + other.mean_func
        cov_total = self.cov_func + other.cov_func
        return self.__class__(mean_total, cov_total)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError

    def predict(self, Xnew, point=None, given=None, diag=False):
        raise NotImplementedError


class Marginal(Base):
    R"""
    Marginal Gaussian process.
    The `gp.Marginal` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed.  For more
    information on the `prior` and `conditional` methods, see their docstrings.
    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.
    Examples
    --------
    .. code:: python
        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Marginal(cov_func=cov_func)
            # Place a GP prior over the function f.
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)
        ...
        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]
        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, name, mean_func=Zero(), cov_func=Constant(0.0)):
        super().__init__(mean_func, cov_func)
        self.name = name

    def _build_marginal_likelihood(self, X):
        mu = npy.deterministic(f"{self.name}_mean", self.mean_func(X))
        Kxx = npy.deterministic(f"{self.name}_Kxx", self.cov_func(X))
        Knx = npy.deterministic(f"{self.name}_Knx", self.noise(X))
        cov = Kxx + Knx
        return mu, cov

    def marginal_likelihood(self, X, y, noise, is_observed=True, **kwargs):
        R"""
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.
        This is integral over the product of the GP prior and a normal likelihood.
        .. math::
           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df
        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise.  Can also be a Covariance for
            non-white noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        if not isinstance(noise, Covariance):
            self.noise = WhiteNoise(noise)
        else:
            self.noise = noise

        mu, cov = self._build_marginal_likelihood(X)
        _ = npy.deterministic(f"{self.name}_y", y)

        if is_observed:
            return npy.sample(
                f"{self.name}",
                dist.MultivariateNormal(loc=mu, covariance_matrix=cov),
                obs=y,
            )
        else:
            shape = infer_shape(X, kwargs.pop("shape", None))
            return npy.sample(
                f"{self.name}", dist.MultivariateNormal(loc=mu, covariance_matrix=cov)
            )

    def _get_given_vals(self, given):
        if given is None:
            given = {}

        if "gp" in given:
            cov_total = given["gp"].cov_func
            mean_total = given["gp"].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ["X", "y", "noise"]):
            X, y, noise = given["X"], given["y"], given["noise"]
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        else:
            X, y, noise = self.X, self.y, self.noise
        return X, y, noise, cov_total, mean_total

    def _build_conditional(self, X=None, Xnew=None):
        # sets deterministic sites to sample from the condtional
        npy.deterministic(f"{self.name}_Kss", self.cov_func(Xnew))
        npy.deterministic(f"{self.name}_Kns", self.cov_func(Xnew))
        npy.deterministic(f"{self.name}_Ksx", self.cov_func(Xnew, X))
        npy.deterministic(f"{self.name}_cond", self.mean_func(Xnew))

    def conditional(self, X=None, Xnew=None):
        self._build_conditional(X, Xnew)
        return None


class LatentKron(Base):
    R"""
    Latent Gaussian process whose covariance is a tensor product kernel.
    The `gp.LatentKron` class is a direct implementation of a GP with a
    Kronecker structured covariance, without reference to any noise or
    specific likelihood.  The GP is constructed with the `prior` method,
    and the conditional GP over new input locations is constructed with
    the `conditional` method.  `conditional` and method.  For more
    information on these methods, see their docstrings.  This GP
    implementation can be used to model a Gaussian process whose inputs
    cover evenly spaced grids on more than one dimension.  `LatentKron`
    is relies on the `KroneckerNormal` distribution, see its docstring
    for more information.
    Parameters
    ----------
    cov_funcs: list of Covariance objects
        The covariance functions that compose the tensor (Kronecker) product.
        Defaults to [zero].
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.
    Examples
    --------
    .. code:: python
        # One dimensional column vectors of inputs
        X1 = np.linspace(0, 1, 10)[:, None]
        X2 = np.linspace(0, 2, 5)[:, None]
        Xs = [X1, X2]
        with pm.Model() as model:
            # Specify the covariance functions for each Xi
            cov_func1 = pm.gp.cov.ExpQuad(1, ls=0.1)  # Must accept X1 without error
            cov_func2 = pm.gp.cov.ExpQuad(1, ls=0.3)  # Must accept X2 without error
            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.LatentKron(cov_funcs=[cov_func1, cov_func2])
            # ...
        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        # Xnew need not be on a full grid
        Xnew1 = np.linspace(-1, 2, 10)[:, None]
        Xnew2 = np.linspace(0, 3, 10)[:, None]
        Xnew = np.concatenate((Xnew1, Xnew2), axis=1)  # Not full grid, works
        Xnew = pm.math.cartesian(Xnew1, Xnew2)  # Full grid, also works
        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, name, mean_func=Zero(), cov_funcs=(Constant(0.0))):
        try:
            self.cov_funcs = list(cov_funcs)
        except TypeError:
            self.cov_funcs = [cov_funcs]
        cov_func = Kron(self.cov_funcs)
        super().__init__(mean_func, cov_func)
        self.name = name

    def __add__(self, other):
        raise TypeError("Additive, Kronecker-structured processes not implemented")

    def _build_prior(self, Xs, **kwargs):
        self.N = np.prod([len(X) for X in Xs])
        mu = self.mean_func(cartesian(Xs))
        chols = []
        for i, (cov, X) in enumerate(zip(self.cov_funcs, Xs)):
            Kxx = npy.deterministic(f"{self.name}_Kxx_{i}", cov(X))
            chol = cholesky(stabilize(Kxx), lower=True)
            chols.append(chol)

        # remove reparameterization option
        v = npy.sample(
            f"{self.name}_rotated",
            dist.Normal(loc=jnp.zeros(self.N), scale=jnp.ones(self.N), **kwargs),
        )
        f = npy.deterministic(self.name, mu + (kron_dot(chols, v)).reshape(-1))
        return f

    def prior(self, Xs, **kwargs):
        """
        Returns the prior distribution evaluated over the input
        locations `Xs`.
        Parameters
        ----------
        name: string
            Name of the random variable
        Xs: list of array-like
            Function input values for each covariance function. Each entry
            must be passable to its respective covariance without error. The
            total covariance function is measured on the full grid
            `cartesian(*Xs)`.
        **kwargs
            Extra keyword arguments that are passed to the `KroneckerNormal`
            distribution constructor.
        """
        if len(Xs) != len(self.cov_funcs):
            raise ValueError("Must provide a covariance function for each X")
        f = self._build_prior(Xs, **kwargs)
        return f

    def _build_conditional(self, Xs=None, Xconds=None, **kwargs):
        # sets deterministic sites to sample from the condtional
        Xs = cartesian(Xs)
        Xconds = cartesian(Xconds)
        npy.deterministic(f"{self.name}_mean", self.mean_func(Xs))
        npy.deterministic(f"{self.name}_cond", self.mean_func(Xconds))
        npy.deterministic(f"{self.name}_Kss", self.cov_func(Xconds))
        npy.deterministic(f"{self.name}_Ksx", self.cov_func(Xconds, Xs))

    def conditional(self, Xs, Xnew, *args, **kwargs):
        self._build_conditional(Xs, Xnew)
        return None
