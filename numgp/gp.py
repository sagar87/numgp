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
    def __init__(self, X, y, mean_func=Zero, cov_func=Constant, noise_func=WhiteNoise):
        super().__init__(mean_func=mean_func, cov_func=cov_func)
        self.noise_func = noise_func
        self.X = X
        self.y = y

        self.cov_kwargs = [
            a for a in inspect.getfullargspec(cov_func).args if a != "self"
        ]
        self.noise_kwargs = [
            a for a in inspect.getfullargspec(noise_func).args if a != "self"
        ]
        self.mean_kwargs = [
            a for a in inspect.getfullargspec(mean_func).args if a != "self"
        ]

    def marginal_likelihood(
        self,
        mean_args=(),
        cov_args=(),
        noise_args=(),
        mean_kwargs={},
        cov_kwargs={},
        noise_kwargs={},
    ):
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

        mu = self.mean_func(*mean_args, **mean_kwargs)(self.X)
        Kxx = self.cov_func(*cov_args, **cov_kwargs)(self.X)
        Knx = self.noise_func(*noise_args, **noise_kwargs)(self.X)
        cov = Kxx + Knx

        return mu, cov

    def _build_conditional(
        self, Xnew, pred_noise, diag, X, y, noise, cov_total, mean_total
    ):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Knx = noise(X)
        rxx = y - mean_total(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.sum(tt.square(A), 0)
            if pred_noise:
                var += noise(Xnew, diag=True)
            return mu, var
        else:
            Kss = self.cov_func(Xnew)
            cov = Kss - tt.dot(tt.transpose(A), A)
            if pred_noise:
                cov += noise(Xnew)
            return mu, cov if pred_noise else stabilize(cov)

    def conditional(self, Xnew, posterior, **kwargs):
        def _predict(X, Y, X_test, *args):
            i, j, k = (
                len(self.mean_kwargs),
                len(self.cov_kwargs),
                len(self.noise_kwargs),
            )
            mean_args = args[:i]
            cov_args = args[i : i + j]
            noise_args = args[i + j : i + j + k]
            rng_key = args[i + j + k]

            mean_func = self.mean_func(*mean_args)
            cov_func = self.cov_func(*cov_args)
            noise_func = self.noise_func(*noise_args)

            # compute kernels between train and test data, etc.
            k_pp = cov_func(X_test, X_test) + noise_func(
                X_test
            )  # kernel(X_test, X_test, var, length, noise, include_noise=True)
            k_pX = cov_func(
                X_test, X
            )  # kernel(X_test, X, var, length, noise, include_noise=False)
            k_XX = cov_func(X, X) + noise_func(
                X
            )  # kernel(X, X, var, length, noise, include_noise=True)
            K_xx_inv = jnp.linalg.inv(k_XX)
            K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
            sigma_noise = jnp.sqrt(
                jnp.clip(jnp.diag(K), a_min=0.0)
            ) * jax.random.normal(rng_key, X_test.shape[:1])
            mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
            # we return both the mean function and a sample from the posterior predictive for the
            # given set of hyperparameters
            return mean, mean + sigma_noise

        mean_param = [posterior[k] for k in self.mean_kwargs]
        cov_param = [posterior[k] for k in self.cov_kwargs]
        noise_param = [posterior[k] for k in self.noise_kwargs]

        rng_key, rng_key_predict = random.split(random.PRNGKey(5))
        keys = random.split(rng_key_predict, cov_param[0].shape[0])
        self.vmap_args = [*mean_param, *cov_param, *noise_param, keys]

        self.means, self.covs = vmap(
            lambda *args: _predict(self.X, self.y, Xnew, *args)
        )(*self.vmap_args)

    def conditional2(self, Xnew, posterior):
        def _predict(X, y, Xnew, *args):
            i, j, k = (
                len(self.mean_kwargs),
                len(self.cov_kwargs),
                len(self.noise_kwargs),
            )
            mean_args = args[:i]
            cov_args = args[i : i + j]
            noise_args = args[i + j :]

            mean_func = self.mean_func(*mean_args)
            cov_func = self.cov_func(*cov_args)
            noise_func = self.noise_func(*noise_args)

            Kxx = cov_func(X) + noise_func(X)
            Kxs = cov_func(X, Xnew)
            Knx = noise_func(X)

            rxx = y - mean_func(X)
            L = cholesky(stabilize(Kxx) + Knx)
            A = solve_lower(L, Kxs)
            v = solve_lower(L, rxx)
            mu = mean_func(Xnew) + A.T @ v

            Kss = cov_func(Xnew)
            cov = Kss - A.T @ A
            return mu, cov

        mean_param = [posterior[k] for k in self.mean_kwargs]
        cov_param = [posterior[k] for k in self.cov_kwargs]
        noise_param = [posterior[k] for k in self.noise_kwargs]

        self.vmap_args = [*mean_param, *cov_param, *noise_param]
        self.means, self.covs = vmap(
            lambda *args: _predict(self.X, self.y, Xnew, *args)
        )(*self.vmap_args)


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
