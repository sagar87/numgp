import jax.numpy as jnp
import numpyro.handlers as handlers
from jax import vmap
from jax.random import PRNGKey
from jax.scipy.linalg import cholesky
from numgp.math import kron_solve_lower, kron_solve_upper
from numgp.util import stabilize
from numpyro.infer import Predictive


class LatentKronConditional:
    __params__ = ["mean", "cond", "Ksx", "Kss"]

    def __init__(self, model, gp, rng_key: int = 353, num_samples=1):
        self.model = model
        self.gp = gp
        self.rng_key = rng_key
        self.num_samples = num_samples

    def _get_var_names(self, *args, **kwargs):

        with handlers.seed(rng_seed=PRNGKey(self.rng_key)):
            trace = handlers.trace(self.model).get_trace(*args, **kwargs)

        self.Kxx = {}

        for key in trace:
            if key in [f"{self.gp}_{param}" for param in self.__params__]:
                setattr(self, key.split("_")[-1], key)
            if key.startswith(f"{self.gp}_Kxx"):
                self.Kxx[int(key.split("_")[-1])] = key

    def _conditional(self, params, *args, **kwargs):
        delta = params[self.gp].squeeze() - params[self.mean].squeeze()

        chols = [
            cholesky(stabilize(params[Kxx].squeeze()), lower=True)
            for _, Kxx in sorted(self.Kxx.items())
        ]
        cholTs = [chol.T for chol in chols]

        Kss = params[self.Kss].squeeze()
        Ksx = params[self.Ksx].squeeze()
        Kxs = Ksx.T

        alpha = kron_solve_lower(chols, delta)
        alpha = kron_solve_upper(cholTs, alpha)

        mu = jnp.dot(Ksx, alpha).ravel() + params[self.cond].squeeze()
        A = kron_solve_lower(chols, Kxs)
        cov = stabilize(Kss - jnp.dot(A.T, A))

        return mu, cov

    def conditional_from_guide(self, guide, params, *args, **kwargs):

        self._get_var_names(*args, **kwargs)
        predictive = Predictive(
            self.model,
            guide=guide,
            params=params,
            num_samples=self.num_samples,
            return_sites=(
                self.gp,
                self.mean,
                self.cond,
                self.Kss,
                self.Ksx,
                *list(self.Kxx.values()),
            ),
        )

        self.cond_params = predictive(PRNGKey(self.rng_key), *args)
        mu, var = vmap(self._conditional)(self.cond_params)
        return mu, var
