from contextlib import suppress
from functools import lru_cache
from typing import Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
import numpyro as npy
import pandas as pd
from jax import lax, random
from jax.numpy import DeviceArray
from numpyro import optim
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO

Model = Callable[[DeviceArray], DeviceArray]
Guide = Callable[[DeviceArray], None]


class Handler(object):
    def __init__(self):
        pass

    def _select(self, which):
        assert which in [
            "prior",
            "posterior",
            "posterior_predictive",
        ], "Please select from 'prior', 'posterior' or 'posterior_predictive'."
        assert hasattr(self, which), f"NutsHandler did not compute the {which} yet."
        return getattr(self, which)

    def mean(self, param, which="posterior"):
        return self._select(which)[param].mean(0)

    @lru_cache(maxsize=128)
    def hpdi(self, param, which="posterior", *args, **kwargs):
        return hpdi(self._select(which)[param], *args, **kwargs)

    @lru_cache(maxsize=128)
    def quantiles(self, param, which="posterior", *args, **kwargs):
        return jnp.quantile(self._select(which)[param], jnp.array([0.05, 0.95]), axis=0)

    def qlower(self, param, which="posterior", *args, **kwargs):
        return self.quantiles(param, which=which, *args, **kwargs)[0]

    def qupper(self, param, which="posterior", *args, **kwargs):
        return self.quantiles(param, which=which, *args, **kwargs)[1]

    def lower(self, param, which="posterior", *args, **kwargs):
        return self.hpdi(param, which=which, *args, **kwargs)[0]

    def upper(self, param, which="posterior", *args, **kwargs):
        return self.hpdi(param, which=which, *args, **kwargs)[1]

    def ci(self, param, which="posterior"):
        return np.abs(self.mean(param, which) - self.hpdi(param, which))


class SVIHandler(Handler):
    def __init__(
        self,
        model: Model,
        guide: Guide,
        loss: Trace_ELBO = Trace_ELBO(num_particles=1),
        optimizer: optim.optimizers.optimizer = optim.Adam,
        lr: float = 0.001,
        rng_key: int = 254,
        num_epochs: int = 100000,
        num_samples: int = 5000,
        log_func=print,
        log_freq=0,
    ):
        self.model = model
        self.guide = guide
        self.loss = loss
        self.optimizer = optimizer(step_size=lr)
        self.rng_key = random.PRNGKey(rng_key)

        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss)
        self.init_state = None

        self.log_func = log_func
        self.log_freq = log_freq
        self.num_epochs = num_epochs
        self.num_samples = num_samples

        self.loss = None

    def _log(self, epoch, loss, n_digits=4):
        msg = f"epoch: {str(epoch).rjust(n_digits)} loss: {loss: 16.4f}"
        self.log_func(msg)

    def _fit(self, epochs, *args):
        return lax.scan(
            lambda state, i: self.svi.update(state, *args),
            self.init_state,
            jnp.arange(epochs),
        )

    def _update_state(self, state, loss):
        self.state = state
        self.init_state = state
        self.loss = loss if self.loss is None else jnp.concatenate([self.loss, loss])

    def fit(self, *args, **kwargs):
        num_epochs = kwargs.pop("num_epochs", self.num_epochs)
        log_freq = kwargs.pop("log_freq", self.log_freq)

        if self.init_state is None:
            self.init_state = self.svi.init(self.rng_key, *args)

        if log_freq <= 0:
            state, loss = self._fit(num_epochs, *args)
            self._update_state(state, loss)
        else:
            steps, rest = num_epochs // log_freq, num_epochs % log_freq

            for step in range(steps):
                state, loss = self._fit(log_freq, *args)
                self._log(log_freq * (step + 1), loss[-1])
                self._update_state(state, loss)

            if rest > 0:
                state, loss = self._fit(rest, *args)
                self._update_state(state, loss)

        self.params = self.svi.get_params(state)
        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=self.num_samples,
            **kwargs,
        )
        self.posterior = predictive(self.rng_key, *args)

    def get_posterior_predictive(self, *args, **kwargs):
        """kwargs -> Predictive, args -> predictive"""
        num_samples = kwargs.pop("num_samples", self.num_samples)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
            **kwargs,
        )
        self.posterior_predictive = predictive(self.rng_key, *args)


class NutsHandler(Handler):
    def __init__(
        self,
        model,
        posterior=None,
        num_warmup=2000,
        num_samples=10000,
        num_chains=1,
        key=0,
        *args,
        **kwargs,
    ):
        self.model = model
        self.rng_key, self.rng_key_ = random.split(random.PRNGKey(key))

        if posterior is not None:
            self.mcmc = posterior
            self.posterior = self.mcmc.get_samples()
        else:
            self.kernel = NUTS(model, **kwargs)
            self.mcmc = MCMC(
                self.kernel, num_warmup, num_samples, num_chains=num_chains
            )

    def _select(self, which):
        assert which in [
            "prior",
            "posterior",
            "posterior_predictive",
        ], "Please select from 'prior', 'posterior' or 'posterior_predictive'."
        assert hasattr(self, which), f"NutsHandler did not compute the {which} yet."
        return getattr(self, which)

    def get_prior(self, *args, **kwargs):
        predictive = Predictive(self.model, num_samples=self.mcmc.num_samples)
        self.prior = predictive(self.rng_key_, *args, **kwargs)

    def get_posterior_predictive(self, *args, **kwargs):
        predictive = Predictive(self.model, self.posterior, **kwargs)
        self.posterior_predictive = predictive(self.rng_key_, *args)

    def fit(self, *args, **kwargs):
        self.mcmc.run(self.rng_key_, *args, **kwargs)
        self.posterior = self.mcmc.get_samples()

    def summary(self, *args, **kwargs):
        self.mcmc.print_summary(*args, **kwargs)

    def dump(self, path):
        with open(path, "wb") as f:
            dill.dump(self.mcmc, f)

    @staticmethod
    def from_dump(model, path):
        with open(path, "rb") as f:
            posterior = dill.load(f)
        return NutsHandler(model, posterior=posterior)
