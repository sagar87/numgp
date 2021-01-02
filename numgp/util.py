from functools import partial

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

solve_lower = partial(solve_triangular, lower=True)
solve_upper = partial(solve_triangular, lower=False)


def infer_shape(X, n_points=None):
    if n_points is None:
        try:
            n_points = np.int(X.shape[0])
        except TypeError:
            raise TypeError("Cannot infer 'shape', provide as an argument")
    return n_points


def stabilize(K):
    """ adds small diagonal to a covariance matrix """
    return K + 1e-6 * jnp.eye(K.shape[0])


def plot_gp_dist(
    ax,
    samples: np.ndarray,
    x: np.ndarray,
    plot_samples=True,
    palette="Reds",
    fill_alpha=0.8,
    samples_alpha=0.1,
    fill_kwargs=None,
    samples_kwargs=None,
):
    """A helper function for plotting 1D GP posteriors from trace
        Parameters
    ----------
    ax: axes
        Matplotlib axes.
    samples: numpy.ndarray
        Array of S posterior predictive sample from a GP.
        Expected shape: (S, X)
    x: numpy.ndarray
        Grid of X values corresponding to the samples.
        Expected shape: (X,) or (X, 1), or (1, X)
    plot_samples: bool
        Plot the GP samples along with posterior (defaults True).
    palette: str
        Palette for coloring output (defaults to "Reds").
    fill_alpha: float
        Alpha value for the posterior interval fill (defaults to 0.8).
    samples_alpha: float
        Alpha value for the sample lines (defaults to 0.1).
    fill_kwargs: dict
        Additional arguments for posterior interval fill (fill_between).
    samples_kwargs: dict
        Additional keyword arguments for samples plot.
    Returns
    -------
    ax: Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if fill_kwargs is None:
        fill_kwargs = {}
    if samples_kwargs is None:
        samples_kwargs = {}
    if np.any(np.isnan(samples)):
        warnings.warn(
            "There are `nan` entries in the [samples] arguments. "
            "The plot will not contain a band!",
            UserWarning,
        )

    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 40)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    samples = samples.T
    x = x.flatten()
    for i, p in enumerate(percs[::-1]):
        upper = np.percentile(samples, p, axis=1)
        lower = np.percentile(samples, 100 - p, axis=1)
        color_val = colors[i]
        ax.fill_between(
            x, upper, lower, color=cmap(color_val), alpha=fill_alpha, **fill_kwargs
        )
    if plot_samples:
        # plot a few samples
        idx = np.random.randint(0, samples.shape[1], 30)
        ax.plot(
            x,
            samples[:, idx],
            color=cmap(0.9),
            lw=1,
            alpha=samples_alpha,
            **samples_kwargs
        )

    return ax
