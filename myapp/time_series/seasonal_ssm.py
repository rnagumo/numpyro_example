"""Forecasting II: state space models

https://pyro.ai/examples/forecasting_ii.html
"""

import pathlib
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import diagnostics, infer
from numpyro.contrib.control_flow import scan


def model(
    covariates: jnp.ndarray,
    x: Optional[jnp.ndarray] = None,
    x_dim: int = 1,
    z_dim: int = 1,
    seasonality: int = 7,
) -> None:

    seq_len, batch, c_dim = covariates.shape
    if x is not None:
        x_dim = x.shape[-1]
    
    season_trans = jax.ops.index_add(jnp.eye(seasonality - 1, k=-1), 0, -1)
    season_var = numpyro.sample("season_var", dist.LogNormal(-5, 5))

    trend_trans = jnp.array([[1, 1], [0, 1]])
    trend_var = numpyro.sample(
        "trend_var", dist.LogNormal(jnp.array([-5, -5]), jnp.array([5, 5]))
    )

    weight_var = numpyro.sample(
        "weight_var", dist.LogNormal(-5 * jnp.ones((c_dim, x_dim)), 5 * jnp.ones((c_dim, x_dim)))
    )
    sigma = numpyro.sample("sigma", dist.LogNormal(-5 * np.ones(x_dim), 5 * np.ones(x_dim)))

    def transition_fn(
        carry: Tuple[jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:

        z_prev, s_prev, t_prev, w_prev = carry

        z = numpyro.sample("z", dist.Normal(z_prev, jnp.ones(z_dim)))

        s = jnp.matmul(s_prev, season_trans.T)
        s0 = numpyro.sample("s0", dist.Normal(s[:, 0], season_var))
        s = jax.ops.index_update(s, jax.ops.index[:, 0], s0)

        trend_mu = jnp.matmul(t_prev, trend_trans.T)
        trend = numpyro.sample("trend", dist.Normal(trend_mu, trend_var))

        weight = numpyro.sample("weight", dist.Normal(w_prev, weight_var))
        exogenous = jnp.matmul(covariates[t], weight)

        numpyro.sample("x", dist.Normal(z.sum(-1) + s0 + trend[:, 0] + exogenous, sigma))

        return (z, s, trend, weight), None

    z_init = jnp.zeros((batch, z_dim))
    s_init = jnp.zeros((batch, seasonality - 1))
    t_init = jnp.zeros((batch, 2))
    w_init = jnp.zeros((c_dim, x_dim))
    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (z_init, s_init, t_init, w_init), jnp.arange(seq_len))


def _load_data(
    num_seasons: int = 100, batch: int = 1, x_dim: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load sequential data with seasonality and trend."""

    t = jnp.sin(jnp.arange(0, 6 * jnp.pi, step=6 * jnp.pi / 700))[:, None, None]

    x = dist.Poisson(100).sample(random.PRNGKey(1234), (7 * num_seasons, batch, x_dim))
    x += jnp.array(np.random.rand(7 * num_seasons).cumsum(0)[:, None, None])
    x += (
        jnp.array(([50] * 5 + [1] * 2) * num_seasons)[:, None, None]
    )
    x = jnp.log1p(x)
    x += t * 2

    assert isinstance(x, jnp.ndarray)
    assert isinstance(t, jnp.ndarray)
    assert x.shape[0] == t.shape[0]
    assert x.shape[1] == t.shape[1]

    return x, t


def _save_results(
    x: jnp.ndarray,
    prior_samples: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
    num_train: int,
) -> None:

    root = pathlib.Path("./data/seasonal")
    root.mkdir(exist_ok=True)

    jnp.savez(root / "piror_samples.npz", **prior_samples)
    jnp.savez(root / "posterior_samples.npz", **posterior_samples)
    jnp.savez(root / "posterior_predictive.npz", **posterior_predictive)

    x_pred = posterior_predictive["x"]

    x_pred_trn = x_pred[:, :num_train]
    x_hpdi_trn = diagnostics.hpdi(x_pred_trn)
    t_train = np.arange(num_train)

    x_pred_tst = x_pred[:, num_train:]
    x_hpdi_tst = diagnostics.hpdi(x_pred_tst)
    num_test = x_pred_tst.shape[1]
    t_test = np.arange(num_train, num_train + num_test)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.figure(figsize=(12, 6))
    plt.plot(x.ravel(), label="ground truth", color=colors[0])

    plt.plot(t_train, x_pred_trn.mean(0)[:, 0], label="prediction", color=colors[1])
    plt.fill_between(
        t_train, x_hpdi_trn[0, :, 0, 0], x_hpdi_trn[1, :, 0, 0], alpha=0.3, color=colors[1]
    )

    plt.plot(t_test, x_pred_tst.mean(0)[:, 0], label="forecast", color=colors[2])
    plt.fill_between(
        t_test, x_hpdi_tst[0, :, 0, 0], x_hpdi_tst[1, :, 0, 0], alpha=0.3, color=colors[2]
    )

    plt.ylim(x.min() - 0.5, x.max() + 0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "prediction.png")
    plt.close()


def main() -> None:

    # Data
    x, t = _load_data()
    num_train = int(len(x) * 0.8)
    x_train = x[:num_train]
    t_train = t[:num_train]

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_infer, rng_key_posterior = random.split(rng_key, 4)

    # prior
    predictive = infer.Predictive(model, num_samples=10)
    prior_samples = predictive(rng_key_prior, t)

    # Inference
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 100, 100)
    mcmc.run(rng_key_infer, t_train, x_train)
    posterior_samples = mcmc.get_samples()

    # Posterior prediction
    predictive = infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        return_sites=["x", "s0", "z", "trend", "weight"],
    )
    posterior_predictive = predictive(rng_key_posterior, t)

    _save_results(x, prior_samples, posterior_samples, posterior_predictive, num_train)


if __name__ == "__main__":
    main()
