"""Univariate, heavy tailed time series.

ref)
https://pyro.ai/examples/forecasting_i.html

data)
https://www.bart.gov/about/reports/ridership
"""

import pathlib
from typing import Dict, Optional, Tuple

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
) -> None:

    if x is not None:
        x_dim = x.shape[-1]

    seq_len, batch, c_dim = covariates.shape
    weight = numpyro.sample(
        "weight", dist.Normal(np.zeros((c_dim, x_dim)), np.ones((c_dim, x_dim)) * 0.1)
    )
    bias = numpyro.sample("bias", dist.Normal(np.zeros(x_dim), np.ones(x_dim) * 10))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5 * np.ones(x_dim), 5 * np.ones(x_dim)))

    def transition_fn(
        carry: Tuple[jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:

        z_prev, *_ = carry
        z = numpyro.sample("z", dist.Normal(z_prev, jnp.ones(z_dim)))
        numpyro.sample("x", dist.Cauchy(z + jnp.matmul(covariates[t], weight) + bias, sigma))
        return (z,), None

    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (jnp.zeros((batch, z_dim)),), jnp.arange(seq_len))


def _load_data(
    num_seasons: int = 10, batch: int = 1, x_dim: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load sequential data with peaky noize.

    ref) http://docs.pyro.ai/en/stable/_modules/pyro/contrib/examples/bart.html

    Returns:
        Time series data with shape of `(seq_len, batch, data_dim)`.
    """

    rng_key = random.PRNGKey(1234)
    rng_key_0, rng_key_1 = random.split(rng_key, 2)
    x = dist.Poisson(100).sample(rng_key_0, (70 * num_seasons, batch, x_dim))
    x += jnp.array(([1] * 65 + [50] * 5) * num_seasons)[:, None, None] * random.normal(
        rng_key_1, (70 * num_seasons, batch, x_dim)
    )

    t = jnp.arange(len(x))[:, None, None]
    t = t.repeat(batch, axis=1)

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

    root = pathlib.Path("./data/heavy_tailed")
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

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure(figsize=(12, 6))
    plt.plot(x.ravel(), label="ground truth", color=colors[0])

    plt.plot(t_train, x_pred_trn.mean(0).ravel(), label="prediction", color=colors[1])
    plt.fill_between(
        t_train, x_hpdi_trn[0].ravel(), x_hpdi_trn[1].ravel(), alpha=0.3, color=colors[1]
    )

    plt.plot(t_test, x_pred_tst.mean(0).ravel(), label="forecast", color=colors[2])
    plt.fill_between(
        t_test, x_hpdi_tst[0].ravel(), x_hpdi_tst[1].ravel(), alpha=0.3, color=colors[2]
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "data.png")
    plt.close()


def main() -> None:

    # Data
    x, t = _load_data(5)
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
    predictive = infer.Predictive(model, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, t)

    _save_results(x, prior_samples, posterior_samples, posterior_predictive, num_train)


if __name__ == "__main__":
    main()
