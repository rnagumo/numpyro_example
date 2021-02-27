"""Boston hous pricing regression with missing features."""

import pathlib
from typing import Dict, Optional, Tuple

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import diagnostics, infer
from sklearn.datasets import load_boston


def bayesian_regression(x: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    batch, x_dim = jnp.shape(x)
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(x_dim), jnp.ones(x_dim) * 100))
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))

    with numpyro.plate("batch", batch, dim=-2):
        x_sample = numpyro.sample("x_sample", dist.Normal(x.mean(axis=0), x.std(axis=0)), obs=x)
    numpyro.sample("y", dist.Normal(jnp.matmul(x_sample, theta), sigma), obs=y)


def _load_dataset(missing_rate: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x, y = load_boston(return_X_y=True)
    x_missing = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_missing[i, j] = np.nan if np.random.rand() < missing_rate else x_missing[i, j]

    assert np.isnan(x).sum() == 0

    return x, y, x_missing


def _save_results(
    y: np.ndarray,
    mcmc: infer.MCMC,
    prior: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
) -> None:

    root = pathlib.Path("./data/boston_reg")
    root.mkdir(exist_ok=True)

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data, var_names=("theta", "sigma"))
    plt.savefig(root / "boston_trace.png")
    plt.close()

    az.plot_ppc(numpyro_data)
    plt.legend(loc="upper right")
    plt.savefig(root / "boston_ppc.png")
    plt.close()

    y_pred = posterior_predictive["y"]
    y_hpdi = diagnostics.hpdi(y_pred)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.figure(figsize=(12, 6))
    plt.plot(y, color=colors[0])
    plt.plot(y_pred.mean(axis=0), color=colors[1])
    plt.fill_between(np.arange(len(y)), y_hpdi[0], y_hpdi[1], color=colors[1], alpha=0.3)
    plt.xlabel("Index [a.u.]")
    plt.ylabel("Target [a.u.]")
    plt.savefig(root / "boston_prediction.png")
    plt.close()

    jnp.savez(root / "boston_posterior.npz", **posterior_samples)


def main() -> None:

    _, y, x_missing = _load_dataset()

    num_chains = 4
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_posterior, rng_key_prior = random.split(rng_key, 3)

    predictive = infer.Predictive(bayesian_regression, num_samples=500)
    prior = predictive(rng_key_prior, x_missing)

    kernel = infer.NUTS(bayesian_regression)
    mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, x_missing, y)
    posterior_samples = mcmc.get_samples()

    predictive = infer.Predictive(bayesian_regression, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, x_missing)

    _save_results(y, mcmc, prior, posterior_samples, posterior_predictive)


if __name__ == "__main__":
    main()
