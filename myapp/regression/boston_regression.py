"""Boston hous pricing regression with missing features.

ref)
mask handler does not behave as expected
https://github.com/pyro-ppl/numpyro/issues/568

Behavior of mask handler with invalid observation — possible bug?
https://forum.pyro.ai/t/behavior-of-mask-handler-with-invalid-observation-possible-bug/1719/3

Improve Your Model with Missing Data | Imputation with NumPyro
https://towardsdatascience.com/improve-your-model-with-missing-data-imputation-with-numpyro-dcb3c3376eff

jax.ops.index_update
https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.index_update.html
"""

import pathlib
from typing import Dict, Optional, Tuple

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import ops, random
from numpyro import diagnostics, infer
from sklearn.datasets import load_boston


def bayesian_regression(x: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    batch, x_dim = jnp.shape(x)
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(x_dim), jnp.ones(x_dim) * 100))
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))

    x_mu = numpyro.sample("x_mu", dist.Normal(jnp.nanmean(x, axis=0), jnp.nanmean(x, axis=0)))
    x_std = numpyro.sample("x_std", dist.Gamma(1.0, 1.0))
    with numpyro.plate("batch", batch, dim=-2):
        mask = ~np.isnan(x)
        numpyro.sample("x", dist.Normal(x_mu, x_std).mask(mask), obs=x)

        index = (~mask).astype(int).nonzero()
        x_sample = numpyro.sample("x_sample", dist.Normal(x_mu, x_std))
        x_filled = ops.index_update(x, index, x_sample[index])

    numpyro.sample("y", dist.Normal(jnp.matmul(x_filled, theta), sigma), obs=y)


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

    jnp.savez(root / "posterior_samples.npz", **posterior_samples)
    jnp.savez(root / "posterior_predictive.npz", **posterior_predictive)

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data, var_names=("theta", "sigma"))
    plt.savefig(root / "trace.png")
    plt.close()

    az.plot_ppc(numpyro_data)
    plt.legend(loc="upper right")
    plt.savefig(root / "ppc.png")
    plt.close()

    y_pred = posterior_predictive["y"]
    y_hpdi = diagnostics.hpdi(y_pred)
    train_len = int(len(y) * 0.8)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure(figsize=(12, 6))
    plt.plot(y, color=colors[0])
    plt.plot(y_pred.mean(axis=0), color=colors[1])
    plt.fill_between(np.arange(len(y)), y_hpdi[0], y_hpdi[1], color=colors[1], alpha=0.3)
    plt.axvline(train_len, linestyle="--", color=colors[2])
    plt.xlabel("Index [a.u.]")
    plt.ylabel("Target [a.u.]")
    plt.savefig(root / "prediction.png")
    plt.close()


def main() -> None:

    _, y, x_missing = _load_dataset()
    train_len = int(len(y) * 0.8)
    x_train = x_missing[:train_len]
    y_train = y[:train_len]

    num_chains = 1
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_posterior, rng_key_prior = random.split(rng_key, 3)

    predictive = infer.Predictive(bayesian_regression, num_samples=500)
    prior = predictive(rng_key_prior, x_train)

    kernel = infer.NUTS(bayesian_regression)
    mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, x_train, y_train)
    posterior_samples = mcmc.get_samples()

    predictive = infer.Predictive(bayesian_regression, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, x_missing)

    _save_results(y, mcmc, prior, posterior_samples, posterior_predictive)


if __name__ == "__main__":
    main()
