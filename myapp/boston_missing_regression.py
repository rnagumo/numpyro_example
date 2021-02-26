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


def bayesian_pca_regression(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    batch: int = 100,
    x_dim: int = 100,
    z_dim: int = 5,
) -> None:

    if x is not None:
        batch, x_dim = jnp.shape(x)

    phi = numpyro.sample("phi", dist.Normal(jnp.zeros((z_dim, x_dim)), jnp.ones((z_dim, x_dim))))
    eta = numpyro.sample("eta", dist.Normal(jnp.zeros(x_dim), jnp.ones(x_dim)))
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))

    z = numpyro.sample("z", dist.Normal(jnp.zeros((batch, z_dim)), jnp.ones((batch, z_dim))))
    numpyro.sample("x", dist.Normal(jnp.matmul(z, phi) + eta, jnp.ones(x_dim)), obs=x)
    numpyro.sample("y", dist.Normal(jnp.matmul(z, theta), sigma), obs=y)


def sample_missing(posterior_samples: Dict[str, jnp.ndarray], rng_key: np.ndarray) -> jnp.ndarray:

    z = posterior_samples["z"]
    phi = posterior_samples["phi"]
    eta = posterior_samples["eta"]
    x_dim = phi.shape[-1]

    x_mu = jnp.einsum("nij,njh->nih", z, phi) + jnp.expand_dims(eta, 1)
    x_sample = dist.Normal(x_mu, jnp.ones((1, 1, x_dim))).sample(rng_key)

    return x_sample


def _load_dataset(missing_rate: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x, y = load_boston(return_X_y=True)
    x_missing = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_missing[i, j] = np.nan if np.random.rand() < missing_rate else x_missing[i, j]

    assert np.isnan(x).sum() == 0

    return x, y, x_missing


def _impute_missing_values(
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
    rng_key: np.ndarray,
) -> Dict[str, jnp.ndarray]:

    x_sample = sample_missing(posterior_samples, rng_key)
    x_filled = np.copy(posterior_predictive["x"])
    mask = np.isnan(x_filled)
    x_filled[mask] = x_sample[mask]
    posterior_predictive["x"] = x_filled

    return posterior_predictive


def _save_results(
    y: np.ndarray,
    mcmc: infer.MCMC,
    prior: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
) -> None:

    jnp.savez("./data/missing_boston_posterior.npz", **posterior_samples)

    # Arviz
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data)
    plt.savefig("./data/missing_boston_trace.png")
    plt.close()

    az.plot_ppc(numpyro_data)
    plt.legend(loc="upper right")
    plt.savefig("./data/missing_boston_ppc.png")
    plt.close()

    # Prediction
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
    plt.savefig("./data/missing_boston_prediction.png")
    plt.close()


def main() -> None:

    x, y, x_missing = _load_dataset()
    batch, x_dim = x_missing.shape

    num_chains = 1
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_posterior, rng_key_prior, rng_key_impute = random.split(rng_key, 4)

    # Sample prior
    predictive = infer.Predictive(bayesian_pca_regression, num_samples=500)
    prior = predictive(rng_key_prior, batch=batch, x_dim=x_dim)

    # Infer posterior
    kernel = infer.NUTS(bayesian_pca_regression)
    mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, x, y)
    posterior_samples = mcmc.get_samples()

    # Sample posterior predictive
    predictive = infer.Predictive(bayesian_pca_regression, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, x)

    posterior_predictive = _impute_missing_values(
        posterior_samples, posterior_predictive, rng_key_impute
    )

    _save_results(y, mcmc, prior, posterior_samples, posterior_predictive)


if __name__ == "__main__":
    main()
