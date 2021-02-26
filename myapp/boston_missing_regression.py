import argparse
from typing import Dict, List, Optional, Tuple

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import diagnostics, infer
from sklearn.datasets import load_boston


def pca_with_missing(
    x: Optional[np.ndarray] = None, batch: int = 100, x_dim: int = 50, z_dim: int = 5
) -> None:

    if x is not None:
        batch, x_dim = jnp.shape(x)

    phi = numpyro.sample("phi", dist.Normal(jnp.zeros((z_dim, x_dim)), jnp.ones((z_dim, x_dim))))
    eta = numpyro.sample("eta", dist.Normal(jnp.zeros(x_dim), jnp.ones(x_dim)))
    with numpyro.plate("batch", batch, dim=-2):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
        numpyro.sample("x", dist.Normal(jnp.matmul(z, phi) + eta, jnp.ones(x_dim)), obs=x)


def latent_regression(z: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    z_dim = z.shape[-1]
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))
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
    mcmc: infer.MCMC,
    y: Optional[np.ndarray] = None,
    prior: Optional[Dict[str, jnp.ndarray]] = None,
    posterior_samples: Optional[Dict[str, jnp.ndarray]] = None,
    posterior_predictive: Optional[Dict[str, jnp.ndarray]] = None,
    *,
    suffix: str = "",
    var_names: Optional[List[str]] = None,
) -> None:

    jnp.savez("./data/missing_boston_posterior.npz", **posterior_samples)

    # Arviz
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data, var_names=var_names)
    plt.savefig(f"./data/missing_boston_trace_{suffix}.png")
    plt.close()

    if y is None:
        return

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
    plt.savefig(f"./data/missing_boston_prediction_{suffix}.png")
    plt.close()


def main(args: argparse.Namespace) -> None:

    x, y, x_missing = _load_dataset()
    batch, x_dim = x.shape

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(args.num_chains)
    rng_key = random.PRNGKey(0)

    # PCA
    rng_key, rng_key_pca, rng_key_pca_pred = random.split(rng_key, 3)
    kernel = infer.NUTS(pca_with_missing)
    mcmc = infer.MCMC(
        kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains
    )
    mcmc.run(rng_key_pca, x)
    posterior_samples_pca = mcmc.get_samples()

    predictive = infer.Predictive(pca_with_missing, posterior_samples=posterior_samples_pca)
    posterior_predictive_pca = predictive(rng_key_pca_pred, batch=batch, x_dim=x_dim)

    rng_key, rng_key_impute = random.split(rng_key, 2)
    posterior_predictive_pca = _impute_missing_values(
        posterior_samples_pca, posterior_predictive_pca, rng_key_impute
    )

    _save_results(
        mcmc, None, None, posterior_samples_pca, posterior_predictive_pca,
        var_names=["phi", "eta"], suffix="pca",
    )

    # Linear regression
    rng_key, rng_key_lr, rng_key_lr_pred = random.split(rng_key, 3)
    z = posterior_samples_pca["z"].mean(axis=0)

    kernel = infer.NUTS(latent_regression)
    mcmc = infer.MCMC(
        kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains
    )
    mcmc.run(rng_key_lr, z, y)
    posterior_samples_lr = mcmc.get_samples()

    predictive = infer.Predictive(latent_regression, posterior_samples=posterior_samples_lr)
    posterior_predictive_lr = predictive(rng_key_lr_pred, z)

    _save_results(
        mcmc, y, None, posterior_samples_lr, posterior_predictive_lr, suffix="linear"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=1)
    args = parser.parse_args()

    main(args)
