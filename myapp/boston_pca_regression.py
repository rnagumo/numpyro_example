import argparse
import pathlib
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


def pca_regression(
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    batch: int = 100,
    x_dim: int = 50,
    z_dim: int = 5,
) -> None:

    if x is not None:
        batch, x_dim = jnp.shape(x)
        x_mu = np.nanmean(x, axis=0)
        x_std = np.nanstd(x, axis=0)
        mask = ~np.isnan(x)
    else:
        x_mu = jnp.zeros(x_dim)
        x_std = jnp.ones(x_dim)
        mask = False

    if y is not None:
        y_mu = np.mean(y) * np.ones(z_dim)
        y_std = np.std(y) * np.ones(z_dim)
        y_std_val = np.std(y)
    else:
        y_mu = np.zeros(z_dim)
        y_std = np.ones(z_dim)
        y_std_val = 1.0

    phi = numpyro.sample("phi", dist.Normal(jnp.zeros((z_dim, x_dim)), jnp.ones((z_dim, x_dim))))
    eta = numpyro.sample("eta", dist.Normal(x_mu, x_std))
    theta = numpyro.sample("theta", dist.Normal(y_mu, y_std))
    sigma = numpyro.sample("sigma", dist.Gamma(y_std_val, 1.0))

    with numpyro.plate("batch", batch, dim=-2):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
        numpyro.sample("x_sample", dist.Normal(jnp.matmul(z, phi) + eta, x_std))
        numpyro.sample("x", dist.Normal(jnp.matmul(z, phi) + eta, x_std).mask(mask), obs=x)

    numpyro.sample("y", dist.Normal(jnp.matmul(z, theta), sigma), obs=y)


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
    prior: Optional[Dict[str, jnp.ndarray]] = None,
    posterior_samples: Optional[Dict[str, jnp.ndarray]] = None,
    posterior_predictive: Optional[Dict[str, jnp.ndarray]] = None,
    *,
    var_names: Optional[List[str]] = None,
) -> None:

    root = pathlib.Path("./data/boston_pca_reg")
    root.mkdir(exist_ok=True)

    jnp.savez(root / "posterior_samples.npz", **posterior_samples)
    jnp.savez(root / "posterior_predictive.npz", **posterior_predictive)

    # Arviz
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data, var_names=var_names)
    plt.savefig(root / "trace.png")
    plt.close()

    az.plot_ppc(numpyro_data)
    plt.legend(loc="upper right")
    plt.savefig(root / "ppc.png")
    plt.close()

    # Prediction
    y_pred = posterior_predictive["y"]
    y_hpdi = diagnostics.hpdi(y_pred)
    train_len = int(len(y) * 0.99)

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


def main(args: argparse.Namespace) -> None:

    _, y, x_missing = _load_dataset()
    batch, x_dim = x_missing.shape
    train_len = int(len(y) * 0.99)
    x_train = x_missing[:train_len]
    y_train = y[:train_len]

    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(args.num_chains)
    rng_key = random.PRNGKey(1)
    rng_key, rng_key_prior, rng_key_posterior, rng_key_pca_pred = random.split(rng_key, 4)

    predictive = infer.Predictive(pca_regression, num_samples=500)
    prior = predictive(rng_key_prior, batch=batch, x_dim=x_dim)

    kernel = infer.NUTS(pca_regression)
    mcmc = infer.MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    mcmc.run(rng_key_posterior, x_train, y_train)
    posterior_samples = mcmc.get_samples()

    posterior_without_z = posterior_samples.copy()
    posterior_without_z.pop("z")
    predictive = infer.Predictive(pca_regression, posterior_samples=posterior_without_z)
    posterior_predictive = predictive(rng_key_pca_pred, x_missing)

    _save_results(
        y,
        mcmc,
        prior,
        posterior_samples,
        posterior_predictive,
        var_names=["phi", "eta", "theta", "sigma"],
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=1)
    args = parser.parse_args()

    main(args)
