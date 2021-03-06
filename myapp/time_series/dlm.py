"""Forecasting with Dynamic Linear Model (DLM)

https://pyro.ai/examples/forecasting_dlm.html
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
) -> None:

    if x is not None:
        x_dim = x.shape[-1]

    seq_len, batch, c_dim = covariates.shape
    weight_var = numpyro.sample(
        "weight_var", dist.LogNormal(-5 * jnp.ones((c_dim, x_dim)), 5 * jnp.ones((c_dim, x_dim)))
    )
    sigma = numpyro.sample("sigma", dist.LogNormal(-20 * jnp.ones(x_dim), 20 * jnp.ones(x_dim)))

    def transition_fn(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        t: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

        z_prev, w_prev = carry
        z = numpyro.sample("z", dist.Normal(z_prev, 1))
        weight = numpyro.sample("weight", dist.Normal(w_prev, weight_var))
        numpyro.sample("x", dist.Normal(z + jnp.matmul(covariates[t], weight), sigma))
        return (z, weight), None

    z_init = jnp.zeros((batch, x_dim))
    w_init = jnp.zeros((c_dim, x_dim))
    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (z_init, w_init), jnp.arange(seq_len))


def _load_data() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    p = 5
    n = 365 * 3

    beta0 = (np.random.randn(n, 1) * 0.1).cumsum(0)
    betas_p = (np.random.randn(n, p) * 0.1).cumsum(0)
    betas = jnp.concatenate([beta0, betas_p], axis=-1)
    covariates = jnp.concatenate([np.ones((n, 1)), np.random.randn(n, p) * 0.1], axis=-1)
    x = ((covariates * betas).sum(axis=-1) + 0.1 * np.random.randn(n))[:, None]

    betas = betas[:, None, :]
    covariates = covariates[:, None, :]
    x = x[:, None, :]

    assert isinstance(betas, jnp.ndarray)
    assert isinstance(covariates, jnp.ndarray)
    assert isinstance(x, jnp.ndarray)

    return x, betas, covariates


def _save_results(
    x: jnp.ndarray,
    betas: jnp.ndarray,
    prior_samples: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
    num_train: int,
) -> None:

    root = pathlib.Path("./data/dlm")
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

    t_axis = np.arange(num_train + num_test)
    w_pred = posterior_predictive["weight"]
    w_hpdi = diagnostics.hpdi(w_pred)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    beta_dim = betas.shape[-1]
    plt.figure(figsize=(8, 12))
    for i in range(beta_dim + 1):
        plt.subplot(beta_dim + 1, 1, i + 1)
        if i == 0:
            plt.plot(x[:, 0], label="ground truth", color=colors[0])
            plt.plot(t_train, x_pred_trn.mean(0)[:, 0], label="prediction", color=colors[1])
            plt.fill_between(
                t_train, x_hpdi_trn[0, :, 0, 0], x_hpdi_trn[1, :, 0, 0], alpha=0.3, color=colors[1]
            )
            plt.plot(t_test, x_pred_tst.mean(0)[:, 0], label="forecast", color=colors[2])
            plt.fill_between(
                t_test, x_hpdi_tst[0, :, 0, 0], x_hpdi_tst[1, :, 0, 0], alpha=0.3, color=colors[2]
            )
            plt.title("ground truth", fontsize=16)
        else:
            plt.plot(betas[:, 0, i - 1], label="ground truth", color=colors[0])
            plt.plot(t_axis, w_pred.mean(0)[:, i - 1], label="prediction", color=colors[1])
            plt.fill_between(
                t_axis, w_hpdi[0, :, i - 1, 0], w_hpdi[1, :, i - 1, 0], alpha=0.3, color=colors[1]
            )
            plt.title(f"coef_{i - 1}", fontsize=16)
        plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(root / "prediction.png")
    plt.close()


def main() -> None:

    # Data
    x, betas, covariates = _load_data()
    num_train = int(len(x) * 0.8)
    x_train = x[:num_train]
    c_train = covariates[:num_train]

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_infer, rng_key_posterior = random.split(rng_key, 4)

    # prior
    predictive = infer.Predictive(model, num_samples=10)
    prior_samples = predictive(rng_key_prior, c_train)

    # Inference
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 100, 100)
    mcmc.run(rng_key_infer, c_train, x_train)
    posterior_samples = mcmc.get_samples()

    # Posterior prediction
    posterior_given = posterior_samples.copy()
    posterior_given.pop("weight")
    predictive = infer.Predictive(
        model, posterior_samples=posterior_given, return_sites=["x", "weight"]
    )
    posterior_predictive = predictive(rng_key_posterior, covariates)

    _save_results(x, betas, prior_samples, posterior_samples, posterior_predictive, num_train)


if __name__ == "__main__":
    main()
