"""Forecasting III: hierarchical models

https://pyro.ai/examples/forecasting_iii.html
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
    x: Optional[jnp.ndarray] = None,
    seq_len: int = 0,
    batch: int = 0,
    x_dim: int = 1,
    z_dim: int = 3,
    future_steps: int = 0,
) -> None:
    """Hierarchical Kalman filter."""

    if x is not None:
        seq_len, batch, x_dim = x.shape

    trans_mu = numpyro.sample(
        "trans_mu", dist.Uniform(-jnp.ones((z_dim, z_dim)), jnp.ones((z_dim, z_dim)))
    )
    trans_var = numpyro.sample(
        "trans_var", dist.LogNormal(-20 * jnp.ones((z_dim, z_dim)), 20 * jnp.ones((z_dim, z_dim)))
    )
    with numpyro.plate("batch", batch, dim=-3):
        trans = numpyro.sample("trans", dist.Normal(trans_mu, trans_var).expand((1, z_dim, z_dim)))

    emit = numpyro.sample("emit", dist.Normal(jnp.zeros((z_dim, x_dim)), jnp.ones((z_dim, x_dim))))
    z_std = numpyro.sample("z_std", dist.Gamma(jnp.ones(z_dim), jnp.ones(z_dim)))
    x_std = numpyro.sample("x_std", dist.Gamma(jnp.ones(x_dim), jnp.ones(x_dim)))

    def transition_fn(
        carry: Tuple[jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:

        z_prev, *_ = carry
        index = jnp.arange(batch)
        z = numpyro.sample("z", dist.Normal(jnp.matmul(z_prev, trans)[index, index], z_std))
        numpyro.sample("x", dist.Normal(jnp.matmul(z, emit), x_std))
        return (z,), None

    z_init = jnp.zeros((batch, z_dim))
    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (z_init,), jnp.arange(seq_len + future_steps))


def _load_dataset() -> jnp.ndarray:
    def _load_single_data() -> jnp.ndarray:
        x0 = jnp.concatenate(
            [
                np.random.randn(10, 2),
                np.random.randn(10, 2) + 1,
                np.random.randn(10, 2) + 1.2,
                np.random.randn(10, 2) + 2,
            ]
        )

        x1 = jnp.concatenate(
            [
                np.random.randn(10, 2) - 0.2,
                np.random.randn(10, 2) - 1,
                np.random.randn(10, 2) - 2.7,
                np.random.randn(10, 2) - 4.2,
            ]
        )

        x = jnp.concatenate([x0[..., None], x1[..., None]], axis=-1)
        return x

    x = jnp.concatenate([_load_single_data() for _ in range(10)], axis=1)
    assert isinstance(x, jnp.ndarray)

    return x


def _save_results(
    x: jnp.ndarray,
    prior_samples: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
    num_train: int,
) -> None:

    root = pathlib.Path("./data/hierarchical")
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

    plt.figure(figsize=(12, 12))

    plt.subplot(211)
    plt.plot(x[..., 0, 0], label="ground truth", color=colors[0])

    plt.plot(t_train, x_pred_trn[..., 0, 0].mean(0), label="prediction", color=colors[1])
    plt.fill_between(
        t_train, x_hpdi_trn[0, :, 0, 0], x_hpdi_trn[1, :, 0, 0], alpha=0.3, color=colors[1]
    )

    plt.plot(t_test, x_pred_tst[..., 0, 0].mean(0), label="forecast", color=colors[2])
    plt.fill_between(
        t_test, x_hpdi_tst[0, :, 0, 0], x_hpdi_tst[1, :, 0, 0], alpha=0.3, color=colors[2]
    )

    plt.legend()

    plt.subplot(212)
    plt.plot(x[..., 0, 1], label="ground truth", color=colors[0])

    plt.plot(t_train, x_pred_trn[..., 0, 1].mean(0), label="prediction", color=colors[1])
    plt.fill_between(
        t_train, x_hpdi_trn[0, :, 0, 1], x_hpdi_trn[1, :, 0, 1], alpha=0.3, color=colors[1]
    )

    plt.plot(t_test, x_pred_tst[..., 0, 1].mean(0), label="forecast", color=colors[2])
    plt.fill_between(
        t_test, x_hpdi_tst[0, :, 0, 1], x_hpdi_tst[1, :, 0, 1], alpha=0.3, color=colors[2]
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig(root / "prediction.png")
    plt.close()


def main() -> None:

    x = _load_dataset()

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_infer, rng_key_posterior = random.split(rng_key, 4)

    # prior
    predictive = infer.Predictive(model, num_samples=10)
    prior_samples = predictive(rng_key_prior, None, *x.shape, future_steps=20)

    # Inference
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 100, 100)
    mcmc.run(rng_key_infer, x)
    posterior_samples = mcmc.get_samples()

    # Posterior prediction
    predictive = infer.Predictive(model, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, None, *x.shape, future_steps=20)

    _save_results(x, prior_samples, posterior_samples, posterior_predictive, len(x))


if __name__ == "__main__":
    main()
