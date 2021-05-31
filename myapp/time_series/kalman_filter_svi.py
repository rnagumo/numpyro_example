"""Kalman filter with SVI.

http://num.pyro.ai/en/stable/svi.html
"""

import pathlib
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import diagnostics, infer, optim
from numpyro.contrib.control_flow import scan
from numpyro.distributions import constraints
from numpyro.infer.svi import SVIRunResult


def model(
    x: Optional[jnp.ndarray] = None,
    seq_len: int = 0,
    batch: int = 0,
    x_dim: int = 1,
    future_steps: int = 0,
) -> None:
    """Simple Kalman filter model (random walk).

    Args:
        x: **Batch-first** data, `shape = (seq_len, batch, data_dim)`.
        seq_len: Length of sequence.
        batch: Batch size for prior sampling.
        x_dim: Dimension of data for prior sampling.
        future_steps: Forecasting time steps.
    """

    if x is not None:
        seq_len, batch, x_dim = x.shape

    trans = numpyro.param("trans", 1.0)
    emit = numpyro.param("emit", 1.0)

    def transition_fn(
        carry: Tuple[jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:

        z_prev, *_ = carry
        z = numpyro.sample("z", dist.Normal(trans * z_prev, 1))
        numpyro.sample("x", dist.Normal(emit * z, 1))
        return (z,), None

    z_init = jnp.zeros((batch, x_dim))
    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (z_init,), jnp.arange(seq_len + future_steps))


def guide(
    x: jnp.ndarray,
    seq_len: int = 0,
    batch: int = 0,
    x_dim: int = 1,
    future_steps: int = 0,
) -> None:

    if x is not None:
        *_, x_dim = x.shape

    phi = numpyro.param("phi", jnp.ones(x_dim))
    sigma = numpyro.param("sigma", jnp.ones(x_dim), constraint=constraints.positive)
    numpyro.sample("z", dist.Normal(x * phi, sigma))


def _save_results(
    x: jnp.ndarray,
    posterior_params: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
    svi_result: SVIRunResult,
) -> None:

    root = pathlib.Path("./data/kalman_svi")
    root.mkdir(exist_ok=True)

    jnp.savez(root / "posterior_params.npz", **posterior_params)
    jnp.savez(root / "posterior_predictive.npz", **posterior_predictive)
 
    len_train = x.shape[0]
    x_pred_trn = posterior_predictive["x"][:, :len_train, 0]
    x_hpdi_trn = diagnostics.hpdi(x_pred_trn)[:, :, 0]
    x_pred_tst = posterior_predictive["x"][:, len_train:, 0]
    x_hpdi_tst = diagnostics.hpdi(x_pred_tst)[:, :, 0]
    len_test = x_pred_tst.shape[1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure(figsize=(12, 6))
    plt.plot(x[:, 0].ravel(), label="ground truth", color=colors[0])

    t_train = np.arange(len_train)
    plt.plot(t_train, x_pred_trn.mean(0).ravel(), label="prediction", color=colors[1])
    plt.fill_between(
        t_train, x_hpdi_trn[0].ravel(), x_hpdi_trn[1].ravel(), alpha=0.3, color=colors[1]
    )

    t_test = np.arange(len_train, len_train + len_test)
    plt.plot(t_test, x_pred_tst.mean(0).ravel(), label="forecast", color=colors[2])
    plt.fill_between(
        t_test, x_hpdi_tst[0].ravel(), x_hpdi_tst[1].ravel(), alpha=0.3, color=colors[2]
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "kalman.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(svi_result.losses[::100])
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(root / "losses.png")
    plt.close()


def main() -> None:

    x = jnp.array(
        [
            jnp.concatenate(
                [
                    np.random.randn(10),
                    np.random.randn(10) + 2,
                    np.random.randn(10) - 1,
                    np.random.randn(10) + 1,
                ]
            )
            for _ in range(100)
        ]
    )
    x = x[..., None]
    x = jnp.swapaxes(x, 0, 1)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_infer, rng_key_posterior = random.split(rng_key, 3)

    # Inference
    adam = optim.Adam(0.00001)
    svi = infer.SVI(model, guide, adam, infer.Trace_ELBO())
    svi_result = svi.run(rng_key_infer, 100000, x)

    # Posterior prediction
    predictive = infer.Predictive(model, params=svi_result.params, num_samples=10)
    posterior_predictive = predictive(rng_key_posterior, None, *x.shape, future_steps=10)

    _save_results(x, svi_result.params, posterior_predictive, svi_result)


if __name__ == "__main__":
    main()
