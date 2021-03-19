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
    x: Optional[jnp.ndarray] = None, future_steps: int = 0, batch: int = 0, x_dim: int = 1
) -> None:
    """Simple Kalman filter model (random walk).

    Args:
        x: **Batch-first** data, `shape = (seq_len, batch, data_dim)`.
        future_steps: Forecasting time steps.
        batch: Batch size for prior sampling.
        x_dim: Dimension of data for prior sampling.
    """

    if x is not None:
        seq_len, batch, x_dim = x.shape
    else:
        seq_len = 0

    trans = numpyro.sample("trans", dist.Normal(0, 1))
    emit = numpyro.sample("emit", dist.Normal(0, 1))

    def transition_fn(
        carry: Tuple[jnp.ndarray], t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray], jnp.ndarray]:

        z_prev, *_ = carry
        z = numpyro.sample("z", dist.Normal(trans * z_prev, 1))
        numpyro.sample("x", dist.Normal(emit * z, 1))
        numpyro.sample("x_sample", dist.Normal(emit * z, 1))
        return (z,), None

    z_init = jnp.zeros((batch, x_dim))
    with numpyro.handlers.condition(data={"x": x}):
        scan(transition_fn, (z_init,), jnp.arange(seq_len + future_steps))


def _save_results(
    x: jnp.ndarray,
    prior_samples: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
) -> None:

    root = pathlib.Path("./data/kalman")
    root.mkdir(exist_ok=True)

    jnp.savez(root / "piror_samples.npz", **prior_samples)
    jnp.savez(root / "posterior_samples.npz", **posterior_samples)
    jnp.savez(root / "posterior_predictive.npz", **posterior_predictive)

    x_pred_trn = posterior_samples["x_sample"]
    x_hpdi_trn = diagnostics.hpdi(x_pred_trn)
    len_train = x_pred_trn.shape[1]

    x_pred_tst = posterior_predictive["x"][:, len_train:]
    x_hpdi_tst = diagnostics.hpdi(x_pred_tst)
    len_test = x_pred_tst.shape[1]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure(figsize=(12, 6))
    plt.plot(x.ravel(), label="ground truth", color=colors[0])

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


def main() -> None:

    x = jnp.concatenate(
        [
            np.random.randn(10),
            np.random.randn(10) + 2,
            np.random.randn(10) - 1,
            np.random.randn(10) + 1,
        ]
    )
    x = x[:, None, None]

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_infer, rng_key_posterior = random.split(rng_key, 4)

    # prior
    predictive = infer.Predictive(model, num_samples=10)
    prior_samples = predictive(rng_key_prior, future_steps=20, batch=10, x_dim=1)

    # Inference
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 100, 100)
    mcmc.run(rng_key_infer, x)
    posterior_samples = mcmc.get_samples()

    # Posterior prediction
    predictive = infer.Predictive(model, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, x, future_steps=10)

    _save_results(x, prior_samples, posterior_samples, posterior_predictive)


if __name__ == "__main__":
    main()
