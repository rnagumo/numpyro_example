from typing import Dict, Optional

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

    _, x_dim = jnp.shape(x)
    theta = numpyro.sample("theta", dist.Normal(jnp.zeros(x_dim), jnp.ones(x_dim) * 100))
    sigma = numpyro.sample("sigma", dist.Gamma(1.0, 1.0))
    numpyro.sample("y", dist.Normal(jnp.matmul(x, theta), sigma), obs=y)    


def _save_results(
    y: np.ndarray,
    mcmc: infer.MCMC,
    prior: Dict[str, jnp.ndarray],
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive: Dict[str, jnp.ndarray],
) -> None:

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

    az.plot_trace(numpyro_data)
    plt.savefig("./data/boston_trace.png")
    plt.close()

    az.plot_ppc(numpyro_data)
    plt.legend(loc="upper right")
    plt.savefig("./data/boston_ppc.png")
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
    plt.savefig("./data/boston_prediction.png")
    plt.close()

    jnp.savez("./data/boston_posterior.npz", **posterior_samples)


def main() -> None:

    x, y = load_boston(return_X_y=True)

    num_chains = 4
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_posterior, rng_key_prior = random.split(rng_key, 3)

    predictive = infer.Predictive(bayesian_regression, num_samples=500)
    prior = predictive(rng_key_prior, x)

    kernel = infer.NUTS(bayesian_regression)
    mcmc = infer.MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
    mcmc.run(rng_key, x, y)
    posterior_samples = mcmc.get_samples()

    predictive = infer.Predictive(bayesian_regression, posterior_samples=posterior_samples)
    posterior_predictive = predictive(rng_key_posterior, x)

    _save_results(y, mcmc, prior, posterior_samples, posterior_predictive)


if __name__ == "__main__":
    main()
