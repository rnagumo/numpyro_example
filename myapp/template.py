"""Template with 8 schools example."""

import pathlib
from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive


def model(sigma: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    num = len(sigma)
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.Normal(0, 5))
    with numpyro.plate("num", num):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def inference(
    model: Callable,
    sigma: np.ndarray,
    y: np.ndarray,
    rng_key: np.ndarray,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, sigma, y)
    if verbose:
        mcmc.print_summary()

    return mcmc.get_samples()


def predict(
    model: Callable,
    sigma: np.ndarray,
    rng_key: np.ndarray,
    *,
    posterior_samples: Optional[Dict[str, jnp.ndarray]] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, jnp.ndarray]:

    predictive = Predictive(model, posterior_samples=posterior_samples, num_samples=num_samples)

    return predictive(rng_key, sigma)


def save_params(path: Union[str, pathlib.Path], params: Dict[str, jnp.ndarray]) -> None:

    jnp.savez(path, **params)


def load_params(path: Union[str, pathlib.Path]) -> Dict[str, jnp.ndarray]:

    with jnp.load(path) as data:
        res = {k: jnp.array(v) for k, v in data.items()}

    return res


if __name__ == "__main__":

    num_chains = 4
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(num_chains)

    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    rng_key = random.PRNGKey(0)

    posterior = inference(model, sigma, y, rng_key, num_chains=num_chains)
    prediction = predict(model, sigma, rng_key, posterior_samples=posterior)
    print(np.mean(prediction["obs"], axis=0))
