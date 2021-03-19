"""Example: Stochastic Volatility

http://num.pyro.ai/en/latest/examples/stochastic_volatility.html
"""

import pathlib
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import infer
from numpyro.examples import datasets


def model(returns: np.ndarray) -> None:

    step_size = numpyro.sample("sigma", dist.Exponential(50.))
    s = numpyro.sample(
        "s", dist.GaussianRandomWalk(scale=step_size, num_steps=jnp.shape(returns)[0])
    )
    nu = numpyro.sample("nu", dist.Exponential(0.1))
    numpyro.sample("r", dist.StudentT(df=nu, loc=0.0, scale=jnp.exp(s)), obs=returns)


def _save_results(
    dates: np.ndarray, returns: np.ndarray, hmc_states: Dict[str, jnp.ndarray]
) -> None:

    root = pathlib.Path("./data/volatility")
    root.mkdir(exist_ok=True)

    dates = mdates.num2date(mdates.datestr2num(dates))

    plt.figure(figsize=(8, 6))
    plt.plot(dates, returns)
    plt.plot(dates, jnp.exp(hmc_states["s"].T), "r", alpha=0.01)
    plt.tight_layout()
    plt.savefig(root / "prediction.png")
    plt.close()


def main() -> None:

    _, fetch = datasets.load_dataset(datasets.SP500, shuffle=True)
    dates, returns = fetch()

    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 1000, 1000)
    mcmc.run(random.PRNGKey(0), returns)
    hmc_states = mcmc.get_samples()

    _save_results(dates, returns, hmc_states)


if __name__ == "__main__":
    main()
