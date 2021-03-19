"""Example: Neal’s Funnel

http://num.pyro.ai/en/latest/examples/funnel.html
"""

import pathlib
from typing import Callable, Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import handlers, infer
from numpyro.infer import reparam


def model(dim: int = 10) -> None:

    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))


reparam_model = handlers.reparam(model, config={"x": reparam.LocScaleReparam(0)})


def run_inference(model: Callable, rng_key: np.ndarray) -> Dict[str, jnp.ndarray]:

    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, 1000, 1000, 1)
    mcmc.run(rng_key)
    return mcmc.get_samples()


def _plot_results(
    samples: Dict[str, jnp.ndarray], reparam_samples: Dict[str, jnp.ndarray]
) -> None:

    root = pathlib.Path("./data/neals")
    root.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 8))

    plt.subplot(211)
    plt.scatter(samples["x"][:, 0], samples["y"], alpha=0.3)
    plt.xlim(-20, 20)
    plt.ylim(-9, 9)
    plt.title("Centered parametrization")

    plt.subplot(212)
    plt.scatter(reparam_samples["x"][:, 0], reparam_samples["y"], alpha=0.3)
    plt.xlim(-20, 20)
    plt.ylim(-9, 9)
    plt.title("Non-centered parametrization")

    plt.tight_layout()
    plt.savefig(root / "samples.png")
    plt.close()


def main() -> None:

    rng_key = random.PRNGKey(0)

    samples = run_inference(model, rng_key)
    reparam_samples = run_inference(reparam_model, rng_key)

    predictive = infer.Predictive(reparam_model, reparam_samples, return_sites=["x", "y"])
    reparam_samples = predictive(random.PRNGKey(1))

    _plot_results(samples, reparam_samples)


if __name__ == "__main__":
    main()
