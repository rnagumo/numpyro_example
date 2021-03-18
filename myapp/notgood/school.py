from typing import Optional

import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import TransformReparam


def model(num: int, sigma: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.Normal(0, 5))
    with numpyro.plate("num", num):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def model_noncentered(num: int, sigma: np.ndarray, y: Optional[np.ndarray] = None) -> None:

    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("num", num):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
                ),
            )

        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


def model_pred() -> np.ndarray:

    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    return numpyro.sample("obs", dist.Normal(mu, tau))


def main() -> None:

    # Data
    num = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    # Random key
    rng_key = random.PRNGKey(0)

    # Inference
    nuts_kernel = NUTS(model_noncentered)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(rng_key, num, sigma, y=y, extra_fields=("potential_energy",))
    print(mcmc.print_summary())

    # Extra
    pe = mcmc.get_extra_fields()["potential_energy"]
    print(f"Expected log joint density: {np.mean(-pe):.2f}")

    # Prediction
    predictive = Predictive(model_pred, num_samples=100)
    samples = predictive(random.PRNGKey(1))
    print("prior", np.mean(samples["obs"]))

    predictive = Predictive(model_pred, mcmc.get_samples())
    samples = predictive(random.PRNGKey(1))
    print("posterior", np.mean(samples["obs"]))


if __name__ == "__main__":
    main()
