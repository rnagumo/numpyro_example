import argparse
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.examples.datasets import BASEBALL, load_dataset
from numpyro.infer import HMC, MCMC, NUTS, SA, Predictive, log_likelihood


def fully_pooled(at_bats: jnp.ndarray, hits: Optional[jnp.ndarray] = None) -> jnp.ndarray:

    phi = numpyro.sample("phi", dist.Uniform(0, 1))
    num_players = at_bats.shape[0]
    with numpyro.plate("num_players", num_players):
        return numpyro.sample("obs", dist.Binomial(at_bats, probs=phi), obs=hits)


def not_pooled(at_bats: jnp.ndarray, hits: Optional[jnp.ndarray] = None) -> jnp.ndarray:

    num_players = at_bats.shape[0]
    with numpyro.plate("num_players", num_players):
        phi = numpyro.sample("phi", dist.Uniform(0, 1))
        return numpyro.sample("obs", dist.Binomial(at_bats, probs=phi), obs=hits)


def partially_pooled(at_bats: jnp.ndarray, hits: Optional[jnp.ndarray] = None) -> jnp.ndarray:

    m = numpyro.sample("m", dist.Uniform(0, 1))
    kappa = numpyro.sample("kappa", dist.Pareto(1, 1.5))
    num_players = at_bats.shape[0]
    with numpyro.plate("num_players", num_players):
        phi = numpyro.sample("phi", dist.Beta(m * kappa, (1 - m) * kappa))
        return numpyro.sample("obs", dist.Binomial(at_bats, probs=phi), obs=hits)


def partially_pooled_with_logit(
    at_bats: jnp.ndarray, hits: Optional[jnp.ndarray] = None
) -> jnp.ndarray:

    loc = numpyro.sample("loc", dist.Normal(-1, 1))
    scale = numpyro.sample("scale", dist.HalfCauchy(1))
    num_players = at_bats.shape[0]
    with numpyro.plate("num_players", num_players):
        alpha = numpyro.sample("alpha", dist.Normal(loc, scale))
        return numpyro.sample("obs", dist.Binomial(at_bats, logits=alpha), obs=hits)


def run_inference(
    model: Callable,
    at_bats: jnp.ndarray,
    hits: jnp.ndarray,
    rng_key: jnp.ndarray,
    *,
    num_warmup: int = 1500,
    num_samples: int = 3000,
    num_chains: int = 1,
    algo_name: str = "NUTS",
) -> jnp.ndarray:

    if algo_name == "NUTS":
        kernel = NUTS(model)
    elif algo_name == "HMC":
        kernel = HMC(model)
    elif algo_name == "SA":
        kernel = SA(model)
    else:
        raise ValueError("Unknown algorithm name")

    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains)
    mcmc.run(rng_key, at_bats, hits)
    return mcmc.get_samples()


def predict(
    model: Callable, at_bats: jnp.ndarray, posterior_samples: jnp.ndarray, rng_key: jnp.ndarray,
) -> jnp.ndarray:

    predictive = Predictive(model, posterior_samples=posterior_samples)
    predictive_samples = predictive(rng_key, at_bats)
    predictions = predictive_samples["obs"]

    return predictions    


def main(args: argparse.Namespace) -> None:

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    _, fetch_train = load_dataset(BASEBALL, split="train", shuffle=False)
    train, player_names = fetch_train()

    _, fetch_test = load_dataset(BASEBALL, split="test", shuffle=False)
    test, _ = fetch_test()

    at_bats = train[:, 0]
    hits = train[:, 1]
    at_bats_test = test[:, 0]
    hist_test = test[:, 1]

    model_list = [fully_pooled, not_pooled, partially_pooled, partially_pooled_with_logit]
    for i, model in enumerate(model_list, 1):
        rng_key, rng_key_predict = random.split(random.PRNGKey(i))
        zs = run_inference(
            model, at_bats, hits, rng_key, num_warmup=args.num_warmup,
            num_samples=args.num_samples, num_chains=args.num_chains, algo_name=args.algo_name,
        )
        predictions = predict(model, at_bats, zs, rng_key_predict)

        print(model.__name__)
        print(hits)
        print(predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-warmup", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--algo-name", type=str, default="NUTS")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
