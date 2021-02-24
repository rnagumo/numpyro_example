"""Hidden Markov Model example.

https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm.py
"""

import argparse
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax, random
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from scipy.stats import gaussian_kde


def simulate_data(
    rng_key: np.ndarray,
    num_categories: int,
    num_words: int,
    num_supervised: int,
    num_unsupservised: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    rng_key, rng_key_transition, rng_key_emission = random.split(rng_key, 3)

    transition_prior = jnp.ones(num_categories)
    emission_prior = jnp.repeat(0.1, num_words)

    transition_prob = dist.Dirichlet(transition_prior).sample(
        rng_key_transition, sample_shape=(num_categories,)
    )
    emission_prob = dist.Dirichlet(emission_prior).sample(
        rng_key_emission, sample_shape=(num_categories,)
    )

    start_prob = jnp.repeat(1.0 / num_categories, num_categories)
    category = 0
    categories = []
    words = []

    for t in range(num_supervised + num_unsupservised):
        rng_key, rng_key_transition, rng_key_emission = random.split(rng_key, 3)
        if t == 0 or t == num_supervised:
            category = dist.Categorical(start_prob).sample(rng_key_transition)
        else:
            category = dist.Categorical(transition_prob[category]).sample(rng_key_transition)

        word = dist.Categorical(emission_prob[category]).sample(rng_key_emission)
        categories.append(category)
        words.append(word)

    # Split data into supervised and unsupervised
    categories = jnp.stack(categories)
    words = jnp.stack(words)

    supervised_categories = categories[:num_supervised]
    supervised_words = words[:num_supervised]
    unsupervised_words = words[num_supervised:]

    return (
        transition_prob,
        emission_prob,
        supervised_categories,
        supervised_words,
        unsupervised_words,
    )


def forward_one_step(
    prev_log_prob: jnp.ndarray,
    curr_word: int,
    transition_log_prob: jnp.ndarray,
    emission_log_prob: jnp.ndarray,
) -> jnp.ndarray:

    log_prob_tmp = jnp.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_log_prob[:, curr_word]
    return logsumexp(log_prob, axis=0)


def forward_log_prob(
    init_log_prob: jnp.ndarray,
    words: jnp.ndarray,
    transition_log_prob: jnp.ndarray,
    emission_log_prob: jnp.ndarray,
) -> jnp.ndarray:
    def scan_fn(log_prob: jnp.ndarray, word: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            forward_one_step(log_prob, word, transition_log_prob, emission_log_prob),
            jnp.zeros((0,)),
        )

    log_prob, _ = lax.scan(scan_fn, init_log_prob, words)

    return log_prob


def semi_supervised_hmm(
    num_categories: int,
    num_words: int,
    supervised_categories: jnp.ndarray,
    supervised_words: jnp.ndarray,
    unsupervised_words: jnp.ndarray,
) -> None:

    transition_prior = jnp.ones(num_categories)
    emission_prior = jnp.repeat(0.1, num_words)

    transition_prob = numpyro.sample(
        "transition_prob",
        dist.Dirichlet(jnp.broadcast_to(transition_prior, (num_categories, num_categories))),
    )
    emission_prob = numpyro.sample(
        "emission_prob",
        dist.Dirichlet(jnp.broadcast_to(emission_prior, (num_categories, num_words))),
    )

    numpyro.sample(
        "supervised_categories",
        dist.Categorical(transition_prob[supervised_categories[:-1]]),
        obs=supervised_categories[1:],
    )
    numpyro.sample(
        "supervised_words",
        dist.Categorical(emission_prob[supervised_categories]),
        obs=supervised_words,
    )

    transition_log_prob = jnp.log(transition_prob)
    emission_log_prob = jnp.log(emission_prob)
    init_log_prob = emission_log_prob[:, unsupervised_words[0]]
    log_prob = forward_log_prob(
        init_log_prob, unsupervised_words[1:], transition_log_prob, emission_log_prob
    )
    log_prob = logsumexp(log_prob, axis=0, keepdims=True)
    numpyro.factor("forward_log_prob", log_prob)


def inference(
    model: Callable,
    num_categories: int,
    num_words: int,
    supervised_categories: jnp.ndarray,
    supervised_words: jnp.ndarray,
    unsupervised_words: jnp.ndarray,
    rng_key: np.ndarray,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(
        rng_key,
        num_categories,
        num_words,
        supervised_categories,
        supervised_words,
        unsupervised_words,
    )
    if verbose:
        mcmc.print_summary()

    return mcmc.get_samples()


def print_results(
    posterior: Dict[str, jnp.ndarray],
    transition_prob: jnp.ndarray,
    emission_prob: jnp.ndarray,
) -> None:

    header = "semi_supervised_hmm - TRAIN"
    columns = ["", "ActualProb", "Pred(p25)", "Pred(p50)", "Pred(p75"]
    header_format = "{:>20} {:>10} {:>10} {:>10} {:>10}"
    row_format = "{:>20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}"

    print("\n", "=" * 20 + header + "=" * 20, "\n")
    print(header_format.format(*columns))

    quantiles = np.quantile(posterior["transition_prob"], [0.25, 0.5, 0.75], axis=0)
    for i in range(transition_prob.shape[0]):
        for j in range(transition_prob.shape[1]):
            idx = f"transition[{i},{j}]"
            print(row_format.format(idx, transition_prob[i, j], *quantiles[:, i, j]), "\n")

    quantiles = np.quantile(posterior["emission_prob"], [0.25, 0.5, 0.75], axis=0)
    for i in range(emission_prob.shape[0]):
        for j in range(emission_prob.shape[1]):
            idx = f"emission[{i},{j}]"
            print(row_format.format(idx, emission_prob[i, j], *quantiles[:, i, j]), "\n")


def plot_results(
    posterior: Dict[str, jnp.ndarray],
    transition_prob: jnp.ndarray,
) -> None:

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure(figsize=(14, 6))
    x = np.linspace(0, 1, 101)
    index = 0
    for i in range(transition_prob.shape[0]):
        for j in range(transition_prob.shape[1]):
            y = gaussian_kde(posterior["transition_prob"][:, i, j])(x)
            title = f"Posterior: trnas_prob[{i},{j}], true value={transition_prob[i, j]:.2f}"

            plt.subplot(transition_prob.shape[0], transition_prob.shape[1], index + 1)
            plt.plot(x, y, color=colors[index])
            plt.axvline(transition_prob[i, j], linestyle="--", color=colors[index], alpha=0.6)
            plt.xlabel("Probability")
            plt.ylabel("Frequency")
            plt.title(title)
            index += 1

    plt.tight_layout()
    plt.show()


def main(args: argparse.Namespace) -> None:

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    (
        transition_prob,
        emission_prob,
        supervised_categories,
        supervised_words,
        unsupervised_words,
    ) = simulate_data(
        random.PRNGKey(1),
        num_categories=args.num_categories,
        num_words=args.num_words,
        num_supervised=args.num_supervised,
        num_unsupservised=args.num_unsupervised,
    )

    rng_key = random.PRNGKey(2)
    posterior = inference(
        semi_supervised_hmm,
        args.num_categories,
        args.num_words,
        supervised_categories,
        supervised_words,
        unsupervised_words,
        rng_key,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )

    print_results(posterior, transition_prob, emission_prob)
    plot_results(posterior, transition_prob)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-categories", type=int, default=3)
    parser.add_argument("--num-words", type=int, default=10)
    parser.add_argument("--num-supervised", type=int, default=100)
    parser.add_argument("--num-unsupervised", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--num-warmup", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
