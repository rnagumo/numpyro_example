"""Enumerate HMM

ref) https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm_enum.py
"""

import argparse
import pathlib
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro import infer
from numpyro.contrib.control_flow import scan
from numpyro.examples.datasets import JSB_CHORALES, load_dataset
from numpyro.handlers import mask


def model_1(
    sequences: Optional[np.ndarray] = None,
    lengths: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    batch: int = 100,
    seq_len: int = 0,
    data_dim: int = 10,
    future_steps: int = 0,
) -> None:

    if sequences is not None:
        assert lengths is not None
        batch, seq_len, data_dim = sequences.shape

        future = np.zeros((batch, future_steps, data_dim))
        sequences = np.concatenate([sequences, future], axis=1)
    else:
        lengths = np.zeros((batch))

    probs_x = numpyro.sample(
        "probs_x", dist.Dirichlet(0.9 * jnp.eye(hidden_dim) + 0.1).to_event(1)
    )
    probs_y = numpyro.sample(
        "probs_y", dist.Beta(0.1, 0.9).expand([hidden_dim, data_dim]).to_event(2)
    )

    def transition_fn(
        carry: Tuple[jnp.ndarray, jnp.ndarray], y: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """One time step funciton."""

        x_prev, t = carry
        with numpyro.plate("sequence", batch, dim=-2):
            with mask(mask=(t < lengths)[..., None]):
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                with numpyro.plate("tones", data_dim, dim=-1):
                    numpyro.sample("y", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    x_init = jnp.zeros((batch, 1), dtype=jnp.int32)

    if sequences is not None:
        # for loop with time step: data shape = (seq, batch, data_dim)
        scan(transition_fn, (x_init, 0), jnp.swapaxes(sequences, 0, 1))
    else:
        scan(transition_fn, (x_init, 0), None, length=seq_len + future_steps)


model_dict = {
    name.strip("model_"): model for name, model in globals().items() if name.startswith("model_")
}


def main(args: argparse.Namespace) -> None:

    model = model_dict[args.model]

    _, fetch = load_dataset(JSB_CHORALES, split="train", shuffle=False)
    lengths, sequences = fetch()

    # Remove never used data dimension to reduce computation time
    present_notes = (sequences == 1).sum(0).sum(0) > 0
    sequences = sequences[..., present_notes]
    batch, seq_len, data_dim = sequences.shape

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_pred = random.split(rng_key, 3)

    predictive = infer.Predictive(model, num_samples=10)
    prior_samples = predictive(
        rng_key_prior, batch=batch, seq_len=seq_len, data_dim=data_dim, future_steps=20
    )

    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(kernel, args.num_warmup, args.num_samples, args.num_chains)
    mcmc.run(rng_key, sequences, lengths)
    posterior_samples = mcmc.get_samples()

    predictive = infer.Predictive(model, posterior_samples)
    predictive_samples = predictive(rng_key_pred, sequences, lengths, future_steps=10)

    path = pathlib.Path("./data/hmm_enum")
    path.mkdir(exist_ok=True)

    jnp.savez(path / "prior_samples.npz", **prior_samples)
    jnp.savez(path / "posterior_samples.npz", **posterior_samples)
    jnp.savez(path / "predictive_samples.npz", **predictive_samples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
