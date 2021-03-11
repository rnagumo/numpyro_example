"""Enumerate HMM

ref) https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm_enum.py
"""

import argparse
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.contrib.control_flow import scan
from numpyro.contrib.indexing import Vindex
from numpyro.examples.datasets import JSB_CHORALES, load_dataset
from numpyro.handlers import mask
from numpyro.infer import MCMC, NUTS


def model_1(sequences: np.ndarray, lengths: np.ndarray, hidden_dim: int = 16) -> None:

    # (batch, seq, data_dim)
    num_sequences, _, data_dim = sequences.shape

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
        with numpyro.plate("sequence", num_sequences, dim=-2):
            with mask(mask=(t < lengths)[..., None]):
                # Forward transition
                x = numpyro.sample("x", dist.Categorical(probs_x[x_prev]))
                with numpyro.plate("tones", data_dim, dim=-1):
                    # Observe y
                    numpyro.sample("y", dist.Bernoulli(probs_y[x.squeeze(-1)]), obs=y)
        return (x, t + 1), None

    x_init = jnp.zeros((num_sequences, 1), dtype=jnp.int32)
    # for loop with time step: data shape = (seq, batch, data_dim)
    scan(transition_fn, (x_init, 0), jnp.swapaxes(sequences, 0, 1))


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

    rng_key = random.PRNGKey(0)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, args.num_chains)
    mcmc.run(rng_key, sequences, lengths)
    mcmc.print_summary()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-warmup", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
