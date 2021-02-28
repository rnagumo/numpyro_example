import argparse
from typing import Callable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, lax, random
from jax.experimental import stax
from numpyro import infer, optim
from numpyro.examples.datasets import MNIST, load_dataset
from numpyro.infer.svi import SVIState


def encoder(hidden_dim: int, z_dim: int) -> Tuple[Callable, Callable]:
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(
                stax.Dense(z_dim, W_init=stax.randn()),
                stax.Exp,
            ),
        ),
    )


def decoder(hidden_dim: int, out_dim: int) -> Tuple[Callable, Callable]:
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )


def model(batch: np.ndarray, hidden_dim: int = 400, z_dim: int = 100) -> None:

    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim), (batch_dim, z_dim))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
    img_loc = decode(z)
    numpyro.sample("obs", dist.Bernoulli(img_loc), obs=batch)


def guide(batch: np.ndarray, hidden_dim: int = 400, z_dim: int = 100) -> None:

    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    numpyro.sample("z", dist.Normal(z_loc, z_std))


@jit
def binarize(rng_key: np.ndarray, batch: np.ndarray) -> jnp.ndarray:
    return random.bernoulli(rng_key, batch).astype(batch.dtype)


def main(args: argparse.Namespace) -> None:

    # Data
    train_init, train_fetch = load_dataset(MNIST, batch_size=args.batch_size, split="train")
    test_init, test_fetch = load_dataset(MNIST, batch_size=args.batch_size, split="test")
    num_train, train_idx = train_init()

    # Model
    encoder_nn = encoder(args.hidden_dim, args.z_dim)
    decoder_nn = decoder(args.hidden_dim, 28 * 28)
    adam = optim.Adam(args.learning_rate)
    svi = infer.SVI(
        model, guide, adam, infer.Trace_ELBO(), hidden_dim=args.hidden_dim, z_dim=args.z_dim
    )

    # Init
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_binarize, rng_key_init = random.split(rng_key, 3)
    sample_batch = binarize(rng_key_binarize, train_fetch(0, train_idx)[0])
    svi_state = svi.init(rng_key_init, sample_batch)

    @jit
    def epoch_train(svi_state: SVIState, rng_key: np.ndarray) -> Tuple[jnp.ndarray, SVIState]:
        def body_fun(
            i: jnp.ndarray, val: Tuple[jnp.ndarray, SVIState]
        ) -> Tuple[jnp.ndarray, SVIState]:
            loss_sum, svi_state = val
            rng_key_binarize = random.fold_in(rng_key, i)
            batch = binarize(rng_key_binarize, train_fetch(i, train_idx)[0])
            svi_state, loss = svi.update(svi_state, batch)
            loss_sum += loss
            return loss_sum, svi_state

        return lax.fori_loop(0, num_train, body_fun, (0.0, svi_state))

    @jit
    def eval_test(svi_state: SVIState, rng_key: np.ndarray) -> jnp.ndarray:
        def body_fun(i: jnp.ndarray, loss_sum: jnp.ndarray) -> jnp.ndarray:
            rng_key_binarize = random.fold_in(rng_key, i)
            batch = binarize(rng_key_binarize, test_fetch(i, test_idx)[0])
            loss = svi.evaluate(svi_state, batch) / len(batch)
            loss_sum += loss
            return loss_sum

        loss = lax.fori_loop(0, num_test, body_fun, 0.0)
        loss = loss / num_test
        return loss

    def reconstruct_image(epoch: int, rng_key: np.ndarray) -> None:

        rng_key_binarize, rng_key_sample = random.split(rng_key)
        img = test_fetch(0, test_idx)[0][0]
        test_sample = binarize(rng_key_binarize, img)
        params = svi.get_params(svi_state)
        z_mean, z_var = encoder_nn[1](params["encoder$params"], test_sample.reshape([1, -1]))
        z = dist.Normal(z_mean, z_var).sample(rng_key_sample)
        img_loc = decoder_nn[1](params["decoder$params"], z).reshape([28, 28])

        plt.imsave(f"./data/original_{epoch}.png", img, cmap="gray")
        plt.imsave(f"./data/reconst_{epoch}.png", img_loc, cmap="gray")

    # Training
    for epoch in range(1, args.num_epochs + 1):
        rng_key, rng_key_train, rng_key_test, rng_key_reconstruct = random.split(rng_key, 4)

        num_train, train_idx = train_init()
        _, svi_state = epoch_train(svi_state, rng_key_train)

        num_test, test_idx = test_init()
        test_loss = eval_test(svi_state, rng_key_test)
        print(epoch, test_loss)

        if epoch % 5 == 0:
            reconstruct_image(epoch, rng_key_reconstruct)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=400)
    args = parser.parse_args()

    main(args)
