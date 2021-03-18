"""Jax basic codes.

ref) https://qiita.com/koshian2/items/44a871386576b4f80aff
"""

import jax.numpy as jnp
from jax import jit, partial


@jit
def static_jax_dot() -> jnp.ndarray:

    x = jnp.arange(25, dtype=jnp.float32).reshape(5, 5)
    x_gram = jnp.dot(x, x.T)
    return x_gram


@partial(jit, static_argnums=(0,))
def variable_jax_dot(size: int) -> jnp.ndarray:

    x = jnp.arange(size ** 2, dtype=jnp.float32).reshape(size, size)
    x_gram = jnp.dot(x, x.T)
    return x_gram


if __name__ == "__main__":
    x = static_jax_dot()
    print(x.block_until_ready())

    x = variable_jax_dot(5)
    print(x.block_until_ready())
