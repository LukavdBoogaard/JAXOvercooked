import jax
import jax.numpy as jnp


def f(x):
    return x ** 2

f_vmap = jax.vmap(f)

x = jnp.array([1, 2, 3])
print(f_vmap(x))

