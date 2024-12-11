import jax
import jax.numpy as jnp

def make_func():
    def func(x):
        return x ** 2
    return func


def main():
    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(0)
        jitted_func = jax.jit(make_func())
        print(jitted_func(2))
        print(jitted_func._cache_size())
        print(jitted_func(jnp.array(2)))
        print(jitted_func._cache_size())

if __name__ == "__main__":
    main()
