from jax import jit
from functools import partial

@jit
def f(x):
    return x * x

@jit
def g(x):
    return f(x)

print("cache size", f._cache_size())

print(g(3))  # 9
print(g(3))  # 9

print("cache size", f._cache_size())
