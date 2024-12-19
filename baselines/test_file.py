# test file to test various things 


import os
import wandb
from dotenv import load_dotenv
import jax
from dataclasses import dataclass
from functools import partial


# test freezing the dataclass
@dataclass(frozen=True)
class Config:
    a : int = 1
    b : int = 2
    c : int = 0

@partial(jax.jit, static_argnums=(0,))
def func(config):
    a = config.a
    b = config.b
    ab = a + b
    print(ab)

def main():
    config = Config()
    func(config)

main()