import jax.numpy as jnp
import random
from flax.core.frozen_dict import FrozenDict
from jax_marl.environments.overcooked_environment import overcooked_layouts

# Define global layouts
AVAIL_LAYOUTS = overcooked_layouts

def generate_sequence(sequence_length=2, strategy='random', layouts=None):
    """
    Generate a sequence of layouts for the agents to play on. 
    """

    if layouts is None:
        layouts = []
        for key, value in AVAIL_LAYOUTS.items():
            layouts.append(key)
    
    if strategy == 'random':
        selected_layouts = random.sample(layouts, sequence_length)
    elif strategy == 'ordered':
        selected_layouts = layouts[:sequence_length]
    else:
        raise NotImplementedError("Strategy not implemented")
    
    print(selected_layouts)
    env_kwargs = [{'layout': layout} for layout in selected_layouts]
    layout_names = selected_layouts

    return env_kwargs, layout_names
    