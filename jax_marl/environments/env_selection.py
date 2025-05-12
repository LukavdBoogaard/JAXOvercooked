import jax.numpy as jnp
import random
from flax.core.frozen_dict import FrozenDict
from jax_marl.environments.overcooked_environment import overcooked_layouts

# Define global layouts
AVAIL_LAYOUTS = overcooked_layouts

# def generate_sequence(sequence_length=None, strategy='random', layout_names=None, seed=None):
#     """
#     Generate a sequence of layouts for the agents to play on. 
#     """
    
#     if layout_names ==["hard_levels"]:
#         layout_names = ["forced_coord", "forced_coord_2", "split_kitchen", "basic_cooperative"]
#     elif layout_names == ["medium_levels"]:
#         layout_names = ["coord_ring", "efficiency_test", "split_work", "bottleneck_small", 
#                         "bottleneck_large", "counter_circuit", "corridor_challenge", "c_kitchen"]
        
#     if sequence_length is None:
#         sequence_length = len(layout_names) if layout_names is not None else len(AVAIL_LAYOUTS)

#     if layout_names is None:
#         layouts = []
#         for key, value in AVAIL_LAYOUTS.items():
#             layouts.append(key)
#     else:
#         layouts = layout_names
            
#     # Set seed if provided
#     if seed is not None:
#         random.seed(seed)
    
#     selected_layouts = []
    
#     if strategy == 'random':
#         if sequence_length <= len(layouts):
#             selected_layouts = random.sample(layouts, sequence_length)
#         else:
#             # Select layouts with replacement
#             available = layouts.copy()
#             last_selected = None
            
#             for _ in range(sequence_length):
#                 if len(available) == 1 and available[0] == last_selected:
#                     selected_layouts.append(available[0])
#                     last_selected = available[0]
#                 else:
#                     current_available = [l for l in available if l != last_selected]
#                     choice = random.choice(current_available)
#                     selected_layouts.append(choice)
#                     last_selected = choice
#     elif strategy == 'ordered':
#         if sequence_length <= len(layouts):
#             selected_layouts = layouts[:sequence_length]
#         else:
#             raise ValueError("Ordered strategy requires sequence_length <= number of available layouts")
#     else:
#         raise NotImplementedError("Strategy not implemented")
    
#     # print(selected_layouts)
#     env_kwargs = [{'layout': layout} for layout in selected_layouts]
#     layout_names = selected_layouts

#     # add a number to the layout name indicating the order in the sequence
#     for i, layout_name in enumerate(layout_names):
#         layout_names[i] = str(i) + "__" + layout_name

#     print("Selected layouts: ", layout_names)
#     return env_kwargs, layout_names
    


def generate_sequence(sequence_length=2, strategy='random', layout_names=None, seed=None):
    """
    Generate a sequence of layouts for the agents to play on. 
    """ 

    if layout_names is None:
        layouts = []
        for key, value in AVAIL_LAYOUTS.items():
            layouts.append(key)
    else:
        layouts = layout_names
            
    # Assert that the sequence length is smaller or equal to the layouts
    assert sequence_length <= len(layouts), "The sequence length is longer than the available layouts"
    
    if strategy == 'random':
        # set seed
        # if seed is not None:
        #     random.seed(seed)
        selected_layouts = random.sample(layouts, sequence_length)
        # reset seed
        random.seed(None)
    elif strategy == 'ordered':
        selected_layouts = layouts[:sequence_length]
    else:
        raise NotImplementedError("Strategy not implemented")
    
    # print(selected_layouts)
    env_kwargs = [{'layout': layout} for layout in selected_layouts]
    layout_names = selected_layouts

    # add a number to the layout name indicating the order in the sequence
    for i, layout_name in enumerate(layout_names):
        layout_names[i] = str(i) + "__" + layout_name

    print("Selected layouts: ", layout_names)
    return env_kwargs, layout_names
    