from flax.core.frozen_dict import FrozenDict
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax.tree_util import tree_flatten, tree_unflatten


cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "goal_idx" : jnp.array([18]),
    "plate_pile_idx" : jnp.array([16]),
    "onion_pile_idx" : jnp.array([5,9]),
    "pot_idx" : jnp.array([2])
}
asymm_advantages = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9,14]),
    "pot_idx" : jnp.array([22,31])
}
coord_ring = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,12,14,
                            15,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([7, 11]),
    "goal_idx" : jnp.array([22]),
    "plate_pile_idx" : jnp.array([10]),
    "onion_pile_idx" : jnp.array([15,21]),
    "pot_idx" : jnp.array([3,9])
}
forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,7,9,
                            10,12,14,
                            15,17,19,
                            20,21,22,23,24]),
    "agent_idx" : jnp.array([11,8]),
    "goal_idx" : jnp.array([23]),
    "onion_pile_idx" : jnp.array([5,10]),
    "plate_pile_idx" : jnp.array([15]),
    "pot_idx" : jnp.array([3,9])
}

l = [FrozenDict(cramped_room)]
print(l)

(l_flattened, pytree) = tree_flatten(l)   
print(l_flattened)

def train(carry, x):
    l_flattened, pytree = x
    tree_unflattened = tree_unflatten(pytree, l_flattened)
    print(tree_unflattened)
    return carry + 1, pytree


a, b = jax.lax.scan(f=train, init=0, xs=(l_flattened, pytree))

print(a, b)