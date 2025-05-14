from typing import Tuple

import flax.linen as nn
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


def weight_reinit(key, shape):
    """Should be the same weight initializer used at model start (orthogonal(√2))."""
    return orthogonal(np.sqrt(2))(key, shape)


def cbp_step(
        params: FrozenDict,
        cbp_state: FrozenDict,
        *,
        rng: jax.random.PRNGKey,
        maturity: int,
        rho: float,  # replacement rate (0.0 - 1.0)
) -> Tuple[FrozenDict, FrozenDict, jax.random.PRNGKey]:
    """
    Pure JAX function: one CBP maintenance step.
    * increments replacement counter for every layer
    * performs (possibly zero) replacements based on counter
    """
    p, s = unfreeze(params), unfreeze(cbp_state)  # python dicts (outside jit)

    def _layer(layer_name, next_layer_name, rng):
        next_layer_no_d = next_layer_name.split("_d")[0]  # remove "_d" from name
        util = s[layer_name][f"{layer_name}_util"]
        age = s[layer_name][f"{layer_name}_age"]
        ctr = s[layer_name][f"{layer_name}_ctr"]
        mature_mask = age > maturity
        n_eligible = jnp.sum(mature_mask).astype(int)
        ctr += n_eligible * rho

        # number of whole neurons to replace
        n_rep = jnp.floor(ctr).astype(int)
        ctr -= jnp.float32(n_rep)
        s[layer_name][f"{layer_name}_ctr"] = ctr

        util_masked = jnp.where(mature_mask, util, jnp.inf)
        sorted_idx = jnp.argsort(util_masked)  # indices ascending by util
        # build inverse permutation to get each index’s rank
        ranks = jnp.zeros_like(sorted_idx).at[sorted_idx].set(jnp.arange(util.shape[0]))
        rep_mask = (ranks < n_rep) & mature_mask  # boolean mask of size n_units

        W_in = p[layer_name][f"{layer_name}_d"]["kernel"]
        b_in = p[layer_name][f"{layer_name}_d"]["bias"]
        if '_out' in next_layer_name:  # handle last layer (actor_out, critic_out) differently
            W_out = p[next_layer_name]["kernel"]
        else:
            W_out = p[next_layer_no_d][next_layer_name]["kernel"]
        rng, k_init = jax.random.split(rng)
        new_W = weight_reinit(k_init, W_in.shape)

        # apply mask
        W_in = jnp.where(rep_mask[None, :], new_W, W_in)
        b_in = jnp.where(rep_mask, 0.0, b_in)
        W_out = jnp.where(rep_mask[:, None], 0.0, W_out)

        # reset bookkeeping
        util = jnp.where(rep_mask, 0.0, util)
        age = jnp.where(rep_mask, 0, age)

        # write back
        p[layer_name][f"{layer_name}_d"]["kernel"] = W_in
        p[layer_name][f"{layer_name}_d"]["bias"] = b_in
        if '_out' in next_layer_name:
            p[next_layer_name]["kernel"] = W_out
        else:
            p[next_layer_no_d][next_layer_name]["kernel"] = W_out
        s[layer_name][f"{layer_name}_util"] = util
        s[layer_name][f"{layer_name}_age"] = age

        return rng

    rng = _layer("actor_fc1", "actor_fc2_d", rng)
    rng = _layer("actor_fc2", "actor_out", rng)
    rng = _layer("critic_fc1", "critic_fc2_d", rng)
    rng = _layer("critic_fc2", "critic_out", rng)

    return freeze(p), freeze(s), rng


class TrainStateCBP(TrainState):
    """TrainState with an *extra* collection that stores CBP variables."""
    cbp_state: FrozenDict
