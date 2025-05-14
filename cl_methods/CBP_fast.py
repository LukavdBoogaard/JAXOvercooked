# Continuous Backprop with fast, JAX‑friendly maintenance ---------------------------------
# Drop‑in replacement for the previous CBP implementation.
# ‑ avoids unfreeze/freeze (host round‑trips)
# ‑ regenerates only the rows that are actually replaced
# ‑ ready for JIT and TPU/GPU execution

import flax
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState


# ───────────────── helper: re‑initialise selected rows ──────────────────────

def _reinit_rows(rng, kernel, row_mask):
    """Re‑sample *all* rows (constant shape) and keep the ones flagged in `row_mask`."""
    rng, sub = jax.random.split(rng)
    new_kernel = jax.nn.initializers.orthogonal(jnp.sqrt(2.0))(sub, kernel.shape)
    return jnp.where(row_mask[None, :], new_kernel, kernel)


# ───────────────── core maintenance step ────────────────────────────────────

def _update_layer(p_layer, c_layer, next_kernel, rng, *, prefix, maturity, rho):
    """Update one hidden layer (actor_fc1_d etc.)."""
    util = c_layer[f"{prefix}_util"]
    age = c_layer[f"{prefix}_age"]
    ctr = c_layer[f"{prefix}_ctr"]

    mature = age > maturity
    ctr += mature.sum() * rho
    n_rep = jnp.floor(ctr).astype(jnp.int32)
    ctr -= n_rep

    util_mask = jnp.where(mature, util, jnp.inf)
    rank = jnp.argsort(util_mask)
    rep_mask = (rank < n_rep) & mature

    rng, sub = jax.random.split(rng)
    k_in = _reinit_rows(sub, p_layer["kernel"], rep_mask)
    b_in = jnp.where(rep_mask, 0.0, p_layer["bias"])
    k_out = jnp.where(rep_mask[:, None], 0.0, next_kernel)

    new_p_layer = {"kernel": k_in, "bias": b_in}
    new_c_layer = {
        f"{prefix}_util": jnp.where(rep_mask, 0.0, util),
        f"{prefix}_age": jnp.where(rep_mask, 0, age),
        f"{prefix}_ctr": ctr,
    }
    return new_p_layer, new_c_layer, k_out, rng


def cbp_step_fast(params: FrozenDict,
                  cbp_state: FrozenDict,
                  rng: jax.random.PRNGKey,
                  *,
                  maturity: int,
                  rho: float):
    """Shape‑stable CBP update that returns *new* FrozenDicts."""
    # --- actor fc1 → fc2 ----------------------------------------------------
    p1, c1, k1, rng = _update_layer(
        params["actor_fc1"]["actor_fc1_d"],
        cbp_state["actor_fc1"],
        params["actor_fc2"]["actor_fc2_d"]["kernel"],
        rng,
        prefix="actor_fc1",
        maturity=maturity,
        rho=rho,
    )
    # --- actor fc2 → out ----------------------------------------------------
    p2, c2, k2, rng = _update_layer(
        params["actor_fc2"]["actor_fc2_d"],
        cbp_state["actor_fc2"],
        params["actor_out"]["kernel"],
        rng,
        prefix="actor_fc2",
        maturity=maturity,
        rho=rho,
    )
    # --- critic fc1 → fc2 ---------------------------------------------------
    p3, c3, k3, rng = _update_layer(
        params["critic_fc1"]["critic_fc1_d"],
        cbp_state["critic_fc1"],
        params["critic_fc2"]["critic_fc2_d"]["kernel"],
        rng,
        prefix="critic_fc1",
        maturity=maturity,
        rho=rho,
    )
    # --- critic fc2 → out ---------------------------------------------------
    p4, c4, _, rng = _update_layer(
        params["critic_fc2"]["critic_fc2_d"],
        cbp_state["critic_fc2"],
        params["critic_out"]["kernel"],
        rng,
        prefix="critic_fc2",
        maturity=maturity,
        rho=rho,
    )

    # FrozenDicts are immutable – convert to mutable dicts, patch, then freeze.
    mutable_p = flax.core.unfreeze(params)
    mutable_cb = flax.core.unfreeze(cbp_state)

    mutable_p["actor_fc1"]["actor_fc1_d"] = p1
    mutable_p["actor_fc2"]["actor_fc2_d"] = p2
    mutable_p["critic_fc1"]["critic_fc1_d"] = p3
    mutable_p["critic_fc2"]["critic_fc2_d"] = p4

    mutable_cb["actor_fc1"] = c1
    mutable_cb["actor_fc2"] = c2
    mutable_cb["critic_fc1"] = c3
    mutable_cb["critic_fc2"] = c4

    new_params = flax.core.freeze(mutable_p)
    new_cbp = flax.core.freeze(mutable_cb)
    return new_params, new_cbp, rng


class TrainStateCBP(TrainState):
    """TrainState with an *extra* collection that stores CBP variables."""
    cbp_state: FrozenDict
