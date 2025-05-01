import os
import jax
import jax.numpy as jnp
import flax 
from flax import struct

@struct.dataclass
class AGEMMemory:
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    advantages: jnp.ndarray
    targets: jnp.ndarray
    values: jnp.ndarray


def make_task_onehot(task_idx: int, num_tasks: int) -> jnp.ndarray:
    """
    Returns a one-hot vector of length `num_tasks` with a 1 at `task_idx`.
    """
    return jnp.eye(num_tasks, dtype=jnp.float32)[task_idx]

def init_agem_memory(max_memory_size: int, obs_dim: int):
    return AGEMMemory(
        obs=jnp.zeros((max_memory_size, obs_dim)),
        actions=jnp.zeros((max_memory_size,)),
        log_probs=jnp.zeros((max_memory_size,)),
        advantages=jnp.zeros((max_memory_size,)),
        targets=jnp.zeros((max_memory_size,)),
        values=jnp.zeros((max_memory_size,)),
    )


def agem_project(grads_ppo, grads_mem):
    """
    Implements the AGEM projection:
      If g_new^T g_mem < 0:
         g_new := g_new - (g_new^T g_mem / ||g_mem||^2) * g_mem
    """
    dot_g = sum([jnp.vdot(gn, gm) for gn, gm in zip(jax.tree_util.tree_leaves(grads_ppo),
                                                    jax.tree_util.tree_leaves(grads_mem))])
    dot_mem = sum([jnp.vdot(gm, gm2) for gm, gm2 in zip(jax.tree_util.tree_leaves(grads_mem),
                                                        jax.tree_util.tree_leaves(grads_mem))])

    def project_fn(g_new):
        # Scale factor
        alpha = dot_g / (dot_mem + 1e-12)
        # Sub each leaf: g_new := g_new - alpha*g_mem
        return jax.tree_util.tree_map(lambda gn, gm: gn - alpha * gm, g_new, grads_mem)

    grads_projected = jax.lax.cond(
        dot_g < 0.0,
        project_fn,
        lambda x: x,
        grads_ppo
    )
    stats = {
        "agem/dot_g": dot_g,
        "agem/dot_mem": dot_mem,
        "agem/alpha": dot_g / (dot_mem + 1e-12),
        "agem/is_projected": (dot_g < 0.0),
    }
    return grads_projected, stats


def sample_memory(agem_mem: AGEMMemory, sample_size: int, rng: jax.random.PRNGKey):
    idxs = jax.random.randint(rng, (sample_size,), minval=0, maxval=agem_mem.obs.shape[0])
    obs = agem_mem.obs[idxs]
    actions = agem_mem.actions[idxs]
    log_probs = agem_mem.log_probs[idxs]
    advs = agem_mem.advantages[idxs]
    targets = agem_mem.targets[idxs]
    vals = agem_mem.values[idxs]
    return obs, actions, log_probs, advs, targets, vals


def compute_memory_gradient(network, params,
                            clip_eps, vf_coef, ent_coef,
                            mem_obs, mem_actions,
                            mem_advs, mem_log_probs,
                            mem_targets, mem_values):
    """Compute the same clipped PPO loss on the memory data."""

    def loss_fn(params):
        pi, value = network.apply(params, mem_obs)  # shapes: [B]
        log_prob = pi.log_prob(mem_actions)

        ratio = jnp.exp(log_prob - mem_log_probs)
        # standard advantage normalization
        adv_std = jnp.std(mem_advs) + 1e-8
        adv_mean = jnp.mean(mem_advs)
        normalized_adv = (mem_advs - adv_mean) / adv_std

        unclipped = ratio * normalized_adv
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_adv
        actor_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        # Critic Loss (same as normal PPO)
        value_pred_clipped = mem_values + (value - mem_values).clip(-clip_eps, clip_eps)
        value_losses = (value - mem_targets) ** 2
        value_losses_clipped = (value_pred_clipped - mem_targets) ** 2
        critic_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

        entropy = jnp.mean(pi.entropy())

        total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
        return total_loss, (critic_loss, actor_loss, entropy)

    (total_loss, (v_loss, a_loss, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    stats = {
        "agem/ppo_total_loss": total_loss,
        "agem/ppo_value_loss": v_loss,
        "agem/ppo_actor_loss": a_loss,
        "agem/ppo_entropy": ent
    }
    return grads, stats


def update_agem_memory(agem_mem: AGEMMemory,
                       new_obs, new_actions, new_log_probs,
                       new_advantages, new_targets, new_values,
                       max_memory_size: int) -> AGEMMemory:
    """
    Insert new transitions into our ring-buffer memory, all at once.
    `new_obs` shape: [B, obs_dim]
    The others shape: [B]
    We'll just do a simple "append and keep last max_memory_size".
    """
    # Concat
    obs_combined = jnp.concatenate([agem_mem.obs, new_obs], axis=0)
    actions_combined = jnp.concatenate([agem_mem.actions, new_actions], axis=0)
    log_probs_combined = jnp.concatenate([agem_mem.log_probs, new_log_probs], axis=0)
    advs_combined = jnp.concatenate([agem_mem.advantages, new_advantages], axis=0)
    targets_combined = jnp.concatenate([agem_mem.targets, new_targets], axis=0)
    vals_combined = jnp.concatenate([agem_mem.values, new_values], axis=0)

    total_len = obs_combined.shape[0]
    if total_len > max_memory_size:
        keep_slice = slice(total_len - max_memory_size, total_len)
        obs_combined = obs_combined[keep_slice]
        actions_combined = actions_combined[keep_slice]
        log_probs_combined = log_probs_combined[keep_slice]
        advs_combined = advs_combined[keep_slice]
        targets_combined = targets_combined[keep_slice]
        vals_combined = vals_combined[keep_slice]

    return AGEMMemory(
        obs=obs_combined,
        actions=actions_combined,
        log_probs=log_probs_combined,
        advantages=advs_combined,
        targets=targets_combined,
        values=vals_combined,
    )