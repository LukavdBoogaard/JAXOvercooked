import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax import struct
from baselines.utils import make_task_onehot, copy_params

@struct.dataclass
class EWCState:
    old_params: FrozenDict
    fisher: FrozenDict
    reg_weights: FrozenDict  # a mask: ones for parameters to regularize, zeros otherwise


def init_cl_state(params: FrozenDict, regularize_critic: bool, regularize_heads: bool) -> EWCState:
    """Initialize old_params with the current parameters, fisher with zeros."""
    old_params = copy_params(params)
    reg_weights = build_reg_weights(params, regularize_critic, regularize_heads)
    fisher = jax.tree_map(lambda x: jnp.zeros_like(x), old_params)
    return EWCState(old_params=old_params, fisher=fisher, reg_weights=reg_weights)


def update_ewc_state(ewc_state: EWCState,
                     new_params: FrozenDict,
                     new_fisher: FrozenDict
                     ) -> EWCState:
    """Append the new snapshot of parameters and its Fisher to the lists."""
    return EWCState(
        old_params=copy_params(new_params),
        fisher=new_fisher,
        reg_weights=ewc_state.reg_weights
    )


def build_reg_weights(params: FrozenDict, regularize_critic: bool = False, regularize_heads: bool = True) -> FrozenDict:
    def _assign_reg_weight(path, x):
        # Join the keys in the path to a string.
        path_str = "/".join(str(key) for key in path)
        # Exclude head parameters: do not regularize if parameter is in actor_head or critic_head.
        if not regularize_heads:
            if "actor_head" in path_str or "critic_head" in path_str:
                return jnp.zeros_like(x)
        # If we're not regularizing the critic, then exclude any parameter from critic branches.
        if not regularize_critic and "critic" in path_str.lower():
            return jnp.zeros_like(x)
        # Otherwise, regularize (the trunk).
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_assign_reg_weight, params)


# @partial(jax.jit, static_argnums=(1,3))
def compute_fisher_with_rollouts(
        config,
        train_state,
        env,
        network,
        env_idx=0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        max_episodes=5,
        max_steps=500,
        normalize_fisher=False
):
    """
    Perform up to `max_episodes` rollouts (each up to `max_steps` steps),
    computing the diagonal Fisher approximation by accumulating:
       E[ grad(log π(a|s))^2 ]
    for all environment steps across all agents.

    Args:
      config: your config containing use_task_id, seq_length, etc.
      train_state: the current model (params + optimizer).
      env: the environment (multi-agent or single-agent).
      network: your ActorCritic module with apply_fn = network.apply.
      env_idx: which task index (for appending task ID if use_task_id == True).
      key: random key.
      max_episodes: how many episodes to roll out for the Fisher estimate.
      max_steps: max steps per episode.
      normalize_fisher: optionally rescale the Fisher to have average magnitude ~ 1.

    Returns:
      fisher_accum: a FrozenDict with the same structure as params,
                    containing the average of grad(log π)^2 across episodes/steps.
    """
    # Initialize fisher_accum to zeros matching shape of your parameters:
    fisher_accum_init = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x),
        train_state.params
    )

    def actor_log_prob_sum(params, obs_dict, actions_dict):
        """
        Sums the log-probs of each agent’s chosen action.
        This is what we'll differentiate to get a diagonal Fisher approximation.
        """
        total_lp = 0.0
        # For each agent in obs_dict, compute log-prob of the action
        for agent_id in obs_dict.keys():
            pi, _ = network.apply(params, obs_dict[agent_id], env_idx=env_idx)
            # shape might be (1,) so we sum or squeeze
            log_p = pi.log_prob(actions_dict[agent_id])
            total_lp += jnp.sum(log_p)  # or just log_p if it's scalar
        return total_lp

    def single_episode_fisher(rng_ep, fisher_accum):
        """
        Runs one rollout (up to max_steps).
        Each step we accumulate grad(log π(a|s))^2 for all agents.
        """
        rng, rng_reset = jax.random.split(rng_ep)
        obs, state = env.reset(rng_reset)
        done = False
        step_count = 0

        while (not done) and (step_count < max_steps):
            # Prepare each agent’s observation as a (1, obs_dim) batch
            # This part is copied from your record_gif_of_episode, minus the 'states' tracking:
            flat_obs = {}
            for agent_id, obs_v in obs.items():
                expected_shape = env.observation_space().shape
                if obs_v.ndim == len(expected_shape):
                    obs_b = jnp.expand_dims(obs_v, axis=0)  # (1, ...)

                if not config.use_cnn:
                    obs_b = jnp.reshape(obs_b, (obs_b.shape[0], -1))
                    if config.use_task_id:
                        onehot = make_task_onehot(env_idx, config.seq_length)
                        onehot = jnp.expand_dims(onehot, axis=0)  # shape (1, seq_length)
                        obs_b = jnp.concatenate([obs_b, onehot], axis=1)
                flat_obs[agent_id] = obs_b

            # Sample an action for each agent
            act_keys = jax.random.split(rng, env.num_agents)
            rng, rng_step = jax.random.split(rng)

            actions = {}
            for i, agent_id in enumerate(env.agents):
                pi, _ = network.apply(train_state.params, flat_obs[agent_id], env_idx=env_idx)
                sampled_action = jnp.squeeze(pi.sample(seed=act_keys[i]), axis=0)
                actions[agent_id] = sampled_action

            # Compute grad of sum of log-probs wrt params
            def _grad_log_prob(p_):
                return actor_log_prob_sum(p_, flat_obs, actions)

            grads = jax.grad(_grad_log_prob)(train_state.params)
            # Square them for the diagonal Fisher approximation
            grad_sqr = jax.tree_util.tree_map(lambda g: g ** 2, grads)

            # Accumulate
            fisher_accum = jax.tree_util.tree_map(
                lambda fa, gs: fa + gs,
                fisher_accum, grad_sqr
            )

            # Step environment
            next_obs, next_state, reward, done_info, _info = env.step(rng_step, state, actions)
            done = done_info["__all__"]

            obs, state = next_obs, next_state
            step_count += 1

        return fisher_accum, step_count

    # Main loop over multiple episodes
    fisher_accum = fisher_accum_init
    total_steps = 0
    rngs = jax.random.split(key, max_episodes)

    for ep_i in range(max_episodes):
        fisher_accum, ep_steps = single_episode_fisher(rngs[ep_i], fisher_accum)
        total_steps += ep_steps

    # Average over the total number of environment steps
    # so we get E[ grad^2 ] rather than a sum.
    if total_steps > 0:
        fisher_accum = jax.tree_util.tree_map(
            lambda x: x / float(total_steps),
            fisher_accum
        )

    # Optional normalization so the average magnitude is ~1
    if normalize_fisher and total_steps > 0:
        total_abs = jax.tree_util.tree_reduce(
            lambda acc, x: acc + jnp.sum(jnp.abs(x)),
            fisher_accum,
            0.0
        )
        param_count = jax.tree_util.tree_reduce(
            lambda acc, x: acc + x.size,
            fisher_accum,
            0
        )
        fisher_mean = total_abs / (param_count + 1e-8)
        fisher_accum = jax.tree_util.tree_map(
            lambda x: x / (fisher_mean + 1e-8),
            fisher_accum
        )

    return fisher_accum


def compute_ewc_loss(params: FrozenDict,
                     ewc_state: EWCState,
                     ewc_coef: float
                     ) -> float:
    """
    Compute EWC penalty: 0.5 * ewc_coef * sum( fisher * (params - old_params)^2 )
    """

    def penalty(p, old_p, f, w):
        return w * f * (p - old_p) ** 2

    ewc_term_tree = jax.tree_util.tree_map(lambda p_, op_, ff_, w: penalty(p_, op_, ff_, w),
                                           params, ewc_state.old_params, ewc_state.fisher, ewc_state.reg_weights)
    ewc_term = jax.tree_util.tree_reduce(lambda acc, x: acc + x.sum(), ewc_term_tree, 0.0)
    return 0.5 * ewc_coef * ewc_term

