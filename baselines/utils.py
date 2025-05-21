import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
from flax.core.frozen_dict import FrozenDict
from matplotlib import pyplot as plt


class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation
    # info: jnp.ndarray # additional information


class Transition_CNN(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray  # whether the episode is done
    action: jnp.ndarray  # the action taken
    value: jnp.ndarray  # the value of the state
    reward: jnp.ndarray  # the reward received
    log_prob: jnp.ndarray  # the log probability of the action
    obs: jnp.ndarray  # the observation
    info: jnp.ndarray  # additional information


def batchify(x: dict, agent_list, num_actors, flatten=True):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param flatten: for MLP architectures
    @param x: dictionary of observations
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    batched = jnp.concatenate(x, axis=0)
    if flatten:
        batched = batched.reshape((num_actors, -1))
    return batched


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param unflatten: for MLP architectures
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def sample_discrete_action(key, action_space):
    """Samples a discrete action based on the action space provided."""
    num_actions = action_space.n
    return jax.random.randint(key, (1,), 0, num_actions)


def make_task_onehot(task_idx: int, num_tasks: int) -> jnp.ndarray:
    """
    Returns a one-hot vector of length `num_tasks` with a 1 at `task_idx`.
    """
    return jnp.eye(num_tasks, dtype=jnp.float32)[task_idx]


def copy_params(params):
    return jax.tree_util.tree_map(lambda x: x.copy(), params)


def compute_fwt(matrix):
    """
    Computes the forward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    # Assert that the matrix has the correct shape
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"

    num_tasks = matrix.shape[1]

    fwt_matrix = np.full((num_tasks, num_tasks), np.nan)

    for i in range(1, num_tasks):
        for j in range(i):  # j < i
            before_learning = matrix[0, i]
            after_task_j = matrix[j + 1, i]
            fwt_matrix[i, j] = after_task_j - before_learning

    return fwt_matrix


def compute_bwt(matrix):
    """
    Computes the backward transfer for all tasks in a sequence
    param matrix: a 2D array of shape (num_tasks + 1, num_tasks) where each entry is the performance of the model on the task
    """
    assert matrix.shape[0] == matrix.shape[1] + 1, "Matrix must have shape (num_tasks + 1, num_tasks)"
    num_tasks = matrix.shape[1]

    bwt_matrix = jnp.full((num_tasks, num_tasks), jnp.nan)

    for i in range(num_tasks - 1):
        for j in range(i + 1, num_tasks):
            after_j = matrix[j + 1, i]  # performance on task i after learning task j
            after_i = matrix[i + 1, i]  # performance on task i after learning task i
            bwt_matrix = bwt_matrix.at[i, j].set(after_j - after_i)

    return bwt_matrix


def show_heatmap_bwt(matrix, run_name, save_folder="heatmap_images"):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    bwt_matrix = compute_bwt(matrix)
    avg_bwt_per_step = np.nanmean(bwt_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(bwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                xticklabels=[f"Task {j}" for j in range(bwt_matrix.shape[1])],
                yticklabels=[f"Task {i}" for i in range(bwt_matrix.shape[0])],
                cbar_kws={"label": "BWT"})
    ax.set_title("Progressive Backward Transfer Matrix")
    ax.set_xlabel("Task B")
    ax.set_ylabel("Task A")
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    # Add average BWT per step below the heatmap
    for j, val in enumerate(avg_bwt_per_step):
        if not np.isnan(val):
            ax.text(j + 0.5, len(avg_bwt_per_step) + 0.2, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=9, color='black')
    plt.text(-0.7, len(avg_bwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

    plt.tight_layout()

    # Save the figure
    file_path = os.path.join(save_folder, f"{run_name}_bwt_heatmap.png")
    plt.savefig(file_path)
    plt.close()


def show_heatmap_fwt(matrix, run_name, save_folder="heatmap_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fwt_matrix = compute_fwt(matrix)
    avg_fwt_per_step = np.nanmean(fwt_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(fwt_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                xticklabels=[f"Task {j}" for j in range(fwt_matrix.shape[1])],
                yticklabels=[f"Task {i}" for i in range(fwt_matrix.shape[0])],
                cbar_kws={"label": "FWT"})
    ax.set_title("Progressive Forward Transfer Matrix")
    ax.set_xlabel("Task B")
    ax.set_ylabel("Task A")

    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    for j, val in enumerate(avg_fwt_per_step):
        if not np.isnan(val):
            ax.text(j + 0.5, len(avg_fwt_per_step) + 0.2, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=9, color='black')

    plt.text(-0.7, len(avg_fwt_per_step) + 0.2, "Avg", fontsize=10, va='bottom', weight='bold')

    plt.tight_layout()

    file_path = os.path.join(save_folder, f"{run_name}_fwt_heatmap.png")
    plt.savefig(file_path)
    plt.close()


def compute_normalized_evaluation_rewards(evaluations, layout_names, practical_baselines, metric):
    """Normalize evaluation rewards using JAX-friendly operations"""
    for i, layout_name in enumerate(layout_names):
        # Record raw evaluation score
        metric[f"Evaluation/{layout_name}"] = evaluations[i]

        # Extract the base layout name
        if "__" in layout_name:
            bare_layout = layout_name.split("__")[1]
        else:
            bare_layout = layout_name

        baseline_key = f"0__{bare_layout}"

        # Default to the unnormalized value
        normalized = evaluations[i]

        # Check if the baseline exists
        if baseline_key in practical_baselines:
            baseline = practical_baselines[baseline_key]["avg_rewards"]

            # Only divide when baseline > 0
            normalized = jnp.where(
                baseline > 0,
                evaluations[i] / baseline,
                1
            )

        metric[f"Scaled returns/evaluation_{layout_name}_scaled"] = normalized

    return metric


def compute_normalized_returns(layouts, practical_baselines, metric, env_counter):
    """
    Computes the normalized returns based on the practical baselines using JAX-friendly operations
    @param layouts: list of layouts
    @param practical_baselines: dictionary of practical baselines
    @param metric: dictionary to store the metrics
    @param env_counter: counter for the environment
    """
    env_name = layouts[env_counter - 1]
    bare_layout = env_name.split("__")[1].strip()
    baseline_format = f"0__{bare_layout}"

    # Default to unnormalized value
    normalized = metric["returned_episode_returns"]

    if baseline_format in practical_baselines:
        baseline_result = practical_baselines[baseline_format]["avg_rewards"]

        normalized = jnp.where(
            baseline_result > 0,
            metric["returned_episode_returns"] / baseline_result,
            1
        )

    metric["Scaled returns/returned_episode_returns_scaled"] = normalized

    return metric


def build_reg_weights(params, regularize_critic: bool, regularize_heads: bool) -> FrozenDict:
    def _mark(path, x):
        path_str = "/".join(map(str, path)).lower()
        if not regularize_heads and ("actor_head" in path_str or "critic_head" in path_str):
            return jnp.zeros_like(x)
        if not regularize_critic and "critic" in path_str:
            return jnp.zeros_like(x)
        return jnp.ones_like(x)

    return jax.tree_util.tree_map_with_path(_mark, params)


# -------------------------------------------------------------------
# helper: pad (or crop) an (H,W,C) image to `target_shape`
#         – no tracers in pad_width, 100 % JIT-safe
# -------------------------------------------------------------------
def _pad_to(img: jnp.ndarray, target_shape):
    th, tw, tc = target_shape  # target (height, width, channels)
    h, w, c = img.shape  # current shape – *Python* ints
    assert c == tc, "channel mismatch"

    dh = th - h  # + ⇒ need pad, − ⇒ need crop
    dw = tw - w

    # amounts have to be Python ints so jnp.pad sees concrete values
    pad_top = max(dh // 2, 0)
    pad_bottom = max(dh - pad_top, 0)
    pad_left = max(dw // 2, 0)
    pad_right = max(dw - pad_left, 0)

    img = jnp.pad(
        img,
        ((pad_top, pad_bottom),
         (pad_left, pad_right),
         (0, 0)),  # no channel padding
        mode="constant",
    )

    # If the image was *larger* than the target we crop back
    return img[:th, :tw, :]


# ---------------------------------------------------------------
# util: build a (2, …) batch without Python branches
# ---------------------------------------------------------------
def _prep_obs(raw_obs, expected_shape, use_cnn):
    """
    Build a (2, …) batch: one row per agent.
    If `use_task_id` is False we *still* append a
    zero-vector of length `seq_len` so that both
    branches have the same dtype & length.
    """

    def _single(img):  # img: (H,W,C)
        if use_cnn:
            img = _pad_to(img, expected_shape)
            return img[None]  # (1, H, W, C)
        else:
            # ---------- MLP branch ------------------------------------------
            vec = img.reshape(-1).astype(jnp.float32)
            return vec[None]  # (1, total_len)

    return jnp.concatenate(
        [_single(raw_obs["agent_0"]),
         _single(raw_obs["agent_1"])],
        axis=0)  # (2, …)
