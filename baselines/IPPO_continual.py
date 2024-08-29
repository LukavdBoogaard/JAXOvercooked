""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jax_marl
from jax_marl.wrappers.baselines import LogWrapper
from jax_marl.environments.overcooked_environment import overcooked_layouts
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import wandb

# Set the global config variable
config = None

class ActorCritic(nn.Module):
    '''
    Class to define the actor-critic networks used in IPPO. Each agent has its own actor-critic network
    '''
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # ACTOR  
        actor_mean = nn.Dense(
            64, # number of neurons
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0) # sets the bias initialization to a constant value of 0
        )(x) # applies a dense layer to the input x

        actor_mean = activation(actor_mean) # applies the activation function to the output of the dense layer

        actor_mean = nn.Dense(
            64, 
            kernel_init=orthogonal(np.sqrt(2)), 
            bias_init=constant(0.0)
        )(actor_mean)

        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean) # creates a categorical distribution over all actions (the logits are the output of the actor network)

        # CRITIC
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)

        critic = activation(critic)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)

        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        # returns the policy (actor) and state-value (critic) networks
        value = jnp.squeeze(critic, axis=-1)
        return pi, value #squeezed to remove any unnecessary dimensions
    

class Transition(NamedTuple):
    '''
    Named tuple to store the transition information
    '''
    done: jnp.ndarray # whether the episode is done
    action: jnp.ndarray # the action taken
    value: jnp.ndarray # the value of the state
    reward: jnp.ndarray # the reward received
    log_prob: jnp.ndarray # the log probability of the action
    obs: jnp.ndarray # the observation
    # info: jnp.ndarray # additional information

############################
##### HELPER FUNCTIONS #####
############################
import jax
import jax.numpy as jnp
import wandb

def evaluate_model(train_state, config, key):
    '''
    Evaluates the model by running 10 episodes on all environments and returns the average reward
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the average reward
    '''

    def run_episode_while(env, key_r, network, network_params, max_steps=1000):
        """
        Run a single episode using jax.lax.while_loop for dynamic episode lengths.
        """
        class LoopState(NamedTuple):
            key: Any
            state: Any
            obs: Any
            done: bool
            total_reward: float
            step_count: int

        def loop_cond(state: LoopState):
            return jnp.logical_and(~state.done, state.step_count < max_steps)

        def loop_body(state: LoopState):
            key, state_env, obs, _, total_reward, step_count = state
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)

            # Flatten observations
            flat_obs = {k: v.flatten() for k, v in obs.items()}

            # Get action distributions
            pi_0, _ = network.apply(network_params, flat_obs["agent_0"])
            pi_1, _ = network.apply(network_params, flat_obs["agent_1"])

            # Sample actions
            actions = {
                "agent_0": pi_0.sample(seed=key_a0),
                "agent_1": pi_1.sample(seed=key_a1)
            }

            # Environment step
            next_obs, next_state, reward, done_step, info = env.step(key_s, state_env, actions)
            done = done_step["__all__"]
            reward = reward["agent_0"]  # Adjust as needed
            total_reward += reward
            step_count += 1

            return LoopState(key, next_state, next_obs, done, total_reward, step_count)

        # Initialize
        key, key_s = jax.random.split(key_r)
        obs, state = env.reset(key_s)
        init_state = LoopState(key, state, obs, False, 0.0, 0)

        # Run while loop
        final_state = jax.lax.while_loop(
            loop_cond,
            loop_body,
            init_state
        )

        return final_state.total_reward

    # Loop through all environments
    all_avg_rewards = []
    for env_args in config["ENV_KWARGS"]:
        env_name = env_args["layout"]  # Extract the layout name
        env = jax_marl.make(config["ENV_NAME"], **env_args)  # Create the environment

        # Initialize the network
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        key, key_a = jax.random.split(key)
        init_x = jnp.zeros(env.observation_space().shape).flatten()  # initializes and flattens observation space

        network.init(key_a, init_x)  # initializes the network with the observation space
        network_params = train_state.params

        # Run 10 episodes
        all_rewards = jax.vmap(lambda k: run_episode_while(env, k, network, network_params, 500))(
            jax.random.split(key, 5)
        )
        
        avg_reward = jnp.mean(all_rewards)
        all_avg_rewards.append(avg_reward)

        # # Log the results to wandb
        # def callback(avg_reward):
        #     wandb.log({
        #         f"{env_name}_reward": avg_reward
        #     })

        # jax.debug.callback(callback, avg_reward)
    return all_avg_rewards


def evaluate_models(train_state, config):
    '''
    Evaluates the model by running 10 episodes on all environments and returns the average reward
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the average reward
    '''

    # Loop through all environments
    for env_args in config["ENV_KWARGS"]:
        env = jax_marl.make(config["ENV_NAME"], **env_args)  # Create the environment
        env_name = env_args["layout"]  # Extract the layout name

        # Initialize the network
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        key = jax.random.PRNGKey(0)
        key, key_r, key_a = jax.random.split(key, 3)
        init_x = jnp.zeros(env.observation_space().shape) # initializes the observation space to zeros
        init_x = init_x.flatten() # flattens the observation space to a 1D array

        network.init(key_a, init_x) # initializes the network with the observation space
        network_params = train_state.params 

        all_rewards = []  # Initialize a list to store the rewards

        # run 10 episodes
        for _ in range(10):
            done = False
            obs, state = env.reset(key_r)
            state_seq = [state]
            rewards = []
            shaped_rewards = []

            while not done:
                key, key_a0, key_a1, key_s = jax.random.split(key, 4)

                obs = {k: v.flatten() for k, v in obs.items()}

                pi_0, _ = network.apply(network_params, obs["agent_0"])
                pi_1, _ = network.apply(network_params, obs["agent_1"])

                actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}

                obs, state, reward, done, info = env.step(key_s, state, actions)
                done = done["__all__"]
                rewards.append(reward["agent_0"])
                shaped_rewards.append(info["shaped_reward"]["agent_0"])

                state_seq.append(state)

            total_rewards = np.sum(rewards)
            all_rewards.append(total_rewards)
        
        avg_reward = np.mean(all_rewards) 


        # log the results to wandb
        wandb.log({
            f"{env_name}_reward": avg_reward
        }) 

    return None

def get_rollout(train_state, config):
    '''
    Simulates the environment using the network
    @param train_state: the current state of the training
    @param config: the configuration of the training
    returns the state sequence
    '''

    for env_args in config["ENV_KWARGS"]:
        # Create the environment
        env = jax_marl.make(config["ENV_NAME"], **env_args)
        env_name = env_args["layout"]


        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) # Sets up the network
        key = jax.random.PRNGKey(0) 
        key, key_r, key_a = jax.random.split(key, 3) 

        init_x = jnp.zeros(env.observation_space().shape) # initializes the observation space to zeros
        init_x = init_x.flatten() # flattens the observation space to a 1D array

        network.init(key_a, init_x) # initializes the network with the observation space
        network_params = train_state.params 

        done = False

        obs, state = env.reset(key_r)
        state_seq = [state]
        rewards = []
        shaped_rewards = []
        while not done:
            key, key_a0, key_a1, key_s = jax.random.split(key, 4)

            # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
            # breakpoint()
            obs = {k: v.flatten() for k, v in obs.items()}

            pi_0, _ = network.apply(network_params, obs["agent_0"])
            pi_1, _ = network.apply(network_params, obs["agent_1"])

            actions = {"agent_0": pi_0.sample(seed=key_a0), "agent_1": pi_1.sample(seed=key_a1)}
            # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
            # env_act = {k: v.flatten() for k, v in env_act.items()}

            # STEP ENV
            obs, state, reward, done, info = env.step(key_s, state, actions)
            done = done["__all__"]
            rewards.append(reward["agent_0"])
            shaped_rewards.append(info["shaped_reward"]["agent_0"])

            state_seq.append(state)
        
        from matplotlib import pyplot as plt
        plt.plot(rewards, label="reward")
        plt.plot(shaped_rewards, label="shaped_reward")
        plt.xlabel("Time Steps")
        plt.ylabel("Reward Value")
        plt.title("Rewards over Time")
        plt.legend()
        plt.savefig("reward_coord_ring.png")

    return state_seq

def run_evaluation(train_state, config):
    '''
    runs an evaluation on a list of different overcooked maps and returns the average reward after 10 runs

    @param train_state: the current state of the training
    @param config: the configuration taken from the yaml file
    returns the average reward for each map
    '''
    results = {}
    maps = config["MAPS"]
    num_rollouts = config["EVALUATION"]["num_rollouts"]
    
    for map_name in maps:
        config["ENV_KWARGS"]["layout"] = map_name  # Set the environment layout to the current map
        all_rewards = []
        
        for _ in range(num_rollouts):
            state_seq = get_rollout(train_state, config)  # Run a rollout
            rewards = [step["reward"] for step in state_seq]  # Extract rewards from the rollout
            total_reward = np.sum(rewards)  # Sum the rewards to get the total reward for this rollout
            all_rewards.append(total_reward)  # Store the total reward
        
        avg_reward = np.mean(all_rewards)  # Compute the average reward across all rollouts for this map
        results[map_name] = avg_reward  # Store the average reward in the results dictionary
    
    return results 

def batchify(x: dict, agent_list, num_actors):
    '''
    converts the observations of a batch of agents into an array of size (num_actors, -1) that can be used by the network
    @param x: dictionary of observations
    @param agent_list: list of agents
    @param num_actors: number of actors
    returns the batchified observations
    '''
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    '''
    converts the array of size (num_actors, -1) into a dictionary of observations for all agents
    @param x: array of observations
    @param agent_list: list of agents
    @param num_envs: number of environments
    @param num_actors: number of actors
    returns the unbatchified observations
    '''
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


##########################################################
##########################################################
#######            TRAINING FUNCTION               #######
##########################################################
##########################################################

def make_train(config):
    '''
    Creates a 'train' function that trains the network using PPO
    @param config: the configuration of the algorithm and environment
    returns the training function
    '''
    def train(rng):

        # step 1: loop through all the environments and 'create' them (i.e. transform from a string to an object)
        envs = []
        for env_args in config["ENV_KWARGS"]:
            # Create the environment
            env = jax_marl.make(config["ENV_NAME"], **env_args)
            env = LogWrapper(env, replace_info=False)
            envs.append(env)

        def linear_schedule(count):
            '''
            Linearly decays the learning rate depending on the number of minibatches and number of epochs
            returns the learning rate
            '''
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac
        
        # step 6: set the extra config parameters based on the environment
        # set extra config parameters based on the environment
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])


        # REWARD SHAPING IN NEW VERSION
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.,
            end_value=0.,
            transition_steps=config["REWARD_SHAPING_HORIZON"]
        )

        # step 2: initialize the network using the first environment
        temp_env = envs[0]
        network = ActorCritic(temp_env.action_space().n, activation=config["ACTIVATION"])

        # step 3: initialize the network parameters
        rng, network_rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape).flatten()
        network_params = network.init(network_rng, init_x)
          # step 4: initialize the optimizer
        if config["ANNEAL_LR"]: 
            # anneals the learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            # uses the default learning rate
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
                optax.adam(config["LR"], eps=1e-5)
            )

        # step 5: Initialize the training state      
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        def train_on_environment(rng, train_state, env):
            '''
            Trains the network using IPPO
            @param rng: random number generator 
            returns the runner state and the metrics
            '''
            
            network = ActorCritic(temp_env.action_space().n, activation=config["ACTIVATION"])

            # step 3: initialize the network parameters
            rng, network_rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space().shape).flatten()
            network_params = network.init(network_rng, init_x)
            # step 4: initialize the optimizer
            if config["ANNEAL_LR"]: 
                # anneals the learning rate
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                # uses the default learning rate
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
                    optax.adam(config["LR"], eps=1e-5)
                )

            # step 5: Initialize the training state      
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

            # # reset the learning rate and the optimizer
            # if config["ANNEAL_LR"]:
            #     tx = optax.chain(
            #         optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            #         optax.adam(learning_rate=linear_schedule, eps=1e-5),
            #     )
            # else:
            #     tx = optax.chain(
            #         optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            #         optax.adam(config["LR"], eps=1e-5)
            #     )
            
            # train_state = train_state.replace(tx=tx)
            
            # Initialize environment 
            rng, env_rng = jax.random.split(rng) 

            # create config["NUM_ENVS"] seeds for each environment 
            reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) 

            # create and reset the environment
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) 
            
            # TRAIN LOOP
            def _update_step(runner_state, unused):
                '''
                perform a single update step in the training loop
                @param runner_state: the carry state that contains all important training information
                returns the updated runner state and the metrics 
                '''

                # COLLECT TRAJECTORIES
                def _env_step(runner_state, unused):
                    '''
                    selects an action based on the policy, calculates the log probability of the action, 
                    and performs the selected action in the environment
                    @param runner_state: the current state of the runner
                    returns the updated runner state and the transition
                    '''
                    # Unpack the runner state
                    train_state, env_state, last_obs, update_step, rng = runner_state

                    # SELECT ACTION
                    # split the random number generator for action selection
                    rng, _rng = jax.random.split(rng)

                    # prepare the observations for the network
                    obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                    print("obs_shape", obs_batch.shape)
                    
                    # apply the policy network to the observations to get the suggested actions and their values
                    pi, value = network.apply(train_state.params, obs_batch)

                    # sample the actions from the policy distribution 
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # format the actions to be compatible with the environment
                    env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                    env_act = {k:v.flatten() for k,v in env_act.items()}
                    
                    # STEP ENV
                    # split the random number generator for stepping the environment
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    
                    # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                    obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                        rng_step, env_state, env_act
                    )

                    # REWARD SHAPING IN NEW VERSION
                    
                    # add the reward of one of the agents to the info dictionary
                    info["reward"] = reward["agent_0"]

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

                    # add the shaped reward to the normal reward 
                    reward = jax.tree_util.tree_map(lambda x,y: x+y * rew_shaping_anneal(current_timestep), reward, info["shaped_reward"])

                    # format the outputs of the environment to a 'transition' structure that can be used for analysis
                    # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info) # this is gone in the new version

                    transition = Transition(
                        batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(), 
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch
                    )

                    runner_state = (train_state, env_state, obsv, update_step, rng)
                    return runner_state, (transition, info)
                
                # Apply the _env_step function a series of times, while keeping track of the runner state
                runner_state, (traj_batch, info) = jax.lax.scan(
                    f=_env_step, 
                    init=runner_state, 
                    xs=None, 
                    length=config["NUM_STEPS"]
                )  

                # unpack the runner state that is returned after the scan function
                train_state, env_state, last_obs, update_step, rng = runner_state
                # print('last observations: ', last_obs)

                # create a batch of the observations that is compatible with the network
                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # apply the network to the batch of observations to get the value of the last state
                _, last_val = network.apply(train_state.params, last_obs_batch)
                # this returns the value network for the last observation batch

                def _calculate_gae(traj_batch, last_val):
                    '''
                    calculates the generalized advantage estimate (GAE) for the trajectory batch
                    @param traj_batch: the trajectory batch
                    @param last_val: the value of the last state
                    returns the advantages and the targets
                    '''
                    def _get_advantages(gae_and_next_value, transition):
                        '''
                        calculates the advantage for a single transition
                        @param gae_and_next_value: the GAE and value of the next state
                        @param transition: the transition to calculate the advantage for
                        returns the updated GAE and the advantage
                        '''
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + config["GAMMA"] * next_value * (1 - done) - value # calculate the temporal difference
                        gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                        ) # calculate the GAE (used instead of the standard advantage estimate in PPO)
                        
                        return (gae, value), gae
                    
                    # iteratively apply the _get_advantages function to calculate the advantage for each step in the trajectory batch
                    _, advantages = jax.lax.scan(
                        f=_get_advantages,
                        init=(jnp.zeros_like(last_val), last_val),
                        xs=traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

                # calculate the generalized advantage estimate (GAE) for the trajectory batch
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    '''
                    performs a single update epoch in the training loop
                    @param update_state: the current state of the update
                    returns the updated update_state and the total loss
                    '''
                    
                    def _update_minbatch(train_state, batch_info):
                        '''
                        performs a single update minibatch in the training loop
                        @param train_state: the current state of the training
                        @param batch_info: the information of the batch
                        returns the updated train_state and the total loss
                        '''
                        # unpack the batch information
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            '''
                            calculates the loss of the network
                            @param params: the parameters of the network
                            @param traj_batch: the trajectory batch
                            @param gae: the generalized advantage estimate
                            @param targets: the targets
                            @param network: the network
                            returns the total loss and the value loss, actor loss, and entropy
                            '''
                            # apply the network to the observations in the trajectory batch
                            pi, value = network.apply(params, traj_batch.obs) 
                            log_prob = pi.log_prob(traj_batch.action)

                            # calculate critic loss 
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"]) 
                            value_losses = jnp.square(value - targets) 
                            value_losses_clipped = jnp.square(value_pred_clipped - targets) 
                            value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()) 

                            # Calculate actor loss
                            ratio = jnp.exp(log_prob - traj_batch.log_prob) 
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)  
                            loss_actor_unclipped = ratio * gae 
                            loss_actor_clipped = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            ) 

                            loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped) # calculate the actor loss as the minimum of the clipped and unclipped actor loss
                            loss_actor = loss_actor.mean() # calculate the mean of the actor loss
                            entropy = pi.entropy().mean() # calculate the entropy of the policy 

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        # returns a function with the same parameters as loss_fn that calculates the gradient of the loss function
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                        # call the grad_fn function to get the total loss and the gradients
                        total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                        # apply the gradients to the network
                        train_state = train_state.apply_gradients(grads=grads)

                        # Of course we also need to add the network to the carry here
                        return train_state, total_loss
                    
                    
                    # unpack the update_state (because of the scan function)
                    train_state, traj_batch, advantages, targets, rng = update_state
                    
                    # set the batch size and check if it is correct
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                    ), "batch size must be equal to number of steps * number of actors"
                    
                    # create a batch of the trajectory, advantages, and targets
                    batch = (traj_batch, advantages, targets)          

                    # reshape the batch to be compatible with the network
                    batch = jax.tree_util.tree_map(
                        f=(lambda x: x.reshape((batch_size,) + x.shape[2:])), tree=batch
                    )
                    # split the random number generator for shuffling the batch
                    rng, _rng = jax.random.split(rng)

                    # creates random sequences of numbers from 0 to batch_size, one for each vmap 
                    permutation = jax.random.permutation(_rng, batch_size)

                    # shuffle the batch
                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    ) # outputs a tuple of the batch, advantages, and targets shuffled 

                    minibatches = jax.tree_util.tree_map(
                        f=(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:]))), tree=shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        f=_update_minbatch, 
                        init=train_state,
                        xs=minibatches
                    )
                    
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                # create a tuple to be passed into the jax.lax.scan function
                update_state = (train_state, traj_batch, advantages, targets, rng)

                update_state, loss_info = jax.lax.scan( 
                    f=_update_epoch, 
                    init=update_state, 
                    xs=None, 
                    length=config["UPDATE_EPOCHS"]
                )

                train_state = update_state[0]
                metric = info
                current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                metric["shaped_reward"] = metric["shaped_reward"]["agent_0"]
                metric["shaped_reward_annealed"] = metric["shaped_reward"]*rew_shaping_anneal(current_timestep)
                metric['learning_rate'] = linear_schedule(current_timestep)

                rng = update_state[-1]

                # Run the evaluation function
                # rng, eval_rng = jax.random.split(rng)
                # evaluation_rewards = evaluate_model(train_state, config, eval_rng)
                # metric[f"evaluation_reward_env_0"] = evaluation_rewards[0]
                # metric[f"evaluation_reward_env_1"] = evaluation_rewards[1]

                # Update the step counter
                update_step = update_step + 1
                # update the metric with the current timestep
                metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)

                # If update step is a multiple of 10, run the evaluation function
                # rng, eval_rng = jax.random.split(rng)
                # train_state_eval = jax.tree_util.tree_map(lambda x: x.copy(), train_state)
                # evaluations = jax.lax.cond((update_step % 20) == 0, 
                #                            lambda x: evaluate_model(train_state_eval, config, eval_rng), 
                #                            lambda x: [0.0, 0.0], 
                #                            None)
                # metric[f"evaluation {config['LAYOUT_NAME'][0]}"] = evaluations[0]
                # metric[f"evaluation {config['LAYOUT_NAME'][1]}"] = evaluations[1]

                metric["update_step"] = update_step
                metric["env_step"] = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]

                def callback(metric):
                    wandb.log(
                        metric
                    )
                jax.debug.callback(callback, metric)
                
                runner_state = (train_state, env_state, last_obs, update_step, rng)

                return runner_state, metric

            rng, train_rng = jax.random.split(rng)

            # initialize a carrier that keeps track of the states and observations of the agents
            runner_state = (train_state, env_state, obsv, 0, train_rng)
            
            # apply the _update_step function a series of times, while keeping track of the state 
            runner_state, metric = jax.lax.scan(
                f=_update_step, 
                init=runner_state, 
                xs=None, 
                length=config["NUM_UPDATES"]
            )

            # Return the runner state after the training loop, and the metric arrays
            return runner_state, metric

        
        # step 7: loop over the environments and train the network
       
        # Sequentially apply the train function across environments using lax.scan
        # def apply_train(carry, env_index):
        #     rng, train_state = carry
        #     env = envs[env_index]
        #     train_state, metrics = train_on_environment(rng, train_state, env)
        #     return (rng, train_state), metrics
        
        # # Call lax.scan over the environments
        # (rng, train_state), metrics = jax.lax.scan(
        #     f=apply_train, 
        #     init=(rng, train_state), 
        #     xs=jnp.array([0,1])
        # )

        # split the random number generator for training on the environments
        rng, _rng = jax.random.split(rng)

        runner_state, metrics1 = train_on_environment(rng, train_state, envs[0])
        train_state = runner_state[0]
        train_state, metrics2 = train_on_environment(_rng, train_state, envs[1])
        

        return {"runner_state": train_state, "metrics": [metrics1, metrics2]}
    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_continual") 
def main(cfg):
    
    # check available devices
    print(jax.devices())

    # set the device to the first available GPU
    jax.config.update("jax_platform_name", "gpu")
    
    # set the config to global 
    global config

    # convert the config to a dictionary
    config = OmegaConf.to_container(cfg)

    # set the layout of the environment
    for layout_config in config["ENV_KWARGS"]:
        # Extract the layout name
        layout_name = layout_config["layout"]

        # Set the layout in the config
        layout_config["layout"] = overcooked_layouts[layout_name]

    # Initialize wandb
    wandb.init(
        project="ippo-overcooked", 
        config=config, 
        mode = config["WANDB_MODE"],
        name = f'ippo_continual'
    )

    # Create the training function
    with jax.disable_jit(False):    
        rng = jax.random.PRNGKey(config["SEED"]) # create a pseudo-random key 
        rngs = jax.random.split(rng, config["NUM_SEEDS"]) # split the random key into num_seeds keys
        train_jit = jax.jit(make_train(config)) # JIT compile the training function for faster execution
        out = jax.vmap(train_jit)(rngs) # Vectorize the training function and run it num_seeds times


    filename = f'{config["ENV_NAME"]}_continual'
    train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

    print("Done")


#     # make sure that the code is running on the GPU
#     print(jax.devices())
#     jax.config.update("jax_platform_name", "gpu")
    
#     # set the config to global 
#     global config

#     # convert the config to a dictionary
#     config = OmegaConf.to_container(cfg)

#     # Initialize wandb
#     wandb.init(
#         project=config["PROJECT"], 
#         config=config, 
#         mode=config["WANDB_MODE"],
#         name=f'ippo_{config["ENV_KWARGS"][0]}'  # Start with the first map name for naming
#     )
    
#     # create a list of all maps to train on 
#     maps = [env_kwargs["layout"] for env_kwargs in config["ENV_KWARGS"]]
#     print(f"Training on maps: {maps}")

#     # # Create the training function
#     # with jax.disable_jit(False):    

#     # Set upt the initial random number generator
#     rng = jax.random.PRNGKey(config["SEED"]) 
#     rngs = jax.random.split(rng, config["NUM_SEEDS"]) 

#     # Set up the training parameters
#     num_steps = config["TOTAL_TIMESTEPS"]
#     steps_per_map = num_steps // len(maps)
#     evaluation_interval = config["EVALUATION"]["evaluation_interval"]

#     for layout_config in config["ENV_KWARGS"]:
#         # Extract the layout name
#         layout_name = layout_config["layout"]
#         print(f"Training on layout: {layout_name}")
        
#         # Set the layout in the config
#         layout_config["layout"] = overcooked_layouts[layout_name]

#     env = jax_marl.make(config["ENV_NAME"], **(config["ENV_KWARGS"][0]))  # Start with the first map
#     network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

#     # Initialize the network parameters
#     params = network.init(rngs[0], jnp.zeros(env.observation_space().shape).flatten())

#     for map_idx, map_name in enumerate(maps):
#         print(f"Training on map: {map_name}")
#         env_args = config["ENV_KWARGS"][map_idx]
#         env = jax_marl.make(config["ENV_NAME"], **env_args)

#         # create the training function
#         train_fn = make_train(config, env, network, params, steps_per_map, evaluation_interval)
#         train_jit = jax.jit(train_fn)
#         output = jax.vmap(train_jit, in_axes=(0, None, None))(rngs, env, network)

#         # Optional: Evaluate the model after training on each map
#         train_state = jax.tree_util.tree_map(lambda x: x[0], output["runner_state"][0])
#         state_seq = get_rollout(train_state, config)
#         viz = OvercookedVisualizer()
#         viz.animate(state_seq, agent_view_size=5, filename=f"{config['ENV_NAME']}_{map_name}.gif")

        
     

    # filename = f'{config["ENV_NAME"]}_{maps[0]}'
    # train_state = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
    # state_seq = get_rollout(train_state, config)
    # viz = OvercookedVisualizer()
    # # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")

    print("Done")
    

if __name__ == "__main__":
    print("Running main...")
    main()

