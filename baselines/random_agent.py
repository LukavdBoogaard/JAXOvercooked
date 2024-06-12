from omegaconf import OmegaConf # a library for managing configuration files in YAML format
import hydra # a framework for elegantly configuring complex applications 
import jax_marl
from jax_marl.environments.overcooked_environment import overcooked_layouts
import jax # a numerical computing library
from jax_marl.wrappers.baselines import LogWrapper
import jax.numpy as jnp 
from typing import Sequence, NamedTuple, Any
import numpy as np
import matplotlib.pyplot as plt
from jax_marl.viz.overcooked_visualizer import OvercookedVisualizer

# Set global variable
config = None

### HELPER FUNCTIONS ###
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

### HELPER CLASS ###
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(config):
    '''
    Function that generates a rollout of the environment
    @param state: state of the environment
    @param config: configuration file
    @return: rollout of the environment
    '''
    key = jax.random.PRNGKey(0)
    key, key_r= jax.random.split(key, 2) 

    # create the environment
    env = jax_marl.registration.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env)

    done = False

    # reset the environment
    obsv, env_state = env.reset(key_r)

    # intitialze array to store the results with the first state
    state_seq = [env_state]

    while not done:
        key, key_a1, key_a2, key_s = jax.random.split(key, 4)

        # get random actions for both agents
        #TODO: change this so that one agent is also possible
        action_a1 = jax.random.choice(key_a1, env.action_space().n)
        action_a2 = jax.random.choice(key_a1, env.action_space().n)
        actions = {'agent_0': action_a1, 'agent_1': action_a2}


        # take a step in the environment
        obsv, state, reward, done, info = env.step(key_s, env_state, actions)

        done = done["__all__"]

        # append the state to the state sequence
        state_seq.append(state) 
    return state_seq


### TRAINING FUNCTION ###
def make_train(config):
    '''
    Function that creates a training function based on the configuration file. The training function 
    is a random agent that selects actions randomly from the action space. 
    @param config: configuration file
    @return: training function
    '''

    print("Creating training function...")

    # Load the environment
    env = jax_marl.registration.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # we need to define a few more parameters in the configuration file
    # Set the number of actors 
    config['NUM_ACTORS'] = env.num_agents * config['NUM_ENVS']

    # Set the number of updates 
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])

    # Set the size of the minibatches
    config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    # We want to log the results of the training, so we wrap the enviornment into a logger
    env = LogWrapper(env)

    def train(rng):
        '''
        Function that runs the training loop for the random agent
        @param prng_key: random number generator key
        '''
        '''
        Trains the random agent
        @param rng: random number generator 
        returns the runner state and the metrics
        '''
        # Initialize environment 
        rng, env_rng = jax.random.split(rng) # env_rng shape is (1, 2)

        # create config["NUM_ENVS"] seeds for each environment 
        reset_rng = jax.random.split(env_rng, config["NUM_ENVS"]) # reset_rng shape is ('num_envs', 2)

        # create and reset the environment
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        rng, train_rng = jax.random.split(rng)


        def _update_step(runner_state, _):
            '''
            Function that updates the runner state
            @param runner_state: runner state
            @param _: not used
            @return: updated runner state
            '''

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                '''
                selects an action based on the policy, calculates the log probability of the action, 
                and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                # unpack the runner state
                env_state, obsv, rng = runner_state

                # create a random key for each actor
                rng, key_a1, key_a2 = jax.random.split(rng, 3)

                # create a dictionary of actions in all environments for each agent
                actions = {
                    'agent_0': jax.random.choice(key_a1, env.action_space().n, shape=(config["NUM_ENVS"],)),
                    'agent_1': jax.random.choice(key_a2, env.action_space().n, shape=(config["NUM_ENVS"],))
                }

                # take a step in the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, actions
                )

                # create a transition object that stores the results of the step
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                action = batchify(actions, env.agents, config["NUM_ACTORS"]).squeeze()
                obs_batch = batchify(obsv, env.agents, config["NUM_ACTORS"]).squeeze()

                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(), 
                    action,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    obs_batch,
                    info 
                )
                runner_state = (env_state, obsv, rng)
                return runner_state, transition

            runner_state, transition_batch = jax.lax.scan(
                f=_env_step,
                init=runner_state,
                xs=None,
                length=config["NUM_STEPS"]
            )

            metric = transition_batch.info
            return runner_state, metric

        

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (env_state, obsv, train_rng)
        
        # apply the _update_step function a series of times, while keeping track of the state 
        runner_state, metric = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config["NUM_UPDATES"]
        )

        # return the runner state and the metrics
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="random_agent")
def main(cfg):
    ''' 
    Main function that runs the model in parallel using the vmap function and visualizes the results
    '''
    global config
    # Load the configuration file
    config =  OmegaConf.to_container(cfg) #converts the config object to a dictionary 

    # Match the config layout from the available layouts in the overcooked_layouts dictionary
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]["layout"]] 

    # for reproducibility, we need to create a seed for the random number generator
    rng = jax.random.PRNGKey(30) # 30 is the seed value
    
    number_of_seeds = 1 # number of seeds to generate

    out = make_train(config)(rng)
    # run the model
    # with jax.disable_jit(True): # disable JIT for debugging purposes
    #     train_jit = jax.jit(jax.vmap(make_train(config))) # make_train is a function that returns a function
    #     prng = jax.random.split(rng, number_of_seeds) # split the random number generator into 20 new keys
    #     out = train_jit(prng) # runs the model 20 times in parallel due to the vmap function

    print("Done running main...")

    # Save results to a gif and a plot
    print('** Saving Results **')

    filename = f'{config["ENV_NAME"]}_cramped_room_new'
    filename = 'random_agent_results'

    # unpack the results
    # obsv, env_state, reward, done, info = out['runner_state']

    rewards = out['metrics']["returned_episode_returns"].mean(-1).reshape((number_of_seeds, -1)) 
    reward_mean = rewards.mean(0)  # mean 
    reward_std = rewards.std(0) / np.sqrt(number_of_seeds)  # standard error
    
    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)), reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    # compute standard error
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    # # animate first seed
    # train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    # state_seq = get_rollout(train_state, config)
    # viz = OvercookedVisualizer()
    # # agent_view_size is hardcoded as it determines the padding around the layout.
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")



if __name__ == "__main__":
    print("Running main...")
    main()