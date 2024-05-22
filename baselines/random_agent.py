from omegaconf import OmegaConf # a library for managing configuration files in YAML format
import hydra # a framework for elegantly configuring complex applications 
import jax_marl
from jax_marl.environments.overcooked_environment import overcooked_layouts
import jax # a numerical computing library
from jax_marl.wrappers.baselines import LogWrapper
import jax.numpy as jnp 
from typing import Sequence, NamedTuple, Any

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

### HELPER CLASSES ###
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


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
                selects an action randomly, and performs the selected action in the environment
                @param runner_state: the current state of the runner
                returns the updated runner state and the transition
                '''
                env_state, last_obs, rng = runner_state

                # SELECT RANDOM ACTION
                rng, _rng = jax.random.split(rng)

                # prepare the observations for the network
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # sample random actions
                actions = {agent: env.action_space().sample(seed=_rng) for agent in last_obs.keys()}

                # format the actions to be compatible with the environment
                env_act = unbatchify(actions, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}
                
                # STEP ENV
                # split the random number generator for stepping the environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                
                # simultaniously step all environments with the selected actions (parallelized over the number of environments with vmap)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )

                # format the outputs of the environment to a 'transition' structure that can be used for analysis
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(), 
                    actions,
                    jnp.zeros_like(batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()), # no value network, so dummy values
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    jnp.zeros_like(batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()), # no policy network, so dummy log probs
                    obs_batch,
                    info
                )
                runner_state = (env_state, obsv, rng)
                return runner_state, transition
            
            # Apply the _env_step function a series of times, while keeping track of the runner state
            runner_state, traj_batch = jax.lax.scan(
                f=_env_step, 
                init=runner_state, 
                xs=None, 
                length=config["NUM_STEPS"]
            ) 

            metric = traj_batch.info
            
            runner_state = (env_state, obsv, rng)
            return runner_state, metric

        rng, train_rng = jax.random.split(rng)

        # initialize a carrier that keeps track of the states and observations of the agents
        runner_state = (env_state, obsv, train_rng)
        
        # apply the _update_step function a series of times, while keeping track of the state 
        runner_state, metric = jax.lax.scan(
            f=_update_step, 
            init=runner_state, 
            xs=None, 
            length=config["NUM_UPDATES"]
        )

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
    
    number_of_seeds = 20 # number of seeds to generate

    # run the model
    with jax.disable_jit(True): # disable JIT for debugging purposes
        train_jit = jax.jit(jax.vmap(make_train(config))) # make_train is a function that returns a function
        prng = jax.random.split(rng, number_of_seeds) # split the random number generator into 20 new keys
        train_jit(prng) # runs the model 20 times in parallel due to the vmap function

    print("Done running main...")


if __name__ == "__main__":
    print("Running main...")
    main()