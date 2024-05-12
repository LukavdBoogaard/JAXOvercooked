from omegaconf import OmegaConf # a library for managing configuration files in YAML format
import hydra # a framework for elegantly configuring complex applications 
import jax_marl
from jax_marl.environments.overcooked_environment import overcooked_layouts
import jax # a numerical computing library
from jax_marl.wrappers.baselines import LogWrapper


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

    def train(prng_key):
        '''
        Function that runs the training loop for the random agent
        @param prng_key: random number generator key
        '''
        pass 

    return train


@hydra.main(version_base=None, config_path="config", config_name="random_agent")
def main(config):
    ''' 
    Main function that runs the model in parallel using the vmap function and visualizes the results
    '''

    # Load the configuration file
    config =  OmegaConf.to_container(config) #converts the config object to a dictionary 
    print('config is: ', config)

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