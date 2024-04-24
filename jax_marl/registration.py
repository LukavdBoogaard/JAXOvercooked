from .environments import Overcooked



def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    
    # Overcooked
    elif env_id == "overcooked":
        env = Overcooked(**env_kwargs)

    return env

registered_envs = ["overcooked"]
