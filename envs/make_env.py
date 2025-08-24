import gymnasium as gym
from gymnasium.wrappers import TimeLimit


def make_env(env_name, seed=None, max_episode_steps=None):
    env = gym.make(env_name)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env
