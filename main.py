import yaml
import time
import torch
import numpy as np
import argparse

from agents.sac_agent import SACAgent
from agents.sac_agent_eain import SACAgentEAIN
from envs.make_env import make_env
from utils.logger import Logger


def evaluate(agent, config, episodes=10):
    env = make_env(config['env_name'], seed=config['seed'] + 100, max_episode_steps=config['max_episode_steps'])
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0
        while not done:
            act = agent.select_action(obs, eval_mode=True)
            obs, rew, term, trunc, _ = env.step(act)
            ret += rew
            done = term or trunc
        returns.append(ret)
    return np.mean(returns), np.std(returns)


def train(config):
    torch.manual_seed(config['seed'])
    env = make_env(config['env_name'], seed=config['seed'], max_episode_steps=config['max_episode_steps'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    if config['policy'] == 'SAC':
        agent = SACAgent(obs_dim, act_dim, act_limit, config)
        print("### Using SAC Agent ###")
    elif config['policy'] == 'SAC_EAIN':
        agent = SACAgentEAIN(obs_dim, act_dim, act_limit, config)
        print("### Using SAC_EAIN Agent ###")
        
    logger = Logger(logdir=f'logs/{config["env_name"]}_{config["policy"]}_seed{config["seed"]}_{config["num_epochs"]}k_{int(time.time())}')

    total_steps = config['num_epochs'] * config['steps_per_epoch']
    obs, _ = env.reset()
    episode_return, episode_len = 0, 0

    for t in range(1, total_steps + 1):
        if t < config['start_steps']:
            act = env.action_space.sample()
        else:
            act = agent.select_action(obs)

        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        agent.store_transition(obs, act, rew, next_obs, done)

        obs = next_obs
        episode_return += rew
        episode_len += 1

        if done or episode_len >= config['max_episode_steps']:
            logger.log('EpisodeReturn', episode_return)
            logger.log('EpisodeLength', episode_len)
            obs, _ = env.reset()
            episode_return, episode_len = 0, 0

        if t >= config['update_after'] and t % config['update_every'] == 0:
            for _ in range(config['update_every']):
                agent.update(config['batch_size'])

        if t % (config['eval_interval'] * config['steps_per_epoch']) == 0:
            eval_return, eval_return_std = evaluate(agent, config)
            logger.log('EvalReturn', eval_return)
            logger.log('EvalReturnStd', eval_return_std)
            logger.write(step=t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Training with seed {config['seed']}...")
    train(config)
