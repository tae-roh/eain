import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam

from networks.actor import SquashedGaussianMLPActor
from networks.critic import MLPQFunction
from replay_buffer.replay_buffer import ReplayBuffer
from utils.torch_utils import soft_update


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, config):
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']
        self.device = torch.device(config['device'])

        self.actor = SquashedGaussianMLPActor(obs_dim, act_dim, act_limit, config['policy']).to(self.device)
        self.critic1 = MLPQFunction(obs_dim, act_dim).to(self.device)
        self.critic2 = MLPQFunction(obs_dim, act_dim).to(self.device)
        self.target_critic1 = MLPQFunction(obs_dim, act_dim).to(self.device)
        self.target_critic2 = MLPQFunction(obs_dim, act_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=config['critic_lr'])
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=config['critic_lr'])

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, config['replay_size'], self.device)

    def select_action(self, obs, eval_mode=False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if eval_mode:
                action, _ = self.actor(obs_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(obs_tensor)
        return action.cpu().numpy()

    def update(self, batch_size):
        if self.replay_buffer.size < batch_size:
            return

        obs, act, rew, next_obs, done = self.replay_buffer.sample_batch(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            backup = rew + self.gamma * (1 - done) * target_q

        # Update critics
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic1_loss = F.mse_loss(q1, backup)
        critic2_loss = F.mse_loss(q2, backup)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_act, log_prob = self.actor.sample(obs)
        q1_pi = self.critic1(obs, new_act)
        q2_pi = self.critic2(obs, new_act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target critic networks
        soft_update(self.critic1, self.target_critic1, self.tau)
        soft_update(self.critic2, self.target_critic2, self.tau)

    def store_transition(self, obs, act, rew, next_obs, done):
        self.replay_buffer.store(obs, act, rew, next_obs, done)
