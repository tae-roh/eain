import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, policy_type='SAC'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.act_limit = act_limit
        self.policy_type = policy_type

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pi_action = mu
        else:
            pi_action = mu + std * torch.randn_like(std)

        if with_logprob:
            if (self.policy_type == 'SAC_EAIN'):
                log_prob = Normal(mu, std).log_prob(pi_action)
                correction = 2 * (torch.log(torch.tensor(2.0, device=log_prob.device)) - pi_action - F.softplus(-2 * pi_action))
            elif (self.policy_type == 'SAC'):
                log_prob = Normal(mu, std).log_prob(pi_action).sum(dim=-1)
                correction = 2 * (torch.log(torch.tensor(2.0, device=log_prob.device)) - pi_action - F.softplus(-2 * pi_action)).sum(dim=-1)
            log_prob = log_prob - correction
        else:
            log_prob = None

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, log_prob

    def sample(self, obs):
        return self.forward(obs, deterministic=False, with_logprob=True)
