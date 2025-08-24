import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        q_input = torch.cat([obs, act], dim=-1)
        return self.q_net(q_input).squeeze(-1)
