import torch
import torch.nn as nn
import torch.nn.functional as F

# Element-wise Action Importance Network
class EAINet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.eai_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, obs):
        net_out = self.eai_net(obs)
        x = F.softplus(net_out) + 1e-6
        x /= (x.sum(dim=-1, keepdim=True).detach() + 1e-8)
        importance = self.act_dim * x
        return importance
    