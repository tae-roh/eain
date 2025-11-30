import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), 
            nn.SiLU(),
            nn.Linear(hid_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

class RND(nn.Module):
    def __init__(self, obs_dim, hid_dim=256, out_dim=128, 
                 ema_alpha=0.99, eps=1e-8, 
                 gate_k=2.5, gate_b=0.0,
                 u_min=0.05, u_max=0.98):
        super().__init__()
        self.target = MLP(obs_dim, hid_dim, out_dim)
        self.predictor = MLP(obs_dim, hid_dim, out_dim)
        for param in self.target.parameters():
            param.requires_grad = False

        self.ema_alpha = ema_alpha
        self.eps = eps
        self.register_buffer("rnd_mean", torch.tensor(0.0))
        self.register_buffer("rnd_var",  torch.tensor(1.0))
        self.register_buffer("rnd_inited", torch.tensor(False))

        self.gate_k = gate_k
        self.gate_b = gate_b
        self.u_min = u_min
        self.u_max = u_max
  
    @torch.no_grad()
    def get_uncertainty(self, obs):
        target_feature = self.target(obs)
        predict_feature = self.predictor(obs)

        e = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=-1, keepdim=True)

        self._update_rnd_ema(e.detach())
        std = torch.sqrt(self.rnd_var + self.eps)
        z = (e - self.rnd_mean) / (std + self.eps)
        uncertainty = torch.sigmoid(self.gate_k * (z - self.gate_b))
        uncertainty = torch.clamp(uncertainty, self.u_min, self.u_max)

        return uncertainty
    
    def loss(self, obs):
        with torch.no_grad():
            target_feature = self.target(obs)
        predict_feature = self.predictor(obs)
        rnd_loss = F.mse_loss(predict_feature, target_feature)
        return rnd_loss
    
    @torch.no_grad()
    def _update_rnd_ema(self, e_batch: torch.Tensor):
        m = e_batch.mean()
        v = e_batch.var(unbiased=False)
        if not bool(self.rnd_inited.item()):
            self.rnd_mean.copy_(m)
            self.rnd_var.copy_(v + self.eps)
            self.rnd_inited.fill_(True)
        else:
            self.rnd_mean.mul_(self.ema_alpha).add_(m * (1 - self.ema_alpha))
            self.rnd_var.mul_(self.ema_alpha).add_(v * (1 - self.ema_alpha))