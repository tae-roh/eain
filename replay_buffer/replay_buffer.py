import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32, device=device)
        self.obs_next_buf = torch.zeros([size, obs_dim], dtype=torch.float32, device=device)
        self.acts_buf = torch.zeros([size, act_dim], dtype=torch.float32, device=device)
        self.rews_buf = torch.zeros([size], dtype=torch.float32, device=device)
        self.done_buf = torch.zeros([size], dtype=torch.float32, device=device)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.obs_next_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,))
        return (self.obs_buf[idxs],
                self.acts_buf[idxs],
                self.rews_buf[idxs],
                self.obs_next_buf[idxs],
                self.done_buf[idxs])
