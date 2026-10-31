import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from flow_wm.core.distributions import LabeledSampleable

class PushTWorldSampler(nn.Module, LabeledSampleable):
    def __init__(self, zarr_path: str = None, img_size: int = 32, pred_step_k: int = 4):
        super().__init__()
        self.img_size = img_size
        self.k = pred_step_k
        self.dummy = nn.Buffer(torch.zeros(1))

        if zarr_path is not None:
            import zarr
            root = zarr.open(zarr_path, mode='r')
            # Standardize image from [0, 255] (N, H, W, 3) -> [-1, 1] (N, 3, img_size, img_size)
            imgs = torch.from_numpy(root['data']['img'][:]).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            self.imgs = F.interpolate(imgs, size=(img_size, img_size), mode='bilinear', align_corners=False)
            
            # Standardize PushT workspace coordinates [0, 512] to [-1, 1]
            states = torch.from_numpy(root['data']['state'][:]).float()
            states[:, :4] = (states[:, :4] / 256.0) - 1.0
            states[:, 4] = states[:, 4] / 3.14159  # Normalize angle to [-1, 1]
            self.states = states
            self.actions = (torch.from_numpy(root['data']['action'][:]).float() / 256.0) - 1.0
            self.episode_ends = root['meta']['episode_ends'][:]
            self.valid_indices = self.compute_valid_indices()
        else:
            # Fallback to mock generation for testing
            self.imgs = None  


    def compute_valid_indices(self):
        valid = []
        start_idx = 0
        for end_idx in self.episode_ends:
            # Don't sample across episode boundaries
            valid.extend(range(start_idx, end_idx - self.k))
            start_idx = end_idx
        return torch.tensor(valid, dtype=torch.long)


    def sample(self, num_samples: int):
        device = self.dummy.device
        
        if self.imgs is not None:
            idx_select = self.valid_indices[torch.randint(0, len(self.valid_indices), (num_samples,))]
            img_t = self.imgs[idx_select].to(device)
            img_next = self.imgs[idx_select + self.k].to(device)
            s_t = self.states[idx_select].to(device)
            s_next = self.states[idx_select + self.k].to(device)
            # Group k consecutive actions together (Action Chunking)
            a_chunk = torch.stack([self.actions[idx_select + i] for i in range(self.k)], dim=1).flatten(1).to(device)
        else:
            # Fallback to mock generation for testing
            img_t = torch.randn(num_samples, 3, self.img_size, self.img_size, device=device).clamp(-1, 1)
            img_next = torch.randn(num_samples, 3, self.img_size, self.img_size, device=device).clamp(-1, 1)
            s_t = torch.rand(num_samples, 5, device=device) * 2 - 1
            s_next = s_t + 0.1 * torch.randn_like(s_t)
            a_chunk = torch.rand(num_samples, 2 * self.k, device=device) * 2 - 1

        delta_state = s_next - s_t
        # Calculate distance between the pusher (agent_x, agent_y) and the center of T block (block_x, block_y)
        dist_agent_block = torch.norm(s_t[:, :2] - s_t[:, 2:4], dim=-1, keepdim=True)
        # In normalized PushT [-1, 1], collision radius is approximately 0.35
        in_contact = (dist_agent_block < 0.35).float()

        y_cond = {
            "img_t": img_t,
            "proprio_action": torch.cat([s_t, a_chunk], dim=-1),
            "delta_state": delta_state,
            "in_contact": in_contact
        }
        return img_next, y_cond