import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple, Dict

from flow_wm.core.distributions import LabeledSampleable

class PushTWorldSampler(nn.Module, LabeledSampleable):
    def __init__(self, zarr_path: str = None, img_size: int = 32, pred_step_k: int = 4):
        super().__init__()
        self.img_size = img_size
        self.k = pred_step_k
        self.dummy = nn.Buffer(torch.zeros(1))

        if zarr_path is not None:
            # Khi chạy với file zarr thật của Diffusion Policy (pusht_cchi_v7_replay.zarr)
            import zarr
            root = zarr.open(zarr_path, mode='r')
            # Chuẩn hóa ảnh từ [0, 255] (N, H, W, 3) -> [-1, 1] (N, 3, img_size, img_size)
            imgs = torch.from_numpy(root['data']['img'][:]).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            self.imgs = F.interpolate(imgs, size=(img_size, img_size), mode='bilinear', align_corners=False)
            # Chuẩn hóa tọa độ không gian làm việc PushT [0, 512] về [-1, 1]
            states = torch.from_numpy(root['data']['state'][:]).float()
            states[:, :4] = (states[:, :4] / 256.0) - 1.0
            states[:, 4] = states[:, 4] / 3.14159  # Góc xoay chuẩn hóa về [-1, 1]
            self.states = states
            self.actions = (torch.from_numpy(root['data']['action'][:]).float() / 256.0) - 1.0
            self.episode_ends = root['meta']['episode_ends'][:]
            self.valid_indices = self._compute_valid_indices()
        else:
            self.imgs = None  # Sẽ dùng hàm sinh nhanh để kiểm thử luồng code (Sanity Check)

    def _compute_valid_indices(self):
        valid = []
        start_idx = 0
        for end_idx in self.episode_ends:
            # Không lấy mẫu vượt qua ranh giới giữa 2 episode khác nhau
            valid.extend(range(start_idx, end_idx - self.k))
            start_idx = end_idx
        return torch.tensor(valid, dtype=torch.long)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.dummy.device
        if self.imgs is not None:
            idx_select = self.valid_indices[torch.randint(0, len(self.valid_indices), (num_samples,))]
            img_t = self.imgs[idx_select].to(device)
            img_next = self.imgs[idx_select + self.k].to(device)
            s_t = self.states[idx_select].to(device)
            s_next = self.states[idx_select + self.k].to(device)
            # Gom cụm k hành động liên tiếp (Action Chunking): (B, k * 2)
            a_chunk = torch.stack([self.actions[idx_select + i] for i in range(self.k)], dim=1).flatten(1).to(device)
        else:
            # Chế độ Fallback (Mock PushT) giúp bạn test kiến trúc mạng trong 5 giây trước khi tải Zarr
            img_t = torch.randn(num_samples, 3, self.img_size, self.img_size, device=device).clamp(-1, 1)
            img_next = torch.randn(num_samples, 3, self.img_size, self.img_size, device=device).clamp(-1, 1)
            s_t = torch.rand(num_samples, 5, device=device) * 2 - 1
            s_next = s_t + 0.1 * torch.randn_like(s_t)
            a_chunk = torch.rand(num_samples, 2 * self.k, device=device) * 2 - 1

        delta_state = s_next - s_t
        # Tính khoảng cách giữa đầu đẩy (agent_x, agent_y) và tâm khối T (block_x, block_y)
        dist_agent_block = torch.norm(s_t[:, :2] - s_t[:, 2:4], dim=-1, keepdim=True)
        # Trong PushT chuẩn hóa [-1, 1], bán kính va chạm xấp xỉ 0.35
        in_contact = (dist_agent_block < 0.35).float()

        y_cond = {
            "img_t": img_t,
            "proprio_action": torch.cat([s_t, a_chunk], dim=-1),  # Chiều: 5 + 2*k
            "delta_state": delta_state,                           # Chiều: 5
            "in_contact": in_contact                              # Chiều: 1
        }
        return img_next, y_cond