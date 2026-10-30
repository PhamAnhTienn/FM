from typing import Optional, Dict, Tuple, Union
from einops import rearrange

import torch
import torch.nn as nn
from .base import ConditionalVectorField
from .layers import Depatchifier, MLP, FourierEncoder, Patchifier


class DiffusionTransformer(nn.Module):
    def __init__(self, depth: int, n_tokens: int, dim: int, heads: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.condition_norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor):
        return self.encoder(tokens + self.condition_norm(condition).unsqueeze(1))


class PushTFlowDiT(ConditionalVectorField):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        c_in: int = 3,
        cond_dim: int = 13,      # 5 state s_t + 8 action (k=4 step * 2)
        dim: int = 256,
        depth: int = 8,
        heads: int = 4,
        final_dim: int = 16,
        use_phys_token: bool = True,
    ):
        super().__init__()
        self.use_phys_token = use_phys_token

        # 0. Bộ nhúng thời gian t và điều kiện liên tục y = (s_t, a_t)
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = MLP([cond_dim, dim, dim])
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))

        # 1. Patchifier: Nhận ghép ảnh đang khử nhiễu x_t (3 kênh) + ảnh hiện tại I_t (3 kênh) = 6 kênh
        self.patchifier = Patchifier(
            img_size=img_size,
            patch_size=patch_size,
            c_in=c_in * 2,
            dim=dim
        )

        # 2. Construct DiT (Tăng thêm 1 token nếu dùng [PHYS_TOKEN])
        n_patches = (img_size // patch_size) ** 2
        n_tokens = n_patches + (1 if use_phys_token else 0)
        if use_phys_token:
            self.phys_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.phys_head = MLP([dim, dim, 5], final_init=True) # Dự đoán delta_s (5 chiều)

        self.dit = DiffusionTransformer(
            depth=depth,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )

        # 3. Construct Depatchifier (Dùng nguyên bản từ Notebook của bạn, xuất ra c_out = 3 kênh RGB)
        self.depatchifier = Depatchifier(
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            final_dim=final_dim,
            c_out=c_in
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[dict] = None, return_aux_physics: bool = False):
        """
        Args:
        - x: Ảnh tương lai đang nhiễu x_t, shape (b, 3, 32, 32)
        - t: Thời gian flow t in [0, 1], shape (b,)
        - y: Dict chứa 'img_t' (b, 3, 32, 32) và 'proprio_action' (b, 13)
        Returns:
        - u_img: Trường vector vận tốc ảnh (b, 3, 32, 32)
        """
        B = x.shape[0]
        if y is None:
            img_t = torch.zeros_like(x)
            pa_cond = self.null_cond.expand(B, -1)
        else:
            img_t = y["img_t"]
            pa_cond = y["proprio_action"]

        # 1. Embed time t và điều kiện vật lý - hành động y
        t_embed = self.time_embedder(t)       # (b, dim)
        y_embed = self.y_embedder(pa_cond)    # (b, dim)

        # 2. Ghép kênh [x, img_t] -> Patchify thành (b, n_patches, dim)
        tokens = self.patchifier(torch.cat([x, img_t], dim=1))

        if self.use_phys_token:
            p_tok = self.phys_token.expand(B, -1, -1)
            tokens = torch.cat([p_tok, tokens], dim=1) # (b, 1 + n_patches, dim)

        # 3. Đi qua DiffusionTransformer của Notebook
        tokens = self.dit(tokens, t_embed + y_embed)

        # 4. Tách [PHYS_TOKEN] (nếu có) và đưa các Visual Tokens vào Depatchifier
        if self.use_phys_token:
            phys_out_tok = tokens[:, 0]      # (b, dim)
            vis_tokens = tokens[:, 1:]       # (b, n_patches, dim)
        else:
            vis_tokens = tokens

        u_img = self.depatchifier(vis_tokens) # (b, 3, 32, 32)

        if return_aux_physics and self.use_phys_token:
            pred_delta_s = self.phys_head(phys_out_tok) # (b, 5)
            return u_img, pred_delta_s

        return u_img