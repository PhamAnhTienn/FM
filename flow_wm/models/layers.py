import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from typing import Optional, List, Type, Tuple, Dict


class MLP(nn.Module):
  def __init__(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU, final_init: bool = False):
    super().__init__()
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    self.net = torch.nn.Sequential(*mlp)

    if final_init:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)

  def forward(self, x: torch.Tensor):
    """
    Args:
    - x: b n d
    Returns:
    - x: b n d
    """
    return self.net(x)


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor):
        """
        Args:
        - t: b
        Returns:
        - embeddings: b d
        """
        #f_i = 2 * pi * w_i * t
        t = t.view(-1, 1) # b 1
        freqs = t * self.weights * 2 * math.pi # b hd

        # sin(f_i) and cos(f_i)
        sin_embed = torch.sin(freqs) # b hd
        cos_embed = torch.cos(freqs) # b hd

        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # b d
    

class Patchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
    super().__init__()
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    self.net = nn.Sequential(
        nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
        Rearrange("b d h w -> b (h w) d"),
    )

  def forward(self, x: torch.Tensor):
    """
    Args:
    - x: (bs, 1, img_size, img_size)
    Returns:
    - x: (bs, 1, img_size, img_size)
    """
    return self.net(x)


class Depatchifier(nn.Module):
  def __init__(self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int):
      super().__init__()
      self.patch_size = patch_size
      assert img_size % patch_size == 0, "Image size must be divisible by patch size"
      h = w = img_size // patch_size


      self.net = nn.Sequential(
          nn.RMSNorm(dim, elementwise_affine=False),
          MLP([dim, 4*dim, final_dim * patch_size ** 2]),

          Rearrange("b (h w) (f ph pw) -> b f (h ph) (w pw)", h=h, w=w, f=final_dim, ph=patch_size, pw=patch_size),

          nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1)
      )

  def forward(self, x: torch.Tensor):
    """
    Args:
    - x: b n d
    Returns:
    - x: b 1 32 32
    """
    return self.net(x)