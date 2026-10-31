import math

import torch
import torch.nn as nn
from einops import rearrange
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


class MHA(nn.Module):
    """
    Multi-headed self-attention implemented from scratch
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0

        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.fold_heads = Rearrange('b n (h d) -> (b h) n d', h=heads)
        self.unfold_heads = Rearrange('(b h) n d -> b n (h d)', h=heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        Returns:
        - x: b n d
        """
        # Compute queries, keys, and values
        q, k, v = self.qkv(x).chunk(3, dim=-1) # b n (h d)

        # Fold head into batch dimension
        q, k, v = map(self.fold_heads, (q, k, v)) # (b h) n d

        # Compute attention
        qk = torch.einsum('bid,bjd->bij', q, k) * self.scale # (b h) n n
        attn = torch.softmax(qk, dim=-1) # (b h) n n

        # Combine with values
        x = torch.einsum('bij,bjd->bid', attn, v) # (b h) n d

        # Unfold heads
        x = self.unfold_heads(x) # b n (h d)

        # Pass through FF and return
        return self.out(x)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - x: b n d
    - scale: b n d
    - bias: b n d
    Returns:
    - x: b n d
    """
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
    ):
        """
        Args:
        - dim: dimension of hidden layers
        - heads: number of attention heads
        """
        super().__init__()

        # Normalization
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6)
        )

        # Initialize conditioning to zero - stabilizes residual connection!
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

        # Attention
        self.attn = MHA(dim, heads)

        # Feedforward
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        - c: b d
        Returns:
        - x: b n d
        """
        # Compute conditioning gating, scaling, and bias
        c = rearrange(self.ada_ln(c), 'b d -> b 1 d') # b 1 d
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)

        # Attention + FF
        x = x + attn_gate * self.attn(
            modulate(self.norm1(x), attn_scale, attn_bias)
        )
        x = x + ff_gate * self.ff(
            modulate(self.norm2(x), ff_scale, ff_bias)
        )
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        n_tokens: int,
        dim: int,
        heads: int = 4,
        **layer_kwargs,
    ):
        """
        Args:
        - depth: number of layers
        - n_tokens: sequence length (for sake of positional embeddings)
        - dim: dimension of hidden layers
        - heads: number of attention heads
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DiffusionTransformerLayer(dim=dim, heads=heads, **layer_kwargs)
            for _ in range(depth)
        ])

        # Positional encodings
        self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        - c: b d
        Returns:
        - x: b n d
        """
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x