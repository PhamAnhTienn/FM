import torch
import torch.nn as nn
from typing import Optional

from .base import ConditionalVectorField
from .layers import (
    FourierEncoder,
    Patchifier,
    Depatchifier,
    DiffusionTransformer,
)


class DiffusionTransformerFlowModel(ConditionalVectorField):
    """
    Diffusion Transformer (DiT) based conditional vector field u_t^theta(x|y).
    Built from scratch using Fourier time encoder, class embedding,
    patchifier, transformer layers with AdaLN modulation, and depatchifier.
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 256,
        heads: int = 4,
        final_dim: int = 10,
        n_classes: int = 11,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.c = c
        self.dim = dim
        self.n_classes = n_classes

        # 0. Construct time_embedder and y_embedder
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(num_embeddings=n_classes, embedding_dim=dim)

        # 1. Construct patchifier
        self.patchifier = Patchifier(
            img_size=img_size,
            patch_size=patch_size,
            c_in=c,
            dim=dim
        )

        # 2. Construct DiT
        n_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )

        # 3. Construct depatchifier
        self.depatchifier = Depatchifier(
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            final_dim=final_dim,
            c_out=c
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b, c, img_size, img_size)
            - t: time tensor of shape (b,) or broadcastable
            - y: class labels of shape (b,)
        Returns:
            - u_t^theta(x|y): velocity field matching x's shape (b, c, img_size, img_size)
        """
        # Ensure t is 1D tensor (b,)
        if t.dim() > 1:
            t = t.view(-1)

        # 1. Embed time and y
        t_embed = self.time_embedder(t)      # (b, d)
        y_embed = self.y_embedder(y)          # (b, d)

        # 2. Patchify
        tokens = self.patchifier(x)          # (b, n_tokens, d)

        # 3. Pass through DiT with conditioning
        tokens = self.dit(tokens, t_embed + y_embed)  # (b, n_tokens, d)

        # 4. Depatchify
        u = self.depatchifier(tokens)        # (b, c, img_size, img_size)

        return u
