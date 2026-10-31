from typing import Optional, List, Type, Tuple
import torch
import torch.nn as nn

from .base import ConditionalVectorField
from .layers import MLP


class MLPConditionalVectorField(ConditionalVectorField):
    """
    MLP-based Conditional Vector Field u_t^theta(x|y).
    Conditioned on class label y via learnable embedding and time t.
    Supports both flat vectors (b, d) and 4D image tensors (b, c, h, w).
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        class_dim: int = 64,
        num_classes: int = 10,
        num_layers: int = 3,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.dim = dim
        self.class_dim = class_dim
        self.num_classes = num_classes

        # Class embedding (num_classes + 1 to account for null_label)
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)

        # MLP layers: [input_dim, hidden, ..., hidden, output_dim]
        # input_dim = x_dim + class_dim + 1 (for time t)
        mlp_dims = [dim + class_dim + 1] + [hidden_dim] * (num_layers - 1) + [dim]
        self.mlp = MLP(mlp_dims, activation=activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (b, d) or (b, c, h, w)
            - t: (b,) or (b, 1)
            - y: (b,) class labels
        Returns:
            - u_t^theta(x|y): velocity field matching x's shape
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.view(orig_shape[0], -1)
        else:
            x_flat = x

        # Ensure t is (b, 1)
        t_flat = t.view(orig_shape[0], 1)

        # Embed label y: (b, class_dim)
        y_embed = self.class_embedding(y)

        # Concatenate [x, y_embed, t]
        xyt = torch.cat([x_flat, y_embed, t_flat], dim=-1)

        # Predict vector field
        u = self.mlp(xyt)

        # Restore original spatial dimensions if input was an image
        if x.dim() > 2:
            u = u.view(*orig_shape)

        return u
