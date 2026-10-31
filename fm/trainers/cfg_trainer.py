import os
from typing import Optional, List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid

from ..core.paths import GaussianConditionalProbabilityPath
from ..core.solvers import CFGVectorFieldODE, EulerSimulator
from .base_trainer import Trainer


@torch.no_grad()
def visualize_output(
    model: nn.Module,
    path: GaussianConditionalProbabilityPath,
    samples_per_class: int = 10,
    num_timesteps: int = 100,
    guidance_scales: List[float] = [1.0, 3.0, 5.0],
    null_label: int = 10,
    save_path: Optional[str] = None,
    use_tqdm: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Visualizes generated MNIST digits conditioned on classes 0-9 and null class
    across different guidance scales w.
    """
    if device is None:
        device = next(model.parameters()).device

    fig, axes = plt.subplots(
        1,
        len(guidance_scales),
        figsize=(10 * len(guidance_scales), 10),
        squeeze=False
    )

    for idx, w in enumerate(guidance_scales):
        ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=null_label)
        simulator = EulerSimulator(ode)

        # Sample initial conditions: classes 0 through 10
        classes = list(range(null_label + 1))
        y = torch.tensor(classes, dtype=torch.int64, device=device).repeat_interleave(samples_per_class)
        num_samples = y.shape[0]

        x0 = path.p_simple.sample(num_samples)
        # Reshape if flat
        if x0.dim() == 2 and hasattr(path, "p_simple_shape"):
            x0 = x0.view(num_samples, *path.p_simple_shape)

        ts = torch.linspace(0.0, 0.999, num_timesteps, device=device).unsqueeze(0).expand(num_samples, -1)
        x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)

        # Plot
        v_min, v_max = x1.min(), x1.max()
        x1_norm = (x1 - v_min) / (v_max - v_min + 1e-8)
        grid = make_grid(x1_norm, nrow=samples_per_class, normalize=True, value_range=(0, 1))

        ax = axes[0, idx]
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"Guidance: $w={w:.1f}$", fontsize=20)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


class CFGTrainer(Trainer):
    """
    Trainer for Classifier-Free Guidance (CFG) flow matching.
    Drops conditioning labels with probability eta by replacing them with null_label.
    """
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        eta: float,
        null_label: int,
        eps: float = 0.001,
        **kwargs
    ):
        assert 0.0 < eta < 1.0, "eta must be in (0, 1)"
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z, y from p_data
        z, y = self.path.p_data.sample(batch_size)

        # Step 2: Set each label to null_label with probability eta
        xi = torch.rand(y.shape[0], device=y.device)
        y = y.clone()
        y[xi < self.eta] = self.null_label

        # Step 3: Sample time t and noisy state x_t ~ p_t(x|z)
        t = torch.rand(batch_size, device=z.device) * (1.0 - self.eps)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress velocity field and compute MSE loss
        ut_theta = self.model(x, t, y)
        ut_ref = self.path.conditional_vector_field(x, z, t)
        return torch.square(ut_theta - ut_ref).mean()


class MNISTCFGTrainer(CFGTrainer):
    """
    CFG Trainer with MNIST-specific checkpointing and sample visualization callback.
    """
    def checkpoint(self, step: int):
        ckpt_model_path = os.path.join(self.output_dir, f"step_{step:06d}_model.pt")
        ckpt_opt_path = os.path.join(self.output_dir, f"step_{step:06d}_opt.pt")
        ckpt_img_path = os.path.join(self.output_dir, f"step_{step:06d}_output.png")

        torch.save(self.model.state_dict(), ckpt_model_path)
        torch.save(self.opt.state_dict(), ckpt_opt_path)

        visualize_output(
            model=self.model,
            path=self.path,
            null_label=self.null_label,
            save_path=ckpt_img_path,
            use_tqdm=False,
        )
