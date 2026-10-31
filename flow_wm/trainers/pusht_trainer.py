import torch
import torch.nn as nn

from ..core.paths import GaussianConditionalProbabilityPath
from .base_trainer import Trainer

class PushTFlowTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float = 0.1, phys_weight: float = 0.2, eps: float = 1e-3):
        super().__init__()
        self.path = path
        self.eta = eta
        self.phys_weight = phys_weight
        self.eps = eps

    def get_train_loss(self, batch_size: int):
        # Sample future frame z = I_{t+k} and conditional dict y
        z_img, y_dict = self.path.sample_conditioning_variable(batch_size)
        device = z_img.device

        # Classifier-Free Guidance dropout on conditions
        drop_mask = (torch.rand(batch_size, 1, 1, 1, device=device) < self.eta).float()
        drop_mask_vec = drop_mask.view(batch_size, 1)

        y_train = {
            "img_t": y_dict["img_t"] * (1.0 - drop_mask),
            "proprio_action": y_dict["proprio_action"] * (1.0 - drop_mask_vec) + self.model.null_cond * drop_mask_vec
        }

        # Sample time t and noisy image ~ p_t(x|z)
        t = torch.rand(batch_size, device=device) * (1.0 - self.eps)
        xt = self.path.sample_conditional_path(z_img, t)

        u_ref = self.path.conditional_vector_field(xt, z_img, t)

        if getattr(self.model, "use_phys_token", False):
            u_pred, pred_delta_s = self.model(xt, t, y_train, return_aux_physics=True)
            cfm_loss = torch.mean((u_pred - u_ref) ** 2)

            true_delta_s = y_dict["delta_state"]
            state_reg_loss = torch.mean((pred_delta_s - true_delta_s) ** 2)

            non_contact_mask = 1.0 - y_dict["in_contact"]  # (B, 1)
            block_motion_violation = pred_delta_s[:, 2:] * non_contact_mask
            contact_prior_loss = torch.mean(block_motion_violation ** 2)

            return cfm_loss + self.phys_weight * (state_reg_loss + 2.0 * contact_prior_loss)
        else:
            u_pred = self.model(xt, t, y_train, return_aux_physics=False)
            return torch.mean((u_pred - u_ref) ** 2)