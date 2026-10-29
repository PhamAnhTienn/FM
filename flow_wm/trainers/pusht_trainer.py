import torch
import torch.nn as nn

from core.paths import GaussianConditionalProbabilityPath
from base_trainer import Trainer

class PushTFlowTrainer(Trainer):
    """
    Kế thừa Trainer từ lab_three.ipynb[cite: 1].
    Kết hợp Conditional Flow Matching Loss (Question 2.2)[cite: 1] với Contact-Aware Physics Loss trên PushT.
    """
    def __init__(self, path: GaussianConditionalProbabilityPath, eta: float = 0.1, phys_weight: float = 0.2, eps: float = 1e-3):
        super().__init__()
        self.path = path
        self.eta = eta
        self.phys_weight = phys_weight
        self.eps = eps

    def get_train_loss(self, batch_size: int):
        # Step 1: Lấy mẫu khung hình tương lai z = I_{t+k} và dict điều kiện y từ PushT[cite: 1]
        z_img, y_dict = self.path.sample_conditioning_variable(batch_size)
        device = z_img.device

        # Step 2: Classifier-Free Guidance dropout trên điều kiện (img_t, proprio_action)[cite: 1]
        drop_mask = (torch.rand(batch_size, 1, 1, 1, device=device) < self.eta).float()
        drop_mask_vec = drop_mask.view(batch_size, 1)

        y_train = {
            "img_t": y_dict["img_t"] * (1.0 - drop_mask),
            "proprio_action": y_dict["proprio_action"] * (1.0 - drop_mask_vec) + self.model.null_cond * drop_mask_vec
        }

        # Step 3: Lấy mẫu thời gian t và ảnh nhiễu nội suy xt ~ p_t(x|z)[cite: 1]
        t = torch.rand(batch_size, device=device) * (1.0 - self.eps)
        xt = self.path.sample_conditional_path(z_img, t)

        # Step 4: Tính loss Flow Matching trên ảnh + Loss vật lý tiếp xúc nếu dùng [PHYS_TOKEN][cite: 1]
        u_ref = self.path.conditional_vector_field(xt, z_img, t)

        if getattr(self.model, "use_phys_token", False):
            u_pred, pred_delta_s = self.model(xt, t, y_train, return_aux_physics=True)
            cfm_loss = torch.mean((u_pred - u_ref) ** 2)

            # 1) Sai số dự đoán dịch chuyển vật lý thực tế (Data supervision)
            true_delta_s = y_dict["delta_state"]
            state_reg_loss = torch.mean((pred_delta_s - true_delta_s) ** 2)

            # 2) Định luật bảo toàn trạng thái tĩnh khi không va chạm (Non-Contact Invariance Prior):
            # Khi in_contact == 0, độ dịch chuyển của khối chữ T (3 chiều cuối: block_x, block_y, block_theta) phải bằng 0
            non_contact_mask = 1.0 - y_dict["in_contact"]  # (B, 1)
            block_motion_violation = pred_delta_s[:, 2:] * non_contact_mask
            contact_prior_loss = torch.mean(block_motion_violation ** 2)

            return cfm_loss + self.phys_weight * (state_reg_loss + 2.0 * contact_prior_loss)
        else:
            u_pred = self.model(xt, t, y_train, return_aux_physics=False)
            return torch.mean((u_pred - u_ref) ** 2)