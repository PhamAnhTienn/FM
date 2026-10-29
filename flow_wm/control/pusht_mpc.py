import torch
from flow_wm.core.solvers import ODE, EulerSimulator
from flow_wm.core.distributions import Sampleable
from flow_wm.models.pusht_dit import PushTFlowDiT


class PushTCFGVectorFieldODE(ODE):
    """
    Wrapper kế thừa class ODE từ lab_three.ipynb để tích hợp Classifier-Free Guidance
    cho điều kiện đầu vào dạng Dict ('img_t', 'proprio_action') của bộ dữ liệu PushT.
    """
    def __init__(self, net: PushTFlowDiT, guidance_scale: float = 1.5):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        y = kwargs.get("y", None)
        if self.guidance_scale == 1.0 or y is None:
            return self.net(xt, t, y, return_aux_physics=False)

        guided_vf = self.net(xt, t, y, return_aux_physics=False)
        unguided_vf = self.net(xt, t, None, return_aux_physics=False)
        return (1.0 - self.guidance_scale) * unguided_vf + self.guidance_scale * guided_vf


class PushTFlowPlanner:
    """
    Bộ lập kế hoạch Model Predictive Control (Shooting MPC) cho bài toán đẩy khối chữ T (PushT).
    Tận dụng khả năng suy luận cực ít bước (NFE = 1..4) của EulerSimulator và [PHYS_TOKEN].
    """
    def __init__(
        self,
        model: PushTFlowDiT,
        p_simple: Sampleable,
        pred_step_k: int = 4,
        num_candidates: int = 128,
        nfe_steps: int = 2,
        guidance_scale: float = 1.0
    ):
        self.model = model
        self.ode = PushTCFGVectorFieldODE(model, guidance_scale=guidance_scale)
        self.simulator = EulerSimulator(self.ode)
        self.p_simple = p_simple
        self.k = pred_step_k
        self.num_candidates = num_candidates
        self.nfe_steps = nfe_steps

    @torch.no_grad()
    def plan_action_chunk(
        self,
        img_t: torch.Tensor,
        state_t: torch.Tensor,
        goal_block_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            - img_t: Khung hình hiện tại (1, 3, H, W)
            - state_t: Trạng thái hiện tại [agent_x, agent_y, block_x, block_y, block_theta] (1, 5)
            - goal_block_pose: Đích đến của khối chữ T [goal_x, goal_y, goal_theta] (1, 3)
        Returns:
            - best_action_chunk: Cụm k hành động tối ưu (k, 2)
        """
        device = img_t.device
        N = self.num_candidates

        # Nhân bản trạng thái hiện tại cho N quỹ đạo ứng viên song song
        img_batch = img_t.expand(N, -1, -1, -1)
        state_batch = state_t.expand(N, -1)

        # Lấy mẫu N cụm hành động ứng viên trong [-1, 1]: (N, 2 * k)
        candidate_actions = torch.rand(N, 2 * self.k, device=device) * 2.0 - 1.0
        y_cond = {
            "img_t": img_batch,
            "proprio_action": torch.cat([state_batch, candidate_actions], dim=-1)
        }

        # Giải ODE bằng EulerSimulator từ lab_three.ipynb với lưới thời gian siêu thưa (NFE thấp)
        x_init = self.p_simple.sample(N)
        ts = torch.linspace(0, 1, self.nfe_steps + 1, device=device).expand(N, -1)

        # Chạy tới bước áp chót bằng EulerSimulator, bước cuối rút trích thêm dự đoán vật lý từ [PHYS_TOKEN]
        if self.nfe_steps > 1:
            x_curr = self.simulator.simulate(x_init, ts[:, :-1], use_tqdm=False, y=y_cond)
        else:
            x_curr = x_init

        t_last = ts[:, -2]
        h_last = ts[:, -1] - ts[:, -2]
        _, pred_delta_s = self.model(x_curr, t_last, y_cond, return_aux_physics=True)

        # Trạng thái dự đoán của khối chữ T tại t + k: [block_x, block_y, block_theta]
        pred_next_state = state_batch + pred_delta_s
        pred_block_pose = pred_next_state[:, 2:]   # (N, 3)
        pred_agent_pos = pred_next_state[:, :2]    # (N, 2)

        # Hàm chi phí: (1) Khoảng cách khối T tới đích + (2) Khuyến khích đầu đẩy tiến sát khối T
        block_to_goal_cost = torch.norm(pred_block_pose[:, :2] - goal_block_pose[:, :2], dim=-1) * 5.0
        angle_cost = torch.abs(pred_block_pose[:, 2] - goal_block_pose[:, 2]) * 2.0
        agent_to_block_cost = torch.norm(pred_agent_pos - pred_block_pose[:, :2], dim=-1) * 0.5

        total_cost = block_to_goal_cost + angle_cost + agent_to_block_cost
        best_idx = torch.argmin(total_cost)

        return candidate_actions[best_idx].view(self.k, 2)