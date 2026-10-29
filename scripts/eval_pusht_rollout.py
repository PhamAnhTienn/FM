import os
import time
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from flow_wm.core.distributions import IsotropicGaussian
from flow_wm.core.solvers import EulerSimulator
from flow_wm.envs.pusht_sampler import PushTWorldSampler
from flow_wm.models.pusht_dit import PushTFlowDiT
from flow_wm.control.pusht_mpc import PushTCFGVectorFieldODE, PushTFlowPlanner


def load_trained_model(ckpt_path: str, device: torch.device) -> PushTFlowDiT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond_dim = 5 + 2 * ckpt["pred_step_k"]
    model = PushTFlowDiT(
        img_size=ckpt["img_size"],
        patch_size=ckpt["patch_size"],
        c_in=3,
        cond_dim=cond_dim,
        dim=ckpt["hidden_dim"],
        depth=ckpt["depth"],
        heads=ckpt["heads"],
        use_phys_token=ckpt["use_phys_token"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_nfe_efficiency_and_quality(
    model: PushTFlowDiT,
    sampler: PushTWorldSampler,
    p_simple: IsotropicGaussian,
    device: torch.device,
    nfe_list=(1, 2, 4, 10, 20),
    batch_size: int = 64
):
    """
    So sánh MSE tái tạo khung hình tương lai, sai số vật lý và độ trễ (ms) theo số bước giải ODE (NFE).
    """
    z_true_img, y_cond = sampler.sample(batch_size)
    ode = PushTCFGVectorFieldODE(model, guidance_scale=1.2)
    simulator = EulerSimulator(ode)

    metrics = {"nfe": [], "img_mse": [], "latency_ms": [], "non_contact_drift": []}

    for nfe in nfe_list:
        x_init = p_simple.sample(batch_size)
        ts = torch.linspace(0, 1, nfe + 1, device=device).expand(batch_size, -1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Mô phỏng quá trình sinh khung hình I_{t+k} bằng EulerSimulator
        pred_img = simulator.simulate(x_init, ts, use_tqdm=False, y=y_cond)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        img_mse = torch.mean((pred_img - z_true_img) ** 2).item()

        # Đo lỗi vi phạm vật lý khi không tiếp xúc (Non-Contact Drift)
        non_contact_mask = (1.0 - y_cond["in_contact"]).view(-1, 1, 1, 1)
        # Độ lệch pixel bất thường giữa ảnh dự đoán và ảnh gốc khi khối T lẽ ra đứng yên
        drift_err = torch.mean(((pred_img - z_true_img) ** 2) * non_contact_mask).item()

        metrics["nfe"].append(nfe)
        metrics["img_mse"].append(img_mse)
        metrics["latency_ms"].append(elapsed_ms)
        metrics["non_contact_drift"].append(drift_err)

        print(f"  [NFE={nfe:2d}] Image MSE: {img_mse:.4f} | Non-Contact Error: {drift_err:.4f} | Latency (batch={batch_size}): {elapsed_ms:.2f} ms")

    return metrics, z_true_img, y_cond


@torch.no_grad()
def generate_multistep_rollout_grid(
    model: PushTFlowDiT,
    sampler: PushTWorldSampler,
    p_simple: IsotropicGaussian,
    device: torch.device,
    horizon_Rollout: int = 5,
    nfe_steps: int = 4,
    save_path: str = "results/pusht_rollout_grid.png"
):
    """
    Cuộn (Rollout) tự hồi quy nhiều bước thời gian liên tiếp: I_0 -> I_k -> I_2k -> ...
    Xuất ra ảnh so sánh trực quan để gắn vào README.md.
    """
    ode = PushTCFGVectorFieldODE(model, guidance_scale=1.2)
    simulator = EulerSimulator(ode)

    num_vis = 4  # Số quỹ đạo hiển thị
    z_target, y_cond = sampler.sample(num_vis)
    cur_img = y_cond["img_t"]  # Khung hình xuất phát I_0: (num_vis, 3, H, W)

    rollout_frames = [cur_img]
    ts = torch.linspace(0, 1, nfe_steps + 1, device=device).expand(num_vis, -1)

    for step in range(horizon_Rollout):
        x_init = p_simple.sample(num_vis)
        step_cond = {
            "img_t": cur_img,
            "proprio_action": y_cond["proprio_action"]
        }
        next_img = simulator.simulate(x_init, ts, use_tqdm=False, y=step_cond).clamp(-1.0, 1.0)
        rollout_frames.append(next_img)
        cur_img = next_img  # Đưa khung hình vừa dự đoán làm đầu vào cho bước kế tiếp (Autoregressive)

    # Xếp thành lưới (Hàng = các quỹ đạo, Cột = thời gian t=0, k, 2k, 3k...)
    all_frames = torch.stack(rollout_frames, dim=1).flatten(0, 1)  # (num_vis * (horizon+1), 3, H, W)
    grid = make_grid(all_frames, nrow=horizon_Rollout + 1, normalize=True, value_range=(-1, 1), pad_value=1.0)
    save_image(grid, save_path)
    print(f"🖼 Đã lưu chuỗi khung hình dự đoán đa bước tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PushT Flow World Model Rollout & MPC Efficiency")
    parser.add_argument("--zarr_path", type=str, default=None, help="Đường dẫn tới pusht_cchi_v7_replay.zarr")
    parser.add_argument("--phys_ckpt", type=str, default="checkpoints/pusht_phys_grounded.pt")
    parser.add_argument("--base_ckpt", type=str, default="checkpoints/pusht_baseline_video.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    all_metrics = {}
    for label, ckpt_path in [
        ("Physically-Grounded (+[PHYS_TOKEN])", args.phys_ckpt),
        ("Baseline Video DiT", args.base_ckpt)
    ]:
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Bỏ qua {ckpt_path} (chưa tìm thấy file checkpoint).")
            continue

        print(f"\n🔍 Đánh giá mô hình: {label}")
        model, ckpt = load_trained_model(ckpt_path, device)
        sampler = PushTWorldSampler(
            zarr_path=args.zarr_path,
            img_size=ckpt["img_size"],
            pred_step_k=ckpt["pred_step_k"]
        ).to(device)
        p_simple = IsotropicGaussian(shape=[3, ckpt["img_size"], ckpt["img_size"]]).to(device)

        metrics, _, y_cond = evaluate_nfe_efficiency_and_quality(model, sampler, p_simple, device)
        all_metrics[label] = metrics

        # Xuất ảnh Rollout đa bước
        tag_slug = "phys" if ckpt["use_phys_token"] else "base"
        generate_multistep_rollout_grid(
            model, sampler, p_simple, device,
            horizon_Rollout=5, nfe_steps=2,
            save_path=f"results/pusht_rollout_{tag_slug}_nfe2.png"
        )

        # Kiểm thử tốc độ lập kế hoạch điều khiển (MPC Planning) nếu có [PHYS_TOKEN]
        if ckpt["use_phys_token"]:
            planner = PushTFlowPlanner(
                model=model, p_simple=p_simple,
                pred_step_k=ckpt["pred_step_k"], num_candidates=128, nfe_steps=2
            )
            goal_pose = torch.tensor([[0.0, 0.0, 0.0]], device=device)
            t0 = time.perf_counter()
            best_actions = planner.plan_action_chunk(
                img_t=y_cond["img_t"][:1],
                state_t=y_cond["proprio_action"][:1, :5],
                goal_block_pose=goal_pose
            )
            mpc_ms = (time.perf_counter() - t0) * 1000.0
            print(f"🎯 [MPC Planner | 128 trajectories | NFE=2] Thời gian hoạch định: {mpc_ms:.2f} ms | Action chunk shape: {tuple(best_actions.shape)}")

    # Vẽ biểu đồ Benchmark NFE vs Latency & Error
    if all_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for label, m in all_metrics.items():
            axes[0].plot(m["nfe"], m["non_contact_drift"], marker="o", linewidth=2, label=label)
            axes[1].plot(m["nfe"], m["latency_ms"], marker="s", linewidth=2, label=label)

        axes[0].set_xlabel("Number of Function Evaluations (NFE - Euler Steps)")
        axes[0].set_ylabel("Non-Contact Physical Violation Error (MSE)")
        axes[0].set_title("Physical Grounding: Error vs. ODE Steps")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Number of Function Evaluations (NFE - Euler Steps)")
        axes[1].set_ylabel("Batch Rollout Latency (ms)")
        axes[1].set_title("Computational Efficiency: Latency vs. ODE Steps")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/pusht_nfe_ablation_benchmark.png", dpi=200)
        print("Đã lưu biểu đồ Benchmark tại: results/pusht_nfe_ablation_benchmark.png")


if __name__ == "__main__":
    main()