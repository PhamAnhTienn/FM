import os
import argparse
import torch
import matplotlib.pyplot as plt

from flow_wm.core.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from flow_wm.envs.pusht_sampler import PushTWorldSampler
from flow_wm.models.pusht_dit import PushTFlowDiT
from flow_wm.trainers.pusht_trainer import PushTFlowTrainer
from flow_wm.trainers.base_trainer import model_size_b, MiB


def train_single_config(
    use_phys_token: bool,
    args: argparse.Namespace,
    device: torch.device,
    sampler: PushTWorldSampler
):
    tag = "phys_grounded" if use_phys_token else "baseline_video"
    print(f"\n{'='*60}")
    print(f"🚀 Bắt đầu huấn luyện cấu hình: {tag.upper()} (use_phys_token={use_phys_token})")
    print(f"{'='*60}")

    # 1. Khởi tạo đường dẫn xác suất Optimal Transport Flow (alpha_t = t, beta_t = 1 - t)
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[3, args.img_size, args.img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    # 2. Khởi tạo kiến trúc PushTFlowDiT
    cond_dim = 5 + 2 * args.pred_step_k  # 5 chiều state + 2*k chiều action chunk
    model = PushTFlowDiT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        c_in=3,
        cond_dim=cond_dim,
        dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        use_phys_token=use_phys_token
    ).to(device)

    size_mib = model_size_b(model) / MiB
    print(f"📦 Kích thước mô hình ({tag}): {size_mib:.3f} MiB")

    # 3. Huấn luyện mô hình
    trainer = PushTFlowTrainer(
        path=path,
        eta=args.cfg_eta,
        phys_weight=args.phys_weight if use_phys_token else 0.0
    )

    losses, steps = trainer.train(
        model=model,
        num_steps=args.num_steps,
        lr=args.lr,
        warmup_steps=min(200, args.num_steps // 5),
        ckpt_every=None,
        run_name=f"pusht_{tag}",
        batch_size=args.batch_size
    )

    # 4. Lưu checkpoint phục vụ đánh giá Rollout & MPC
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"pusht_{tag}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "use_phys_token": use_phys_token,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "pred_step_k": args.pred_step_k,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "heads": args.heads,
        "size_mib": size_mib,
        "losses": losses
    }, ckpt_path)
    print(f"✅ Đã lưu trọng số tại: {ckpt_path}")

    return steps, losses, size_mib


def main():
    parser = argparse.ArgumentParser(description="Train PushT Flow Matching World Model")
    parser.add_argument("--zarr_path", type=str, default=None, help="Đường dẫn tới pusht_cchi_v7_replay.zarr (để trống nếu chạy Mock test)")
    parser.add_argument("--img_size", type=int, default=32, help="Độ phân giải khung hình (32 hoặc 64)")
    parser.add_argument("--patch_size", type=int, default=4, help="Kích thước patch của Patchifier")
    parser.add_argument("--pred_step_k", type=int, default=4, help="Số bước gộp hành động (Action Chunking horizon k)")
    parser.add_argument("--hidden_dim", type=int, default=192, help="Số chiều ẩn của Transformer")
    parser.add_argument("--depth", type=int, default=4, help="Số lớp Transformer")
    parser.add_argument("--heads", type=int, default=6, help="Số lượng Attention Heads")
    parser.add_argument("--num_steps", type=int, default=3000, help="Số bước huấn luyện")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cfg_eta", type=float, default=0.1, help="Xác suất dropout điều kiện cho CFG")
    parser.add_argument("--phys_weight", type=float, default=0.2, help="Trọng số phạt vật lý tiếp xúc")
    parser.add_argument("--mode", type=str, default="both", choices=["phys", "base", "both"], help="Chế độ chạy: chỉ chạy mô hình có [PHYS_TOKEN], không có, hoặc cả hai")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Thư mục lưu model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Thiết bị sử dụng: {device}")

    # Khởi tạo Sampler dùng chung cho cả 2 lần chạy
    sampler = PushTWorldSampler(
        zarr_path=args.zarr_path,
        img_size=args.img_size,
        pred_step_k=args.pred_step_k
    ).to(device)

    results = {}
    if args.mode in ["base", "both"]:
        results["Baseline Video DiT (No Phys Token)"] = train_single_config(False, args, device, sampler)

    if args.mode in ["phys", "both"]:
        results["Physically-Grounded DiT (+[PHYS_TOKEN])"] = train_single_config(True, args, device, sampler)

    # Vẽ biểu đồ Loss so sánh
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for label, (steps, losses, size_mib) in results.items():
        # Làm mượt đường loss bằng trung bình trượt
        window = max(1, len(losses) // 50)
        smoothed = [sum(losses[max(0, i-window):i+1]) / len(losses[max(0, i-window):i+1]) for i in range(len(losses))]
        plt.plot(steps, smoothed, label=f"{label} ({size_mib:.2f} MiB)", linewidth=2)

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("PushT World Model Training Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_loss_comparison.png", dpi=200)
    print("Đã xuất biểu đồ hội tụ tại: results/training_loss_comparison.png")


if __name__ == "__main__":
    main()