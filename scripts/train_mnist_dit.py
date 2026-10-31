import os
import argparse
import matplotlib.pyplot as plt
import torch

from flow_wm.core.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from flow_wm.envs.mnist_sampler import MNISTSampler
from flow_wm.models.dit import DiffusionTransformerFlowModel
from flow_wm.trainers.cfg_trainer import MNISTCFGTrainer, visualize_output
from flow_wm.trainers.base_trainer import model_size_b, MiB


def main():
    parser = argparse.ArgumentParser(description="Train MNIST Diffusion Transformer (DiT) Flow Matching Model")
    parser.add_argument("--data_root", type=str, default="./data", help="Path to MNIST data directory")
    parser.add_argument("--img_size", type=int, default=32, help="Image resolution (32x32)")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size of Patchifier (e.g., 4 or 8)")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of DiT Transformer layers")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension of DiT")
    parser.add_argument("--heads", type=int, default=8, help="Number of Attention heads")
    parser.add_argument("--final_dim", type=int, default=10, help="Final dimension for Depatchifier")
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--eta", type=float, default=0.35, help="CFG label dropout probability")
    parser.add_argument("--null_label", type=int, default=10, help="Null label token for unguided condition")
    parser.add_argument("--ckpt_every", type=int, default=1000, help="Save checkpoint and sample every N steps")
    parser.add_argument("--run_name", type=str, default="mnist_dit", help="Run name for output directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output dir override")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset Sampler
    print(f"Loading MNIST dataset from {args.data_root}...")
    sampler = MNISTSampler(
        root=args.data_root,
        train=True,
        img_size=args.img_size,
        download=True
    ).to(device)

    # 2. Probability Path (OT path: alpha_t = t, beta_t = 1 - t)
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[1, args.img_size, args.img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    # 3. DiT Flow Model
    model = DiffusionTransformerFlowModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        c=1,
        dim=args.dim,
        heads=args.heads,
        final_dim=args.final_dim,
        n_classes=args.null_label + 1,
    ).to(device)

    size_mib = model_size_b(model) / MiB
    print(f"DiffusionTransformerFlowModel size: {size_mib:.3f} MiB")

    # 4. Trainer
    trainer = MNISTCFGTrainer(
        path=path,
        eta=args.eta,
        null_label=args.null_label,
    )

    print(f"Starting DiT Flow Matching training for {args.num_steps} steps...")
    losses, steps = trainer.train(
        model=model,
        num_steps=args.num_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        ckpt_every=args.ckpt_every,
        run_name=args.run_name,
        batch_size=args.batch_size,
    )

    # 5. Final Visualization & Loss Curve
    out_dir = trainer.output_dir
    os.makedirs(out_dir, exist_ok=True)

    final_img_path = os.path.join(out_dir, "final_samples.png")
    visualize_output(
        model=model,
        path=path,
        samples_per_class=10,
        num_timesteps=100,
        guidance_scales=[1.0, 3.0, 5.0],
        null_label=args.null_label,
        save_path=final_img_path,
        use_tqdm=True,
    )
    print(f"Saved generated samples to {final_img_path}")

    loss_plot_path = os.path.join(out_dir, "loss_curve.png")
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, label="Train Loss", alpha=0.7)
    window = max(1, len(losses) // 50)
    if len(losses) > window:
        smoothed = [sum(losses[max(0, i-window):i+1]) / len(losses[max(0, i-window):i+1]) for i in range(len(losses))]
        plt.plot(steps, smoothed, label="Smoothed Loss", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MNIST DiT Flow Matching Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")


if __name__ == "__main__":
    main()
