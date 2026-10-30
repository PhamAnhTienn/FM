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
    tag = "phys" if use_phys_token else "base"
    print(f"\n{'='*60}")
    print(f"Start training: {tag.upper()} (use_phys_token={use_phys_token})")
    print(f"{'='*60}")

    # 1. Init probs path with optimal transport (alpha_t = t, beta_t = 1 - t)
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[3, args.img_size, args.img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    # 2. Init model
    cond_dim = 5 + 2 * args.pred_step_k  # 5 state + 2*k action chunk
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
    print(f"Model size ({tag}): {size_mib:.3f} MiB")

    # 3. Training
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

    # 4. Save checkpoint for rollout & mpc evaluation
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
    print(f"Saved checkpoint at: {ckpt_path}")

    return steps, losses, size_mib


def main():
    parser = argparse.ArgumentParser(description="Train PushT Flow Matching World Model")
    parser.add_argument("--zarr_path", type=str, default=None, help="Path to pusht_cchi_v7_replay.zarr (leave empty for mock test)")
    parser.add_argument("--img_size", type=int, default=32, help="Image resolution (32 or 64)")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size of Patchifier")
    parser.add_argument("--pred_step_k", type=int, default=4, help="Number of action chunks (Action Chunking horizon k)")
    parser.add_argument("--hidden_dim", type=int, default=192, help="Hidden dimension of Transformer")
    parser.add_argument("--depth", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--heads", type=int, default=6, help="Number of Attention Heads")
    parser.add_argument("--num_steps", type=int, default=3000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--cfg_eta", type=float, default=0.1, help="Conditional dropout probability for CFG")
    parser.add_argument("--phys_weight", type=float, default=0.2, help="Physics penalty weight for contact")
    parser.add_argument("--mode", type=str, default="both", choices=["phys", "base", "both"], help="Run mode: only phys token, no phys token, or both")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/pusht_fm", help="Checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Sampler
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

    # Draw loss curve
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for label, (steps, losses, size_mib) in results.items():
        # Smooth loss curve with moving average
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
    print("Saved training loss curve at: results/training_loss_comparison.png")


if __name__ == "__main__":
    main()