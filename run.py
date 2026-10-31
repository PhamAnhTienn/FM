#!/usr/bin/env python3
"""
End-to-End Benchmark & Visualization Suite for Flow Matching README
===================================================================
Runs all 3 experiments needed for the README and prints ready-to-paste Markdown tables:
  1. Backbone & Token Scaling (MLP vs. DiT-S/8 vs. DiT-S/4): Params, MiB, Final Loss, Latency, Acc & FID
  2. Computational Efficiency (NFE Sweep: 5, 10, 25, 50, 100 Euler steps): Batch Latency & FID/Acc
  3. Continuous-Time Trajectory Rollout & CFG Visuals: Saves `assets/dit_cfg_samples.png`
     and `assets/ode_trajectory_rollout.png` using `simulate_with_trajectory`.

Usage:
  # Run full pipeline (trains evaluator + trains/evaluates MLP, DiT-S/8, DiT-S/4 + generates plots)
  python run.py --data_root ./data

  # Or evaluate existing checkpoints without retraining:
  python run.py \
      --skip_train \
      --mlp_ckpt runs/mnist_mlp/final_model.pt \
      --dit_p8_ckpt runs/mnist_dit_p8/final_model.pt \
      --dit_p4_ckpt runs/mnist_dit_p4/final_model.pt
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Support both `fm` (actual package directory) and `flow_wm` (legacy import name)
try:
    from fm.core.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
    from fm.core.solvers import CFGVectorFieldODE, EulerSimulator
    from fm.envs.mnist_sampler import MNISTSampler
    from fm.models.mlp import MLPConditionalVectorField
    from fm.models.dit import DiffusionTransformerFlowModel
    from fm.trainers.cfg_trainer import MNISTCFGTrainer, visualize_output
    from fm.trainers.base_trainer import model_size_b, MiB
except ImportError:
    from flow_wm.core.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
    from flow_wm.core.solvers import CFGVectorFieldODE, EulerSimulator
    from flow_wm.envs.mnist_sampler import MNISTSampler
    from flow_wm.models.mlp import MLPConditionalVectorField
    from flow_wm.models.dit import DiffusionTransformerFlowModel
    from flow_wm.trainers.cfg_trainer import MNISTCFGTrainer, visualize_output
    from flow_wm.trainers.base_trainer import model_size_b, MiB


# =====================================================================
# 0. Lightweight MNIST Feature Extractor & Classifier (for FID & Acc)
# =====================================================================
class MNISTEvalCNN(nn.Module):
    """Lightweight CNN trained on 32x32 MNIST to measure Conditional Acc (%) and Feature-FID."""

    def __init__(self, feat_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 128 x 2 x 2 = 512
        )
        self.fc_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feat_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_feat(self.conv(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits, feat


def get_or_train_evaluator(
    sampler: MNISTSampler,
    device: torch.device,
    ckpt_path: str = "runs/eval_cnn_32x32.pt",
    epochs: int = 3,
) -> MNISTEvalCNN:
    """Trains or loads a fast 32x32 MNIST evaluator (~99% accuracy in 3 epochs)."""
    evaluator = MNISTEvalCNN().to(device)
    if os.path.exists(ckpt_path):
        evaluator.load_state_dict(torch.load(ckpt_path, map_location=device))
        evaluator.eval()
        return evaluator

    print("[Evaluator] Training lightweight MNIST CNN for FID & Conditional Accuracy...")
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    loader = DataLoader(sampler.dataset, batch_size=256, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(evaluator.parameters(), lr=1e-3, weight_decay=1e-4)

    evaluator.train()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = evaluator(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

    evaluator.eval()
    torch.save(evaluator.state_dict(), ckpt_path)
    print(f"[Evaluator] Saved MNIST evaluator checkpoint to {ckpt_path}")
    return evaluator


@torch.no_grad()
def compute_frechet_distance(feat_real: torch.Tensor, feat_gen: torch.Tensor) -> float:
    """Computes exact Fréchet Distance between two Gaussian feature distributions in pure PyTorch."""
    x_r = feat_real.to(torch.float64)
    x_g = feat_gen.to(torch.float64)

    mu_r = x_r.mean(dim=0)
    mu_g = x_g.mean(dim=0)
    diff = mu_r - mu_g

    cov_r = torch.cov(x_r.T) + torch.eye(x_r.shape[1], dtype=torch.float64, device=x_r.device) * 1e-6
    cov_g = torch.cov(x_g.T) + torch.eye(x_g.shape[1], dtype=torch.float64, device=x_g.device) * 1e-6

    # Symmetric matrix square root via eigendecomposition: cov_r^{1/2} cov_g cov_r^{1/2}
    eigvals_r, eigvecs_r = torch.linalg.eigh(cov_r)
    sqrt_cov_r = eigvecs_r @ torch.diag(torch.clamp(eigvals_r, min=0.0).sqrt()) @ eigvecs_r.T

    prod = sqrt_cov_r @ cov_g @ sqrt_cov_r
    eigvals_p, _ = torch.linalg.eigh(prod)
    tr_covmean = torch.clamp(eigvals_p, min=0.0).sqrt().sum()

    fid = diff.dot(diff) + torch.trace(cov_r) + torch.trace(cov_g) - 2.0 * tr_covmean
    return float(torch.clamp(fid, min=0.0).item())


# =====================================================================
# 1. Sampling, Timing & Quality Evaluation Helper
# =====================================================================
@torch.no_grad()
def evaluate_flow_model(
    model: nn.Module,
    path: GaussianConditionalProbabilityPath,
    evaluator: MNISTEvalCNN,
    real_features: torch.Tensor,
    num_timesteps: int = 25,
    guidance_scale: float = 3.0,
    num_eval_samples: int = 1000,
    batch_size: int = 100,
    null_label: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Benchmarks latency (ms/batch and ms/sample), Conditional Accuracy (%), and FID."""
    model.eval()
    ode = CFGVectorFieldODE(model, guidance_scale=guidance_scale, null_label=null_label)
    simulator = EulerSimulator(ode)

    # Warmup pass for accurate GPU timing
    x_warm = path.p_simple.sample(10).view(10, 1, 32, 32)
    y_warm = torch.arange(10, device=device)
    ts_warm = torch.linspace(0.0, 0.999, num_timesteps, device=device).unsqueeze(0).expand(10, -1)
    _ = simulator.simulate(x_warm, ts_warm, y=y_warm, use_tqdm=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    all_feats = []
    correct = 0
    total = 0
    total_time_ms = 0.0
    curr_batch_size = min(batch_size, num_eval_samples)
    curr_batch_size = max(10, (curr_batch_size // 10) * 10)
    num_batches = max(1, num_eval_samples // curr_batch_size)

    for _ in range(num_batches):
        # Balanced class conditioning (0..9 repeated)
        y = torch.arange(10, device=device).repeat(curr_batch_size // 10)
        x0 = path.p_simple.sample(curr_batch_size).view(curr_batch_size, 1, 32, 32)
        ts = torch.linspace(0.0, 0.999, num_timesteps, device=device).unsqueeze(0).expand(curr_batch_size, -1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        x1 = simulator.simulate(x0, ts, y=y, use_tqdm=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time_ms += (time.perf_counter() - t0) * 1000.0

        logits, feats = evaluator(x1)
        preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += curr_batch_size
        all_feats.append(feats)

    gen_features = torch.cat(all_feats, dim=0)
    fid = compute_frechet_distance(real_features, gen_features)
    acc = 100.0 * correct / max(1, total)
    batch_latency_ms = total_time_ms / num_batches
    sample_latency_ms = batch_latency_ms / curr_batch_size

    return {
        "batch_latency_ms": batch_latency_ms,
        "sample_latency_ms": sample_latency_ms,
        "acc": acc,
        "fid": fid,
    }


# =====================================================================
# 2. Experiment 3 Visuals: Trajectory Rollout & CFG Comparison Grid
# =====================================================================
@torch.no_grad()
def generate_readme_figures(
    model: nn.Module,
    path: GaussianConditionalProbabilityPath,
    null_label: int = 10,
    assets_dir: str = "assets",
    device: torch.device = torch.device("cpu"),
):
    """Generates both README hero images: `dit_cfg_samples.png` and `ode_trajectory_rollout.png`."""
    os.makedirs(assets_dir, exist_ok=True)
    model.eval()

    # 1. CFG Grid across w in {1.0, 3.0, 5.0}
    cfg_path = os.path.join(assets_dir, "dit_cfg_samples.png")
    visualize_output(
        model=model,
        path=path,
        samples_per_class=8,
        num_timesteps=50,
        guidance_scales=[1.0, 3.0, 5.0],
        null_label=null_label,
        save_path=cfg_path,
        use_tqdm=False,
        device=device,
    )
    print(f"[Visuals] Saved CFG comparison grid -> {cfg_path}")

    # 2. Continuous-Time ODE Trajectory Rollout (t = 0.0 -> 1.0) using simulate_with_trajectory
    traj_path = os.path.join(assets_dir, "ode_trajectory_rollout.png")
    num_timesteps = 51  # indices 0, 10, 20, 30, 40, 50 correspond to t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    snap_indices = [0, 10, 20, 30, 40, 50]
    snap_labels = ["t=0.0", "t=0.2", "t=0.4", "t=0.6", "t=0.8", "t=1.0"]

    ode = CFGVectorFieldODE(model, guidance_scale=3.0, null_label=null_label)
    simulator = EulerSimulator(ode)

    # 10 digits (0..9), each row is a digit, each column is a time snapshot t
    y = torch.arange(10, device=device, dtype=torch.int64)
    x0 = path.p_simple.sample(10).view(10, 1, 32, 32)
    ts = torch.linspace(0.0, 0.999, num_timesteps, device=device).unsqueeze(0).expand(10, -1)

    # x_traj shape: (10, num_timesteps, 1, 32, 32)
    x_traj = simulator.simulate_with_trajectory(x0, ts, y=y, use_tqdm=False)
    snaps = x_traj[:, snap_indices]  # (10, 6, 1, 32, 32)

    # Normalize each snapshot column for clean visual progression
    snaps_flat = snaps.reshape(-1, 1, 32, 32)
    v_min = snaps_flat.amin(dim=(1, 2, 3), keepdim=True)
    v_max = snaps_flat.amax(dim=(1, 2, 3), keepdim=True)
    snaps_norm = (snaps_flat - v_min) / (v_max - v_min + 1e-8)

    grid = make_grid(snaps_norm, nrow=len(snap_indices), padding=2, normalize=False)

    fig, ax = plt.subplots(figsize=(8, 11))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.set_title("Continuous-Time Euler ODE Rollout ($t = 0.0 \\rightarrow 1.0$, $w = 3.0$)", fontsize=14, pad=12)

    # Add column headers for time t
    col_width = grid.shape[2] / len(snap_indices)
    for idx, t_str in enumerate(snap_labels):
        ax.text(
            (idx + 0.5) * col_width,
            -4,
            t_str,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visuals] Saved ODE trajectory rollout -> {traj_path}")


# =====================================================================
# 3. Main Benchmark Runner
# =====================================================================
def train_or_load_model(
    name: str,
    model: nn.Module,
    path: GaussianConditionalProbabilityPath,
    num_steps: int,
    lr: float,
    warmup_steps: int,
    eta: float,
    batch_size: int,
    ckpt_path: Optional[str],
    skip_train: bool,
    device: torch.device,
) -> float:
    """Trains a flow model (or loads checkpoint) and returns its smoothed final CFM loss."""
    if skip_train and ckpt_path and os.path.exists(ckpt_path):
        print(f"[{name}] Loading existing checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        # Estimate final loss over 20 batches
        trainer = MNISTCFGTrainer(path=path, eta=eta, null_label=10)
        trainer.model = model
        model.eval()
        with torch.no_grad():
            losses = [float(trainer.get_train_loss(batch_size=batch_size).item()) for _ in range(20)]
        return sum(losses) / len(losses)

    print(f"\n[{name}] Training for {num_steps} steps (batch_size={batch_size}, lr={lr})...")
    trainer = MNISTCFGTrainer(path=path, eta=eta, null_label=10)
    losses, _ = trainer.train(
        model=model,
        num_steps=num_steps,
        lr=lr,
        warmup_steps=warmup_steps,
        ckpt_every=max(1000, num_steps // 5),
        run_name=name,
        batch_size=batch_size,
    )
    # Save final checkpoint explicitly
    final_ckpt = os.path.join(trainer.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[{name}] Saved final checkpoint to {final_ckpt}")

    window = min(100, len(losses))
    return sum(losses[-window:]) / window


def main():
    parser = argparse.ArgumentParser(description="Run all 3 README experiments & generate figures")
    parser.add_argument("--data_root", type=str, default="./data", help="MNIST data root")
    parser.add_argument("--mlp_steps", type=int, default=5000, help="Training steps for MLP")
    parser.add_argument("--dit_steps", type=int, default=20000, help="Training steps for DiT-S/8 and DiT-S/4")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples for FID/Acc evaluation")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and load checkpoints")
    parser.add_argument("--mlp_ckpt", type=str, default=None, help="Path to MLP checkpoint if --skip_train")
    parser.add_argument("--dit_p8_ckpt", type=str, default=None, help="Path to DiT-S/8 checkpoint if --skip_train")
    parser.add_argument("--dit_p4_ckpt", type=str, default=None, help="Path to DiT-S/4 checkpoint if --skip_train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on device: {device}")

    # 1. Dataset & Optimal Transport Probability Path
    sampler = MNISTSampler(root=args.data_root, train=True, img_size=32, download=True).to(device)
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[1, 32, 32],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    # 2. Evaluator CNN & Real Feature Distribution (for FID)
    evaluator = get_or_train_evaluator(sampler, device=device)
    with torch.no_grad():
        real_imgs, _ = sampler.sample(args.eval_samples)
        real_features = evaluator.extract_features(real_imgs)

    # 3. Instantiate the 3 Models for Experiment 1
    models_cfg = [
        (
            "mnist_mlp",
            "MLP (4-Layer)",
            "—",
            "—",
            MLPConditionalVectorField(dim=32 * 32, hidden_dim=512, class_dim=64, num_classes=10, num_layers=4).to(device),
            args.mlp_steps,
            1e-3,
            200,
            0.25,
            args.mlp_ckpt,
        ),
        (
            "mnist_dit_p8",
            "DiT-S/8 (8-Layer)",
            "$8 \\times 8$",
            "16",
            DiffusionTransformerFlowModel(
                img_size=32, patch_size=8, num_layers=8, c=1, dim=256, heads=8, final_dim=10, n_classes=11
            ).to(device),
            args.dit_steps,
            4e-4,
            500,
            0.35,
            args.dit_p8_ckpt,
        ),
        (
            "mnist_dit_p4",
            "DiT-S/4 (8-Layer)",
            "$4 \\times 4$",
            "64",
            DiffusionTransformerFlowModel(
                img_size=32, patch_size=4, num_layers=8, c=1, dim=256, heads=8, final_dim=10, n_classes=11
            ).to(device),
            args.dit_steps,
            4e-4,
            500,
            0.35,
            args.dit_p4_ckpt,
        ),
    ]

    exp1_rows = []
    dit_p4_model = None

    for run_name, label, patch_str, tokens_str, model, steps, lr, warmup, eta, ckpt in models_cfg:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        mem_mib = model_size_b(model) / MiB

        final_loss = train_or_load_model(
            name=run_name,
            model=model,
            path=path,
            num_steps=steps,
            lr=lr,
            warmup_steps=warmup,
            eta=eta,
            batch_size=args.batch_size,
            ckpt_path=ckpt,
            skip_train=args.skip_train,
            device=device,
        )

        metrics = evaluate_flow_model(
            model=model,
            path=path,
            evaluator=evaluator,
            real_features=real_features,
            num_timesteps=25,
            guidance_scale=3.0,
            num_eval_samples=args.eval_samples,
            batch_size=100,
            device=device,
        )

        exp1_rows.append({
            "label": label,
            "patch": patch_str,
            "tokens": tokens_str,
            "params": f"{params_m:.2f}M",
            "mib": f"{mem_mib:.2f}",
            "loss": f"{final_loss:.4f}",
            "latency": f"{metrics['sample_latency_ms']:.2f} ms",
            "acc_fid": f"{metrics['acc']:.1f}% / {metrics['fid']:.2f}",
        })

        if run_name == "mnist_dit_p4":
            dit_p4_model = model

    # 4. Run Experiment 2: NFE Sweep on DiT-S/4
    nfe_list = [5, 10, 25, 50, 100]
    exp2_results = {}
    print("\n[Experiment 2] Running Euler ODE Steps (NFE) Sweep on DiT-S/4 (w=3.0)...")
    for nfe in nfe_list:
        m = evaluate_flow_model(
            model=dit_p4_model,
            path=path,
            evaluator=evaluator,
            real_features=real_features,
            num_timesteps=nfe,
            guidance_scale=3.0,
            num_eval_samples=args.eval_samples,
            batch_size=100,
            device=device,
        )
        exp2_results[nfe] = m

    # 5. Run Experiment 3: Generate Hero Visuals for README
    print("\n[Experiment 3] Generating CFG & Continuous-Time Trajectory figures...")
    generate_readme_figures(model=dit_p4_model, path=path, assets_dir="assets", device=device)

    # 6. Print Ready-to-Paste Markdown Tables for README.md
    print("\n" + "=" * 80)
    print("COPY & PASTE INTO README.md -> SECTION 1 (Backbone Scaling Table)")
    print("=" * 80)
    print(
        "| Backbone | Patch Size ($p$) | Tokens ($N$) | Params | Memory (MiB) | Final CFM Loss ($\\downarrow$) | Latency @ 25 NFE (ms/sample $\\downarrow$) | Classifier Acc. / FID ($\\uparrow$ / $\\downarrow$) |"
    )
    print("| :--- | :---: | :---: | right | right | right | right | right |")
    for r in exp1_rows:
        print(
            f"| **{r['label']}** | {r['patch']} | {r['tokens']} | {r['params']} | {r['mib']} | `{r['loss']}` | `{r['latency']}` | `{r['acc_fid']}` |"
        )

    print("\n" + "=" * 80)
    print("COPY & PASTE INTO README.md -> SECTION 2 (NFE Trade-off Table)")
    print("=" * 80)
    print("| Euler Steps (NFE) | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |")
    print("| :--- | right | right | right | right | right |")
    lat_str = " | ".join([f"`{exp2_results[n]['batch_latency_ms']:.1f} ms`" for n in nfe_list])
    fid_str = " | ".join([f"`{exp2_results[n]['fid']:.2f} ({exp2_results[n]['acc']:.1f}%)`" for n in nfe_list])
    print(f"| **Batch Latency (ms, $B=100$)** | {lat_str} |")
    print(f"| **Sample Fidelity (FID $\\downarrow$ / Acc $\\uparrow$)** | {fid_str} |")
    print("=" * 80)


if __name__ == "__main__":
    main()