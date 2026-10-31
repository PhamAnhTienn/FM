# Conditional Flow Matching with MLP & Diffusion Transformer (DiT)

A PyTorch reimplementation of Conditional Flow Matching (CFM) with Optimal Transport (OT) probability paths and Classifier-Free Guidance (CFG). It compares MLP and Diffusion Transformer (DiT) backbones with further experimentations evaluating sample quality and inference efficiency across different numbers of Euler ODE steps (NFE).

<p align="center">
  <img src="assets/dit_cfg_samples.png" width="48%" alt="DiT CFG Samples (w=1.0, 3.0, 5.0)" />
  <img src="assets/ode_trajectory_rollout.png" width="48%" alt="Continuous-Time ODE Trajectory Rollout (t=0 to t=1)" />
</p>

## Repository Structure

```text
├── fm/
│   ├── core/
│   │   ├── distributions.py   # Noise distributions and sampling interfaces
│   │   ├── paths.py           # Conditional probability paths
│   │   └── solvers.py         # ODE/SDE solvers and CFG integration
│   ├── models/
│   │   ├── layers.py          # Shared layers and DiT components
│   │   ├── mlp.py             # MLP vector field
│   │   └── dit.py             # Diffusion Transformer vector field
│   ├── trainers/
│   │   ├── base_trainer.py    # Base training loop
│   │   └── cfg_trainer.py     # Classifier-Free Guidance training
│   └── envs/
│       └── mnist_sampler.py   # MNIST data sampler
└── scripts/
    ├── train_mnist_mlp.py     # Train MLP model
    ├── train_mnist_dit.py     # Train DiT model
    └── eval_mnist.py          # Evaluate models and sampling configurations
```

## Quick Start

```bash
# Install the package
pip install -e .

# Train MLP baseline
python scripts/train_mnist_mlp.py --num_steps 5000 --batch_size 256 --lr 1e-3 --eta 0.25

# Train DiT
python scripts/train_mnist_dit.py --patch_size 4 --num_layers 8 --dim 256 --heads 8 --num_steps 20000

# Evaluate with different ODE steps and guidance scales
python scripts/eval_mnist.py \
    --ckpt_path runs/mnist_dit/step_020000_model.pt \
    --model_type dit \
    --num_timesteps 25 \
    --guidance_scales 1.0 3.0 5.0 \
    --output_path results/dit_nfe25_eval.png
```

## Experiments & Benchmarks

### 1. Backbone Scaling & Computational Efficiency (MLP vs. DiT)

All models are trained on $32 \times 32$ MNIST with batch size $256$ and evaluated using an Euler ODE solver at guidance scale $w = 3.0$.

| Backbone | Patch Size ($p$) | Tokens ($N$) | Params | Memory (MiB) | Final CFM Loss ($\downarrow$) | Latency @ 25 NFE (ms/sample $\downarrow$) | Classifier Acc. / FID ($\uparrow$ / $\downarrow$) |
| :--- | :---: | :---: | right | right | right | right | right |
| **MLP (4-Layer)** | — | — | 1.61M | 6.14 | `[e.g., 0.182]` | `[e.g., 0.4 ms]` | `[e.g., 88.4% / 18.2]` |
| **DiT-S/8 (8-Layer)** | $8 \times 8$ | 16 | 9.91M | 37.80 | `[e.g., 0.134]` | `[e.g., 1.1 ms]` | `[e.g., 96.8% / 6.4]` |
| **DiT-S/4 (8-Layer)** | $4 \times 4$ | 64 | 9.92M | 37.84 | `[e.g., 0.115]` | `[e.g., 2.8 ms]` | `[e.g., 98.5% / 3.9]` |

### 2. ODE Integration Steps (NFE) vs. Rollout Quality (DiT-S/4, $w = 3.0$)

Because Optimal Transport (linear) probability paths encourage straight velocity trajectories, Euler integration maintains high sample fidelity even under aggressive step reduction:

| Euler Steps (NFE) | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |
| :--- | right | right | right | right | right |
| **Batch Latency (ms, $B=100$)** | `[e.g., 14 ms]` | `[e.g., 28 ms]` | `[e.g., 70 ms]` | `[e.g., 140 ms]` | `[e.g., 280 ms]` |
| **Sample Fidelity (FID $\downarrow$)** | `[e.g., 14.5]` | `[e.g., 6.8]` | `[e.g., 4.1]` | `[e.g., 3.9]` | `[e.g., 3.8]` |

### 3. Classifier-Free Guidance ($w$) Ablation

- **$w = 1.0$ (Conditional, Unamplified)**: High intra-class diversity with occasional ambiguous stroke boundaries.
- **$w = 3.0$ (Balanced Guidance)**: Optimal trade-off between structural sharpness, class consistency, and stroke diversity.
- **$w = 5.0$ (Strong Guidance)**: High-contrast canonical digit modes with reduced variance across trajectories.
