import os
import argparse
import torch
import matplotlib.pyplot as plt

from flow_wm.core.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from flow_wm.core.distributions import IsotropicGaussian
from flow_wm.models.dit import DiffusionTransformerFlowModel
from flow_wm.models.mlp import MLPConditionalVectorField
from flow_wm.trainers.cfg_trainer import visualize_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate / Sample MNIST Flow Matching Model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model_type", type=str, default="dit", choices=["dit", "mlp"], help="Model architecture")
    parser.add_argument("--img_size", type=int, default=32, help="Image resolution")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for DiT")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads (for DiT)")
    parser.add_argument("--num_layers", type=int, default=8, help="Layers (for DiT) or MLP layers")
    parser.add_argument("--samples_per_class", type=int, default=10, help="Samples per digit class")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of ODE solver steps")
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[1.0, 3.0, 5.0], help="CFG scales")
    parser.add_argument("--null_label", type=int, default=10, help="Null label index")
    parser.add_argument("--output_path", type=str, default="results/mnist_eval.png", help="Output image file path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    if args.model_type == "dit":
        model = DiffusionTransformerFlowModel(
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            c=1,
            dim=args.dim,
            heads=args.heads,
            final_dim=10,
            n_classes=args.null_label + 1,
        ).to(device)
    else:
        flat_dim = args.img_size * args.img_size
        model = MLPConditionalVectorField(
            dim=flat_dim,
            hidden_dim=args.dim,
            class_dim=64,
            num_classes=args.null_label,
            num_layers=args.num_layers,
        ).to(device)

    # Load checkpoint
    print(f"Loading weights from {args.ckpt_path}...")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    # Probability path
    path = GaussianConditionalProbabilityPath(
        p_data=None,
        p_simple_shape=[1, args.img_size, args.img_size],
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    print(f"Simulating ODE generation with guidance scales: {args.guidance_scales}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    visualize_output(
        model=model,
        path=path,
        samples_per_class=args.samples_per_class,
        num_timesteps=args.num_timesteps,
        guidance_scales=args.guidance_scales,
        null_label=args.null_label,
        save_path=args.output_path,
        use_tqdm=True,
        device=device,
    )
    print(f"Saved evaluation samples to: {args.output_path}")


if __name__ == "__main__":
    main()
