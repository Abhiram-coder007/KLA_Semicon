import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Visualize test restorations side-by-side.")
    parser.add_argument(
    "--degraded_dir",
    type=Path,
    required=True,
    help="Directory containing degraded .npy images.",
    )
    parser.add_argument(
        "--restored_dir",
        type=Path,
        required=True,
        help="Directory containing restored .npy images.",
    )
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()

    deg_files = sorted(args.degraded_dir.glob("*.npy"))
    
    if not deg_files:
        raise FileNotFoundError(f"No degraded .npy files found in {args.degraded_dir}")

    for i in range(min(args.num_images, len(deg_files))):
        deg_path = deg_files[i]
        rest_path = args.restored_dir / deg_path.name
        
        if not rest_path.exists():
            print(f"⚠️ Could not find restored file for {deg_path.name}")
            continue

        # Load and format the arrays
        deg_img = np.clip(np.load(deg_path).squeeze(), 0.0, 1.0)
        rest_img = np.clip(np.load(rest_path).squeeze(), 0.0, 1.0)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(deg_img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Degraded Input ({deg_path.name})", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(rest_img, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Your Model's Restoration", fontsize=14)
        axes[1].axis("off")

        plt.tight_layout()
        
        # Save the visualization
        viz_dir = ROOT_DIR / "results" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        out_name = viz_dir / f"test_viz_{i:04d}.png"
        fig.savefig(out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"✓ Saved comparison to: {out_name}")

if __name__ == "__main__":
    main()