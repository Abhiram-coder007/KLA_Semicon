import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError as error:
    raise SystemExit(
        "scikit-image is required for PSNR/SSIM. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from error

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset import KLARestorationDataset
from run import select_device
from model import NAFNetDWT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NAFNetDWT on held-out validation pairs."
    )

    parser.add_argument(
        "--degraded_dir",
        type=Path,
        default=ROOT_DIR / "data" / "val_degraded",
        help="Validation degraded-image directory.",
    )

    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=ROOT_DIR / "data" / "val_gt",
        help="Validation ground-truth directory.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR / "weights" / "kla_model_final.pth",
        help="Model checkpoint path.",
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT_DIR / "results" / "validation_metrics.json",
        help="Path for the machine-readable metric report.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Evaluation device (default: auto).",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dataset = KLARestorationDataset(args.degraded_dir, args.gt_dir)
    if not dataset:
        raise ValueError("Validation dataset is empty.")

    device = select_device(args.device)
 
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_kwargs = checkpoint.get("model_kwargs", {})
    model = NAFNetDWT(**model_kwargs).to(device)

    raw_state_dict = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict"))
    
    #Strip the 'module.' prefix left over from Kaggle's Multi-GPU setup
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        if k == "n_averaged":  # Skip the EMA step counter
            continue
        clean_key = k.replace("module.", "")
        clean_state_dict[clean_key] = v
        
    #Load the cleaned weights into the model
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_image = []
    
    with torch.inference_mode():
        for degraded, ground_truth, filenames in dataloader:
            
            # Standard 1-view forward pass 
            prediction = model(degraded.to(device))
            
            prediction_np = prediction.squeeze().float().cpu().numpy().clip(0.0, 1.0)
            target_np = ground_truth.squeeze().cpu().numpy().clip(0.0, 1.0)
            
            if prediction_np.shape != target_np.shape:
                raise RuntimeError(
                    f"Prediction shape {prediction_np.shape} does not match target shape {target_np.shape}."
                )

            image_psnr = float(peak_signal_noise_ratio(target_np, prediction_np, data_range=1.0))
            
            # Standardized SSIM calculation using Gaussian window parameters matching literature
            image_ssim = float(
                structural_similarity(
                    target_np, 
                    prediction_np, 
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False
                )
            )
            per_image.append(
                {"filename": filenames[0], "psnr_db": image_psnr, "ssim": image_ssim}
            )

    psnr_values = np.array([item["psnr_db"] for item in per_image])
    ssim_values = np.array([item["ssim"] for item in per_image])
    
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "image_count": len(per_image),
        "psnr_db_mean": float(psnr_values.mean()),
        "psnr_db_std": float(psnr_values.std(ddof=0)),
        "ssim_mean": float(ssim_values.mean()),
        "ssim_std": float(ssim_values.std(ddof=0)),
        "per_image": per_image,
    }
    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Validation images: {report['image_count']}")
    print(f"Mean PSNR: {report['psnr_db_mean']:.3f} dB ± {report['psnr_db_std']:.3f}")
    print(f"Mean SSIM: {report['ssim_mean']:.5f} ± {report['ssim_std']:.5f}")
    print(f"Saved report: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
