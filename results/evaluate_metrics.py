import argparse
from pathlib import Path
import torch
import lpips
from pytorch_msssim import ms_ssim
from tqdm import tqdm

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


from dataset import KLARestorationDataset
from torch.utils.data import DataLoader
from run import select_device
from model import NAFNetDWT

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate Perceptual Metrics (LPIPS and MS-SSIM).")
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
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Evaluation device (default: auto).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    #Setup device and dataset
    device = select_device(args.device)
    print(f"Evaluating on: {device}")
    
    val_dataset = KLARestorationDataset(args.degraded_dir, args.gt_dir)
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")
        
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    #Extract model arguments and initialize model
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = NAFNetDWT(**model_kwargs).to(device)
    
    # Grab the smoothed EMA weights (or fallback to standard weights)
    raw_state_dict = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict"))
    
    # Strip the 'module.' prefix left over from Kaggle's Multi-GPU setup
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        if k == "n_averaged":  # Skip the EMA step counter
            continue
        clean_key = k.replace("module.", "")
        clean_state_dict[clean_key] = v
        
    # Load the cleaned weights into the model
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    # Initialize Perceptual (LPIPS) model
    print("Loading VGG perceptual model for LPIPS...")
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    print(f"Starting evaluation on {len(val_dataset)} images...")
    
    # Evaluation Loop
    with torch.inference_mode():
        for deg, gt, _ in tqdm(val_loader, desc="Calculating Metrics"):
            deg, gt = deg.to(device), gt.to(device)
            
            pred = model(deg).clamp(0.0, 1.0)
            gt = gt.clamp(0.0, 1.0)
            
            mse = (pred - gt).square().mean()
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
            
            # Use functional ms_ssim which automatically handles multi-channel tensors
            ssim_val = ms_ssim(pred, gt, data_range=1.0, size_average=True)
            
            # LPIPS expects inputs in [-1, 1] range and 3 channels
            pred_lpips = (pred * 2.0 - 1.0).repeat(1, 3, 1, 1)
            gt_lpips = (gt * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lpips_val = lpips_fn(pred_lpips, gt_lpips).mean()
            
            total_psnr += psnr.item()
            total_ssim += ssim_val.item()
            total_lpips += lpips_val.item()
            count += 1
            
    print("\n" + "="*40)
    print("      FINAL RESTORATION METRICS      ")
    print("="*40)
    print(f"Images Evaluated : {count}")
    print(f"Mean pSNR        : {total_psnr / count:.3f} dB  (Higher is better)")
    print(f"Mean MS-SSIM     : {total_ssim / count:.4f}     (Closer to 1.0 is better)")
    print(f"Mean LPIPS       : {total_lpips / count:.4f}     (Closer to 0.0 is better)")
    print("="*40)

if __name__ == "__main__":
    main()
