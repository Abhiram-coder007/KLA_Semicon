import argparse
import csv
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torchvision.transforms.functional as TF

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import lpips

from dataset import KLARestorationDataset, PatchAugmentedDataset
from model import NAFNetDWT


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.eps**2).mean()


def gaussian(window_size, sigma):
    gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2 * sigma**2))) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window(window_size, channel, 1.5))
        
        raw_weights = torch.tensor([0.0448, 0.2856, 0.3001])
        self.register_buffer('weights', raw_weights / raw_weights.sum())

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq)
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return ssim_map.mean(), cs_map.mean()

    def forward(self, pred, gt):
        orig_dtype = pred.dtype
        pred = pred.float()
        gt = gt.float()
        levels = self.weights.size(0)
        mssim, mcs = [], []
        window = self.window.to(pred.device).float()
        channel = pred.size(1)

        if channel != self.channel:
            self.channel = channel
            window = create_window(self.window_size, channel, 1.5).to(pred.device)
            self.register_buffer('window', window)

        for i in range(levels):
            ssim_val, cs_val = self._ssim(pred, gt, window, self.window_size, channel)
            mssim.append(ssim_val)
            mcs.append(cs_val)
            if i < levels - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                gt = F.avg_pool2d(gt, kernel_size=2, stride=2)
        
        mssim = torch.relu(torch.stack(mssim)) + 1e-8
        mcs = torch.relu(torch.stack(mcs)) + 1e-8
        output = torch.prod(mcs[:-1] ** self.weights[:-1]) * (mssim[-1] ** self.weights[-1])
        return (1.0 - output).to(orig_dtype)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window(window_size, channel, sigma))

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        window = self.window.to(pred.device).float()
        if channel != self.channel:
            self.channel = channel
            window = create_window(self.window_size, channel, 1.5).to(pred.device)
            self.register_buffer('window', window)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(gt, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.relu(F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(gt * gt, window, padding=self.window_size // 2, groups=channel) - mu2_sq)
        sigma12 = F.conv2d(pred * gt, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class MasterRestorationLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.2, ssim_full_weight: float = 1.0, fft_weight: float = 0.05, lpips_weight: float = 0.02, eps: float = 1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ms_ssim = MSSSIMLoss(channel=1)
        self.ssim_full = SSIMLoss(channel=1)
        self.ssim_weight = ssim_weight
        self.ssim_full_weight = ssim_full_weight
        self.fft_weight = fft_weight
        self.lpips_weight = lpips_weight
        
        if self.lpips_weight > 0.0:
            self.lpips_model = lpips.LPIPS(net="vgg").eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        else:
            self.lpips_model = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        charb_loss = self.charbonnier(prediction, target)
        
        # FFT Frequency Loss
        fft_pred = torch.fft.fft2(prediction, norm='ortho')
        fft_target = torch.fft.fft2(target, norm='ortho')
        fft_loss = torch.mean(torch.abs(fft_pred - fft_target))
        
        # MS-SSIM Loss 
        ssim_loss = torch.tensor(0.0, device=prediction.device)
        if self.ssim_weight > 0.0:
            ssim_loss = self.ms_ssim(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))

        # Full-resolution SSIM Loss 
        ssim_full_loss = torch.tensor(0.0, device=prediction.device)
        if self.ssim_full_weight > 0.0:
            ssim_full_loss = self.ssim_full(prediction.clamp(0.0, 1.0), target.clamp(0.0, 1.0))

        # LPIPS Loss
        lpips_loss = torch.tensor(0.0, device=prediction.device)
        if self.lpips_weight > 0.0 and self.lpips_model is not None:
            pred_scaled = (prediction.clamp(0.0, 1.0) * 2.0 - 1.0).repeat(1, 3, 1, 1)
            target_scaled = (target.clamp(0.0, 1.0) * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lpips_loss = self.lpips_model(pred_scaled, target_scaled).mean()

        total_loss = (
            charb_loss
            + (self.fft_weight * fft_loss)
            + (self.ssim_weight * ssim_loss)
            + (self.ssim_full_weight * ssim_full_loss)
            + (self.lpips_weight * lpips_loss)
        )

        return {
            "total": total_loss,
            "charbonnier": charb_loss,
            "fft": fft_loss,
            "ssim": ssim_loss,
            "ssim_full": ssim_full_loss,
            "lpips": lpips_loss
        }


def apply_gpu_augmentations(deg_batch: torch.Tensor) -> torch.Tensor:
    B = deg_batch.shape[0]
    device = deg_batch.device
    rand_vals = torch.rand(B, device=device)
    
    mask = (rand_vals < 0.2).view(B, 1, 1, 1)
    sigma_add = torch.empty(B, 1, 1, 1, device=device).uniform_(0.02, 0.08)
    thermal = torch.randn_like(deg_batch) * sigma_add
    deg_batch = torch.where(mask, deg_batch + thermal, deg_batch)
    
    mask = ((rand_vals >= 0.2) & (rand_vals < 0.5)).view(B, 1, 1, 1)
    sigma_spk = torch.empty(B, 1, 1, 1, device=device).uniform_(0.05, 0.10)
    speckle = torch.randn_like(deg_batch) * sigma_spk
    deg_batch = torch.where(mask, deg_batch * (1.0 + speckle), deg_batch)

    mask = ((rand_vals >= 0.5) & (rand_vals < 0.7)).view(B, 1, 1, 1)
    c = torch.empty(B, 1, 1, 1, device=device).uniform_(0.9, 1.1)
    b = torch.empty(B, 1, 1, 1, device=device).uniform_(-0.05, 0.05)
    mean_val = deg_batch.mean(dim=[-2, -1], keepdim=True)
    contrast_batch = (deg_batch - mean_val) * c + mean_val + b
    deg_batch = torch.where(mask, contrast_batch, deg_batch)

    mask = ((rand_vals >= 0.7) & (rand_vals < 0.85)).view(B, 1, 1, 1)
    if mask.any():
        blur_sigma = torch.empty(1).uniform_(0.5, 1.2).item()
        blurred = TF.gaussian_blur(deg_batch, kernel_size=[5, 5], sigma=[blur_sigma, blur_sigma])
        deg_batch = torch.where(mask, blurred, deg_batch)

    mask = (rand_vals >= 0.85).view(B, 1, 1, 1)
    sp_mask = torch.rand_like(deg_batch) < 0.005
    salt = torch.rand_like(deg_batch) > 0.5
    sp_batch = torch.where(sp_mask, torch.where(salt, 1.0, 0.0), deg_batch)
    deg_batch = torch.where(mask, sp_batch, deg_batch)

    return deg_batch


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    parser = argparse.ArgumentParser(description="Train NAFNetDWT for KLA image restoration.")
    
    parser.add_argument("--train_degraded", type=Path, default=data_dir / "train_degraded")
    parser.add_argument("--train_gt", type=Path, default=data_dir / "train_gt")
    parser.add_argument("--val_degraded", type=Path, default=data_dir / "val_degraded")
    parser.add_argument("--val_gt", type=Path, default=data_dir / "val_gt")
    parser.add_argument("--output_dir", type=Path, default=script_dir)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_epochs", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_dim", type=int, default=32)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--ssim_full_weight", type=float, default=1.0)
    parser.add_argument("--fft_weight", type=float, default=0.05)
    parser.add_argument("--lpips_weight", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    
    return parser.parse_args()


def select_device(requested_device: str) -> torch.device:
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested_device == "auto":
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return torch.device(name)
    elif available[requested_device]:
        return torch.device(requested_device)
    raise RuntimeError(f"Requested device '{requested_device}' is not available.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def validate_dataset(dataset: KLARestorationDataset, name: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty.")
    degraded, target, filename = dataset[0]
    if degraded.ndim != 3 or target.ndim != 3:
        raise ValueError(f"{name} sample {filename} must have [C, H, W] tensors.")
    if degraded.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(f"{name} must contain single-channel grayscale images.")
    if tuple(target.shape[-2:]) != tuple(size * 2 for size in degraded.shape[-2:]):
        raise ValueError(
            f"{name} sample {filename} must be 2x super-resolution; got "
            f"input {tuple(degraded.shape)} and target {tuple(target.shape)}."
        )


def move_batch(degraded: torch.Tensor, target: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return degraded.to(device, non_blocking=non_blocking), target.to(device, non_blocking=non_blocking)


def calculate_batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def evaluate_validation(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[dict, float]:
    model.eval()
    metrics = {"total": 0.0, "charbonnier": 0.0, "fft": 0.0, "ssim": 0.0, "ssim_full": 0.0, "lpips": 0.0}
    psnr_total = 0.0
    image_count = 0

    with torch.inference_mode():
        for degraded, target, _ in dataloader:
            degraded, target = move_batch(degraded, target, device)
            prediction = model(degraded)
            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"Model output shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}."
                )

            batch_size = degraded.shape[0]
            losses = criterion(prediction, target)
            for k in metrics:
                metrics[k] += losses[k].item() * batch_size
                
            psnr_total += calculate_batch_psnr(prediction, target).sum().item()
            image_count += batch_size

    avg_metrics = {k: v / image_count for k, v in metrics.items()}
    avg_psnr = psnr_total / image_count
    return avg_metrics, avg_psnr


def save_checkpoint(path: Path, model: nn.Module, ema_model: nn.Module, optimizer: optim.Optimizer, scheduler: CosineAnnealingLR, epoch: int, best_val_psnr: float, model_kwargs: dict, args: argparse.Namespace) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_psnr": best_val_psnr,
        "model_kwargs": model_kwargs,
        "seed": args.seed,
    }
    torch.save(checkpoint, path)


LOG_FIELDS = [
    "epoch", "train_total", "train_charb", "train_fft", "train_ssim", "train_ssim_full", "train_lpips",
    "val_total", "val_charb", "val_fft", "val_ssim", "val_ssim_full", "val_lpips", "val_psnr", "learning_rate",
]


def append_log(log_path: Path, row: dict[str, float | int]) -> None:
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers cannot be negative.")
    if args.patience < 0 or args.grad_clip_norm < 0:
        raise ValueError("patience and grad_clip_norm cannot be negative.")

    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "master_training_log.csv"
    if log_path.exists():
        backup_path = log_path.with_name(f"training_log_{int(time.time())}.csv")
        log_path.rename(backup_path)
        print(f"Existing log moved to {backup_path}")

    train_dataset = PatchAugmentedDataset(args.train_degraded, args.train_gt, patch_size=args.patch_size)
    val_dataset = KLARestorationDataset(args.val_degraded, args.val_gt)
    validate_dataset(val_dataset, "Validation")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = build_loader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", generator=generator
    )
    val_loader = build_loader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda"
    )

    model_kwargs = {
        "in_channels": 1, "out_channels": 1, "base_dim": args.base_dim,
        "enc_blks": (1, 1, 1, 2), "middle_blk": 2, "dec_blks": (1, 1, 1, 1), "upscale": 2,
    }
    model = NAFNetDWT(**model_kwargs).to(device)

    # Strictly load base weights BEFORE initializing the EMA tracker
    if args.resume and args.resume.exists():
        print(f"Loading pre-trained weights from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Weights loaded successfully!")

    # Initialize EMA Tracker seeded with champion weights
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)
    if args.resume and args.resume.exists() and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

    criterion = MasterRestorationLoss(
        ssim_weight=args.ssim_weight, ssim_full_weight=args.ssim_full_weight,
        fft_weight=args.fft_weight, lpips_weight=args.lpips_weight
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.lr_epochs or args.epochs, eta_min=args.lr * 0.01
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = ckpt_dir / "kla_model.pth"
    last_checkpoint = ckpt_dir / "last_model.pth"

    epochs_without_improvement = 0

    print(f"Training on: {device}")
    print(f"Training pairs: {len(train_dataset)} | Validation pairs: {len(val_dataset)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"EMA Tracking: ENABLED | MS-SSIM: ENABLED | FFT Loss: ENABLED")
    print(f"Best checkpoint: {best_checkpoint}")

    initial_metrics, best_val_psnr = evaluate_validation(ema_model, val_loader, criterion, device)
    save_checkpoint(
        best_checkpoint, model, ema_model, optimizer, scheduler,
        epoch=0, best_val_psnr=best_val_psnr, model_kwargs=model_kwargs, args=args
    )
    print(
        f"Initial bilinear baseline: val loss={initial_metrics['total']:.6f} | "
        f"val PSNR={best_val_psnr:.3f} dB"
    )
    
    append_log(
        log_path,
        {
            "epoch": 0, "train_total": float("nan"), "val_total": initial_metrics['total'],
            "val_charb": initial_metrics['charbonnier'], "val_fft": initial_metrics['fft'],
            "val_ssim": initial_metrics['ssim'], "val_ssim_full": initial_metrics['ssim_full'],
            "val_lpips": initial_metrics['lpips'],
            "val_psnr": best_val_psnr, "learning_rate": args.lr
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_losses = {"total": 0.0, "charbonnier": 0.0, "fft": 0.0, "ssim": 0.0, "ssim_full": 0.0, "lpips": 0.0}
        image_count = 0
        
        progress_bar = (
            tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}")
            if tqdm is not None
            else train_loader
        )

        for degraded, target, _ in progress_bar:
            degraded, target = move_batch(degraded, target, device)
            degraded = apply_gpu_augmentations(degraded)
            
            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                prediction = model(degraded)
                if prediction.shape != target.shape:
                    raise RuntimeError(
                        f"Model output shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}."
                    )
                losses = criterion(prediction, target)
                loss = losses["total"]

            if amp_enabled:
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

            # Update EMA weights
            ema_model.update_parameters(model)

            batch_size = degraded.shape[0]
            image_count += batch_size
            for k in running_losses:
                running_losses[k] += losses[k].item() * batch_size

            if tqdm is not None:
                progress_bar.set_postfix(
                    tot=f"{loss.item():.4f}", 
                    charb=f"{losses['charbonnier'].item():.4f}",
                    fft=f"{losses['fft'].item():.4f}",
                    ssim=f"{losses['ssim'].item():.4f}",
                    ssimF=f"{losses['ssim_full'].item():.4f}",
                    lpips=f"{losses['lpips'].item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

        train_metrics = {k: v / image_count for k, v in running_losses.items()}
        val_metrics, val_psnr = evaluate_validation(ema_model, val_loader, criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"\n--- Epoch {epoch:03d} Summary ---")
        print(f"  [Train] Total: {train_metrics['total']:.4f} | Charb: {train_metrics['charbonnier']:.4f} | FFT: {train_metrics['fft']:.4f} | MS-SSIM: {train_metrics['ssim']:.4f} | SSIM(full): {train_metrics['ssim_full']:.4f} | LPIPS: {train_metrics['lpips']:.4f}")
        print(f"  [Val]   Total: {val_metrics['total']:.4f} | Charb: {val_metrics['charbonnier']:.4f} | FFT: {val_metrics['fft']:.4f} | MS-SSIM: {val_metrics['ssim']:.4f} | SSIM(full): {val_metrics['ssim_full']:.4f} | LPIPS: {val_metrics['lpips']:.4f} | PSNR: {val_psnr:.3f} dB\n")

        improved = val_psnr > best_val_psnr
        if improved:
            best_val_psnr = val_psnr
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint, model, ema_model, optimizer, scheduler,
                epoch, best_val_psnr, model_kwargs, args
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint, model, ema_model, optimizer, scheduler,
            epoch, best_val_psnr, model_kwargs, args
        )
        
        append_log(
            log_path,
            {
                "epoch": epoch, "train_total": train_metrics["total"],
                "train_charb": train_metrics["charbonnier"], "train_fft": train_metrics["fft"],
                "train_ssim": train_metrics["ssim"], "train_ssim_full": train_metrics["ssim_full"],
                "train_lpips": train_metrics["lpips"],
                "val_total": val_metrics["total"], "val_charb": val_metrics["charbonnier"],
                "val_fft": val_metrics["fft"], "val_ssim": val_metrics["ssim"],
                "val_ssim_full": val_metrics["ssim_full"],
                "val_lpips": val_metrics["lpips"], "val_psnr": val_psnr,
                "learning_rate": learning_rate
            },
        )

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without PSNR improvement.")
            break

    print(f"Finished. Best validation PSNR: {best_val_psnr:.3f} dB")
    print(f"Use this checkpoint for inference: {best_checkpoint}")


if __name__ == "__main__":
    train()
