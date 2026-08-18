from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KLARestorationDataset(Dataset):

    def __init__(self, deg_dir: str | Path, gt_dir: str | Path):
        self.deg_dir = Path(deg_dir)
        self.gt_dir = Path(gt_dir)
        if not self.deg_dir.is_dir():
            raise NotADirectoryError(f"Degraded-image directory not found: {self.deg_dir}")
        if not self.gt_dir.is_dir():
            raise NotADirectoryError(f"Ground-truth directory not found: {self.gt_dir}")

        degraded_files = {path.name for path in self.deg_dir.glob("*.npy")}
        ground_truth_files = {path.name for path in self.gt_dir.glob("*.npy")}
        missing_ground_truth = sorted(degraded_files - ground_truth_files)
        missing_degraded = sorted(ground_truth_files - degraded_files)
        if missing_ground_truth or missing_degraded:
            details = []
            if missing_ground_truth:
                details.append(f"missing GT for: {', '.join(missing_ground_truth[:5])}")
            if missing_degraded:
                details.append(f"missing degraded input for: {', '.join(missing_degraded[:5])}")
            raise ValueError("Paired filenames do not match; " + "; ".join(details))

        self.files = sorted(degraded_files)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_grayscale_tensor(path: Path) -> torch.Tensor:
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2:
            image = array
        elif array.ndim == 3 and array.shape[-1] == 1:
            image = array[..., 0]
        elif array.ndim == 3 and array.shape[0] == 1:
            image = array[0]
        else:
            raise ValueError(
                f"{path.name} must have shape (H, W), (H, W, 1), or (1, H, W); "
                f"got {array.shape}."
            )
        if not np.isfinite(image).all():
            raise ValueError(f"{path.name} contains NaN or infinite values.")

        image = np.ascontiguousarray(image, dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.files[index]
        degraded = self._load_grayscale_tensor(self.deg_dir / filename)
        ground_truth = self._load_grayscale_tensor(self.gt_dir / filename)
        expected_shape = tuple(size * 2 for size in degraded.shape[-2:])
        if tuple(ground_truth.shape[-2:]) != expected_shape:
            raise ValueError(
                f"Pair {filename} is not 2x super-resolution: input is "
                f"{tuple(degraded.shape)}, target is {tuple(ground_truth.shape)}."
            )
        return degraded, ground_truth, filename


class PatchAugmentedDataset(Dataset):
    
    def __init__(self, deg_dir: str | Path, gt_dir: str | Path, patch_size: int = 128, epoch_multiplier: int = 4):
        self.base = KLARestorationDataset(deg_dir, gt_dir)
        self.patch_size = patch_size
        self.epoch_multiplier = epoch_multiplier

    def __len__(self) -> int:
        return len(self.base) * self.epoch_multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        degraded, gt, filename = self.base[index % len(self.base)]

        h, w = degraded.shape[-2:]
        top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        
        deg_patch = degraded[:, top:top+self.patch_size, left:left+self.patch_size].clone()
        gt_patch = gt[:, top*2:(top+self.patch_size)*2, left*2:(left+self.patch_size)*2].clone()

        if torch.rand(1).item() < 0.5:
            deg_patch, gt_patch = deg_patch.flip(-1), gt_patch.flip(-1)
        if torch.rand(1).item() < 0.5:
            deg_patch, gt_patch = deg_patch.flip(-2), gt_patch.flip(-2)
            
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            deg_patch = torch.rot90(deg_patch, k, [-2, -1])
            gt_patch = torch.rot90(gt_patch, k, [-2, -1])

        return deg_patch, gt_patch, filename