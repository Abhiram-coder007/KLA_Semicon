import argparse
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from model import NAFNetDWT


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="High-performance batched inference for NAFNetDWT."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing degraded .npy images.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where restored .npy images will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=script_dir / "models" / "kla_model_final.pth",
        help="Model checkpoint path (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size per shape bucket.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for loading input data.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile optimization.",
    )
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


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load directly to target device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state dictionary or a checkpoint dictionary.")

    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
        print("✓ Successfully loaded EMA weights (Smoothed & Generalized).")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("✓ Successfully loaded standard model weights.")
    else:
        state_dict = checkpoint
        print("✓ Loaded raw checkpoint dictionary.")

    model_kwargs = checkpoint.get("model_kwargs", {})

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k == "n_averaged":  
            continue
        new_key = k.replace("module.", "")
        clean_state_dict[new_key] = v

    return clean_state_dict, model_kwargs


def get_spatial_shape(path: Path) -> Tuple[int, int]:
    """Extract canonical (H, W) spatial dimensions regardless of on-disk array layout."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim == 2:
        return arr.shape
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        return arr.shape[:2]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        return arr.shape[1:]
    else:
        return arr.shape[-2:]


class InferenceDataset(Dataset):
    def __init__(self, input_dir: Path):
        self.image_paths = sorted(input_dir.glob("*.npy"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in: {input_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        array = np.load(path, allow_pickle=False)
        
        if array.ndim == 2:
            image = array
        elif array.ndim == 3 and array.shape[-1] == 1:
            image = array[..., 0]
        elif array.ndim == 3 and array.shape[0] == 1:
            image = array[0]
        else:
            raise ValueError(f"{path.name} must be grayscale; got {array.shape}.")

        image = np.ascontiguousarray(image, dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), path.name


class ShapeBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: InferenceDataset, batch_size: int):
        self.batch_size = batch_size
        self.buckets = defaultdict(list)
        
        for idx, path in enumerate(dataset.image_paths):
            shape = get_spatial_shape(path)
            self.buckets[shape].append(idx)
            
    def __iter__(self) -> Iterator[List[int]]:
        for shape, indices in self.buckets.items():
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
                
    def __len__(self) -> int:
        total = 0
        for shape, indices in self.buckets.items():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def writer_worker(q: queue.Queue, output_dir: Path, error_list: List, error_lock: threading.Lock) -> None:
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        array, filename = item
        try:
            np.save(output_dir / filename, array)
        except Exception as e:
            with error_lock:
                error_list.append((filename, e))
        finally:
            q.task_done()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def warmup_model(
    model: torch.nn.Module,
    batch_sampler: ShapeBucketBatchSampler,
    device: torch.device,
    use_channels_last: bool,
    amp_dtype: torch.dtype,  # Corrected type annotation
    amp_enabled: bool,
) -> None:
    seen = set()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        for (h, w), indices in batch_sampler.buckets.items():
            for i in range(0, len(indices), batch_sampler.batch_size):
                b = len(indices[i:i + batch_sampler.batch_size])
                key = (b, h, w)
                if key in seen:
                    continue
                seen.add(key)
                
                dummy = torch.zeros(b, 1, h, w, device=device)
                if use_channels_last:
                    dummy = dummy.to(memory_format=torch.channels_last)
                _ = model(dummy)
    synchronize(device)


def restore_images(
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[float, int]:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    dataset = InferenceDataset(input_dir)
    batch_sampler = ShapeBucketBatchSampler(dataset, batch_size=batch_size)
    
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    use_channels_last = (device.type == "cuda")

    # Warm up compilation/CUDA graphs efficiently before starting the performance timer
    warmup_model(model, batch_sampler, device, use_channels_last, amp_dtype, amp_enabled)

    loader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler,
        num_workers=num_workers, 
        pin_memory=(device.type == "cuda")
    )

    # Thread-safe error tracking for background writer
    write_queue = queue.Queue(maxsize=16)
    error_list = []
    error_lock = threading.Lock()
    
    writer_thread = threading.Thread(
        target=writer_worker, 
        args=(write_queue, output_dir, error_list, error_lock), 
        daemon=True
    )
    writer_thread.start()

    synchronize(device)
    start_time = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        for tensors, filenames in loader:
            if use_channels_last:
                tensors = tensors.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                tensors = tensors.to(device, non_blocking=True)
            
            restored = model(tensors)

            upscale_factor = getattr(model, 'upscale', 1)
            expected_h = tensors.shape[-2] * upscale_factor
            expected_w = tensors.shape[-1] * upscale_factor
            
            if restored.shape[-2] != expected_h or restored.shape[-1] != expected_w:
                raise RuntimeError(
                    f"Model output shape {tuple(restored.shape[-2:])} does not match expected SR shape {(expected_h, expected_w)}."
                )

            outputs = restored.float().cpu().numpy()
            
            for i in range(len(filenames)):
                out_array = outputs[i, 0]

                out_array = np.nan_to_num(
                    out_array,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                out_array = np.clip(
                    out_array,
                    0.0,
                    1.0,
                ).astype(np.float32, copy=False)
                
                write_queue.put((out_array, filenames[i]))

    # Shutdown background writer thread and check for failures
    write_queue.put(None)
    writer_thread.join()

    if error_list:
        filename, err = error_list[0]
        raise RuntimeError(f"Failed to save output file {filename}: {err}")

    synchronize(device)
    return time.perf_counter() - start_time, len(dataset)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    
    # H100 / CUDA Hardware Optimizations
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load checkpoint directly to device
    state_dict, model_kwargs = load_checkpoint(args.checkpoint, device)
    
    # Instantiate model directly on target device
    model = NAFNetDWT(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    
    # Apply channels_last memory format for tensor core optimization
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Compile safely using mode="default"
    if not args.no_compile and device.type == "cuda" and hasattr(torch, "compile"):
        print("Optimizing model with torch.compile (mode='default')...")
        model = torch.compile(model, mode="default", dynamic=True)

    print(f"Starting shape-bucketed batched inference (batch_size={args.batch_size}) on {device}...")
    elapsed_seconds, count = restore_images(model, args.input_dir, args.output_dir, device, args.batch_size, args.num_workers)
    
    print(f"Restored {count} images on {device} in {elapsed_seconds:.3f} seconds.")
    print(f"Throughput: {(count / elapsed_seconds):.2f} images/sec.")
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
