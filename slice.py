import argparse
import random
import shutil
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a validation split by moving random files from train to val.")
    parser.add_argument(
        "--train_degraded", type=Path, default=Path("train_degraded"), help="Path to training degraded images."
    )
    parser.add_argument(
        "--train_gt", type=Path, default=Path("train_gt"), help="Path to training ground truth images."
    )
    parser.add_argument(
        "--val_degraded", type=Path, default=Path("val_degraded"), help="Path to output validation degraded images."
    )
    parser.add_argument(
        "--val_gt", type=Path, default=Path("val_gt"), help="Path to output validation ground truth images."
    )
    parser.add_argument(
        "--num_val", type=int, default=400, help="Number of image pairs to move to the validation set (default: 400)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Verify that the training directories exist
    if not args.train_degraded.exists():
        raise FileNotFoundError(f"Training directory not found: {args.train_degraded.resolve()}")
    if not args.train_gt.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {args.train_gt.resolve()}")

    # Create the destination validation directories
    args.val_degraded.mkdir(parents=True, exist_ok=True)
    args.val_gt.mkdir(parents=True, exist_ok=True)

    # Gather all .npy files from the degraded directory
    degraded_files = sorted(list(args.train_degraded.glob("*.npy")))
    
    if not degraded_files:
        print(f"No .npy files found in {args.train_degraded}.")
        return

    # Pair them with ground truth files and ensure they exist
    valid_pairs = []
    for deg_path in degraded_files:
        gt_path = args.train_gt / deg_path.name
        if gt_path.exists():
            valid_pairs.append((deg_path, gt_path))
        else:
            print(f"Warning: Missing matching ground truth for {deg_path.name}")

    total_pairs = len(valid_pairs)
    print(f"Found {total_pairs} valid matched pairs in the training directories.")

    if total_pairs == 0:
        print("No matching pairs found. Exiting.")
        return

    # Determine how many to move
    num_to_move = min(args.num_val, total_pairs)
    
    # Shuffle deterministically so the split is always identical for judges
    random.seed(args.seed)
    random.shuffle(valid_pairs)
    
    val_pairs = valid_pairs[:num_to_move]

    print(f"Moving {num_to_move} pairs to validation folders...")

    # Move the files safely
    for deg_path, gt_path in val_pairs:
        val_deg_dest = args.val_degraded / deg_path.name
        val_gt_dest = args.val_gt / gt_path.name
        
        shutil.move(str(deg_path), str(val_deg_dest))
        shutil.move(str(gt_path), str(val_gt_dest))

    print("\n✓ Validation split created successfully!")
    print(f"  - Train set remaining: {total_pairs - num_to_move} images")
    print(f"  - Validation set size: {num_to_move} images")

if __name__ == "__main__":
    main()
