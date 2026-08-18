import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert output .npy arrays to .png images.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing the generated .npy files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the converted .png images.",
    )
    return parser.parse_args()

def convert_npy_to_png(npy_path: Path, out_path: Path) -> None:
    # Load the numpy array
    arr = np.load(npy_path)
    
    # Squeeze unnecessary dimensions (e.g., [1, H, W] or [H, W, 1] becomes [H, W])
    arr = np.squeeze(arr)
    
    # Clip the values to [0.0, 1.0] to prevent overflow artifacts
    arr = np.clip(arr, 0.0, 1.0)
    
    # Scale to [0, 255] and convert to 8-bit unsigned integer
    arr_8bit = (arr * 255.0).astype(np.uint8)
    
    # Create an image from the array
    img = Image.fromarray(arr_8bit)
    
    # Default to Grayscale if the array is 2D (semiconductor images are typically grayscale)
    if img.mode not in ['L', 'RGB']:
        img = img.convert('L')
         
    # Save as PNG
    img.save(out_path)

def main() -> None:
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(args.input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {args.input_dir.resolve()}")
        return
        
    print(f"Found {len(npy_files)} .npy files. Converting to .png...")
    
    for count, npy_path in enumerate(npy_files, 1):
        # Change file extension from .npy to .png
        out_path = args.output_dir / f"{npy_path.stem}.png"
        try:
            convert_npy_to_png(npy_path, out_path)
            # Print progress every 10 images
            if count % 10 == 0 or count == len(npy_files):
                print(f"Converted {count}/{len(npy_files)} images...")
        except Exception as e:
            print(f"Failed to convert {npy_path.name}: {e}")
            
    print(f"\n✓ Successfully saved all PNGs to: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
