import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm # For progress bar

def split_data_into_sets_train_val_only():
    """
    Splits image-mask pairs into train and validation sets randomly
    with an 85% train and 15% validation ratio.
    """
    base_path = Path(__file__).resolve().parent.parent

    source_images_dir = base_path / "data/raw_training_data/images"
    source_masks_dir = base_path / "data/raw_training_data/masks"

    # Destination base directory for final data
    final_data_dir = base_path / "data/final_data"

    # Define destination subdirectories (only train and val)
    splits = {
        "train": final_data_dir / "train",
        "val": final_data_dir / "val"
    }

    # Create all necessary destination directories
    for split_name, split_path in splits.items():
        (split_path / "images").mkdir(parents=True, exist_ok=True)
        (split_path / "masks").mkdir(parents=True, exist_ok=True)
        print(f"Created/Ensured directory: {split_path / 'images'}")
        print(f"Created/Ensured directory: {split_path / 'masks'}")

    print(f"\nSource Images Directory: {source_images_dir}")
    print(f"Source Masks Directory: {source_masks_dir}")
    print(f"Destination Base Directory: {final_data_dir}\n")

    # Collect all image files (assuming they are .png as per your last clarification)
    image_files = sorted(list(source_images_dir.glob("*.png")))

    if not image_files:
        print(f"No PNG images found in {source_images_dir}. Exiting.")
        return

    # Create a list of (image_path, mask_path) tuples
    data_pairs = []
    for img_path in image_files:
        # --- CRITICAL CHANGE HERE: Look for .png masks ---
        mask_path = source_masks_dir / f"{img_path.stem}.png" # Assumes mask has same stem but .png extension
        
        if mask_path.exists():
            data_pairs.append((img_path, mask_path))
        else:
            print(f"Warning: No corresponding mask found for {img_path.name} at {mask_path}. Skipping this pair.")

    if not data_pairs:
        print("No valid image-mask pairs found for splitting. Exiting.")
        return

    print(f"Found {len(data_pairs)} image-mask pairs for splitting.")

    # Shuffle the data pairs randomly
    random.seed(42) # Added a seed for reproducibility of the split
    random.shuffle(data_pairs)

    # Define the split ratios for train and val
    total_samples = len(data_pairs)
    train_ratio = 0.85 # 85%
    val_ratio = 0.15   # 15%

    # Calculate split sizes
    num_train = int(total_samples * train_ratio)
    num_val = total_samples - num_train # Assign remaining to val to ensure sum is exact

    print(f"Splitting data: Total={total_samples}")
    print(f"   Train samples: {num_train} ({train_ratio*100:.1f}%)")
    print(f"   Validation samples: {num_val} ({val_ratio*100:.1f}%)")
    print(f"   (Actual ratios may slightly vary due to rounding, sum is exact)")

    # Divide the shuffled data
    train_data = data_pairs[:num_train]
    val_data = data_pairs[num_train:]

    # Copy files to their respective directories
    print("\nCopying files to train set...")
    for img_path, mask_path in tqdm(train_data, desc="Copying Train Data"):
        shutil.copy2(img_path, splits["train"] / "images" / img_path.name)
        shutil.copy2(mask_path, splits["train"] / "masks" / mask_path.name)

    print("\nCopying files to validation set...")
    for img_path, mask_path in tqdm(val_data, desc="Copying Val Data"):
        shutil.copy2(img_path, splits["val"] / "images" / img_path.name)
        shutil.copy2(mask_path, splits["val"] / "masks" / mask_path.name)

    print("\nData splitting and copying complete!")

if __name__ == "__main__":
    split_data_into_sets_train_val_only()