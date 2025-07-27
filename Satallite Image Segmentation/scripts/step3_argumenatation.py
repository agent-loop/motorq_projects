import os
from pathlib import Path
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from tqdm import tqdm # Added for progress bar during augmentation

def _check_dims(img_data, mask_data, context_msg=""):
    """Helper to check if image and mask data have matching HxW dimensions."""
    img_h, img_w = img_data.shape[:2]
    mask_h, mask_w = mask_data.shape[:2] # Masks are expected to be 2D here

    if img_h != mask_h or img_w != mask_w:
        print(f"!!! DIMENSION MISMATCH {context_msg} !!!")
        print(f"    Image dimensions: ({img_h}, {img_w})")
        print(f"    Mask dimensions:  ({mask_h}, {mask_w})")
        return False
    return True


def augment_and_save_data_in_place():
    """
    Performs data augmentation on image-mask pairs and saves the augmented data
    back into the original 'images' and 'masks' directories, avoiding duplicates
    for the original image.

    The base directory is automatically determined as the parent directory
    of the directory where this script is located (assuming script is in 'scripts' folder).
    """

    base_path = Path(__file__).resolve().parent.parent

    images_dir = base_path / "data/raw_training_data/images"
    masks_dir = base_path / "data/raw_training_data/masks"

    if not images_dir.is_dir():
        print(f"Error: Images directory not found at {images_dir}. Exiting.")
        return
    if not masks_dir.is_dir():
        print(f"Error: Masks directory not found at {masks_dir}. Exiting.")
        return

    print(f"Base Directory (Resolved): {base_path}")
    print(f"Images Directory: {images_dir}")
    print(f"Masks Directory: {masks_dir}")

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))), # apply gaussian blur to 30% of images
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
            rotate=(-25, 25), # rotate by -25 to +25 degrees
            shear=(-8, 8), # shear by -8 to +8 degrees
            order=[0, 1], # use nearest neighbour (0) or bilinear (1) interpolation
            cval=(0, 255), # if we have to fill in new pixels, what value do they get
            mode=ia.ALL # use ia.ALL from the top-level imgaug import
        ),
        iaa.Multiply((0.8, 1.2), per_channel=0.2), # change brightness of images (80-120% of original value)
        iaa.LinearContrast((0.75, 1.5)), # improve or worsen the contrast
    ], random_order=True) # apply augmenters in random order

    image_files = sorted(list(images_dir.glob("*.png"))) # Still looking for .png images

    if not image_files:
        print(f"No PNG images found in {images_dir}. Exiting.")
        return

    processed_base_images_stems = set() # Use set to store stems of original images we've processed
    count = 0
    print(f"Found {len(image_files)} image files. Starting augmentation...")
    for img_path in tqdm(image_files, desc="Augmenting data"):
        image_stem = img_path.stem

        # This logic ensures we only augment *original* images, not already augmented ones
        if "_aug" in image_stem:
            # If it's an augmented file, extract the original base name
            base_stem = image_stem.split('_aug')[0]
            # If we've already processed the original for this base_stem, skip
            if base_stem in processed_base_images_stems:
                # print(f"Skipping already augmented instance or base already processed: {img_path.name}")
                continue
            # If it's an augmented file and its base hasn't been processed, it implies
            # this augmented file was there from a previous run without its original base
            # being processed in *this* run. We still want to skip it from being processed again.
            # print(f"Skipping already augmented image (from previous run): {img_path.name}")
            continue

        # Add the stem of the current original image to the set of processed base images
        # so we don't accidentally process it again if it appears later in the sorted list
        processed_base_images_stems.add(image_stem)


        mask_path = masks_dir / f"{image_stem}.png"

        if not mask_path.exists():
            print(f"Skipping {img_path.name}: No corresponding mask found at {mask_path}")
            continue

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not load image {img_path.name}. Skipping.")
            continue
        
        # Convert to 3 channels (BGR) if grayscale or RGBA
        if len(image.shape) == 2: # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Load mask (read as single channel grayscale for segmentation values)
        mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_data is None:
            print(f"Error: Could not load mask {mask_path.name}. Skipping.")
            continue
        
        # Ensure mask is uint8 (important for segmentation maps)
        if mask_data.dtype != np.uint8:
            mask_data = mask_data.astype(np.uint8)
        
        # Ensure mask is 2D (height, width) as SegmentationMapOnImage expects
        if mask_data.ndim == 3:
            # If it somehow loaded as 3D (e.g., BGR), convert to single channel
            if mask_data.shape[2] == 3:
                mask_data = cv2.cvtColor(mask_data, cv2.COLOR_BGR2GRAY)
            elif mask_data.shape[2] == 1:
                mask_data = mask_data.squeeze(axis=-1) # Remove the channel dimension if it's 1
            else:
                print(f"Warning: Mask {mask_path.name} has unexpected shape {mask_data.shape}. Attempting to use first channel.")
                mask_data = mask_data[:,:,0] # Take the first channel
        
        # --- Pre-augmentation Dimension Check ---
        print(f"  Checking original dims for {img_path.name}:")
        if not _check_dims(image, mask_data, f"ORIGINAL ({img_path.name})"):
            print(f"    WARNING: Original image and mask for {img_path.name} already have mismatched dimensions. This needs to be fixed at source data level.")
            count = count+1
            # You might want to skip this image or raise an error here if you want strict pre-checks.
            # For now, we'll continue to see if augmentation itself introduces new issues.
            pass # Keep processing to see if imgaug aligns them or adds new issues
        
        segmap = SegmentationMapOnImage(mask_data, shape=image.shape) # Pass original image shape for reference
        
        # Perform 5 augmentations per original image
        for i in range(5):
            # Apply augmentation for each iteration
            images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
            
            aug_image = images_aug_i
            aug_mask = segmaps_aug_i.get_arr()
            # --- Post-augmentation Dimension Check (BEFORE SAVING) ---
            print(f"  Checking augmented dims (aug{i}) for {img_path.name} BEFORE SAVING:")
            if not _check_dims(aug_image, aug_mask, f"AUGMENTED {i} ({img_path.name})"):
                print(f"    ERROR: Augmented image and mask for {img_path.name} (aug{i}) have mismatched dimensions. This indicates an issue with the augmentation pipeline itself.")
                # count = count + 1
                # You might want to break or exit here, as saving mismatched data is problematic.
                # For now, we'll continue to save, but be aware it's bad data.
            
            save_image_path = images_dir / f"{image_stem}_aug{i}.png"
            save_mask_path = masks_dir / f"{image_stem}_aug{i}.png"

            # Save augmented image
            cv2.imwrite(str(save_image_path), aug_image)

            # Save augmented mask as grayscale (PNG is good for this)
            cv2.imwrite(str(save_mask_path), aug_mask)

    print("\nData augmentation and saving complete!")
    print(count)

if __name__ == "__main__":
    augment_and_save_data_in_place()