import os
import shutil

# Define directories
raw_images_dir = 'data/raw_processed'
masks_dir = 'data/processed/masks'
output_dir = 'data/processed/images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get the list of filenames (without extensions) from masks directory
mask_filenames = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))}

# Copy matching images
copied_count = 0
print(copied_count)
for filename in os.listdir(raw_images_dir):
    file_path = os.path.join(raw_images_dir, filename)
    name, ext = os.path.splitext(filename)

    if name in mask_filenames:
        dest_path = os.path.join(output_dir, filename)
        shutil.copy2(file_path, dest_path)
        copied_count += 1

print(f"Copied {copied_count} files to '{output_dir}'")

