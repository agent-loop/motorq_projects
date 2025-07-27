import os
import json
import numpy as np
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm

CLASS_MAP = {
    '_background_': 0,
    'Background': 0,
    'Forest': 1,
    'Road': 2,
    'Building': 3,
    'Water': 4,
    'Land': 5
}

def convert_labelme_to_masks():
    json_dir = 'data/raw_training_data/json_files'
    # Assuming original images are stored here
    original_images_dir = 'data/raw_training_data/images' 
    mask_dir = 'data/raw_training_data/masks'
    
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(original_images_dir, exist_ok=True) # Ensure images dir exists if not already there

    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    print(f"Found {len(json_files)} annotation files")
    
    processed_count = 0
    for json_file in tqdm(json_files, desc="Converting LabelMe JSONs to Masks"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            img_filename = os.path.basename(data['imagePath'])
            original_img_path = os.path.join(original_images_dir, img_filename)

            # --- CRITICAL CHANGE HERE ---
            # Instead of relying on JSON's imageHeight/Width,
            # load the actual image to get its current dimensions.
            if not os.path.exists(original_img_path):
                print(f"Warning: Original image '{original_img_path}' not found for JSON '{json_file}'. Skipping.")
                continue
            
            try:
                with Image.open(original_img_path) as img:
                    actual_img_width, actual_img_height = img.size
            except Exception as img_err:
                print(f"Error opening image '{original_img_path}': {img_err}. Skipping JSON '{json_file}'.")
                continue

            # Compare with JSON's stored dimensions (for logging/warning)
            json_img_height = data.get('imageHeight')
            json_img_width = data.get('imageWidth')

            if json_img_height is not None and json_img_width is not None:
                if (actual_img_height != json_img_height) or (actual_img_width != json_img_width):
                    print(f"  Warning: Dimensions in JSON '{json_file}' ({json_img_width}x{json_img_height}) "
                          f"do not match actual image '{img_filename}' ({actual_img_width}x{actual_img_height}). "
                          f"Using actual image dimensions for mask creation.")
            else:
                print(f"  Info: 'imageHeight' or 'imageWidth' missing in JSON '{json_file}'. Using actual image dimensions.")

            # Use actual image dimensions for mask creation
            mask = np.zeros((actual_img_height, actual_img_width), dtype=np.uint8)
            
            # Create a PIL ImageDraw object for the mask
            # The 'L' mode is for grayscale (8-bit pixels, black and white)
            polygon_mask_img = Image.new('L', (actual_img_width, actual_img_height), 0)
            draw = ImageDraw.Draw(polygon_mask_img)

            for shape in data['shapes']:
                label = shape['label']
                if label not in CLASS_MAP:
                    print(f"Warning: Unknown label '{label}' in {json_file}. Skipping shape.")
                    continue
                
                pixel_value = CLASS_MAP[label]
                
                points = np.array(shape['points'], dtype=np.int32)
                
                # Flatten the points array to (x1, y1, x2, y2, ...) for PIL.ImageDraw.polygon
                points_flat = [coord for point in points for coord in point]
                
                # Draw the polygon directly onto the shared polygon_mask_img
                # This ensures that overlapping polygons are handled correctly
                # (later polygons overwrite earlier ones if fill value is different)
                draw.polygon(points_flat, fill=pixel_value)
            
            # Convert the PIL image to a NumPy array for final mask saving
            mask = np.array(polygon_mask_img)
            
            mask_filename = os.path.splitext(img_filename)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Save the mask. PIL's Image.fromarray handles 2D uint8 NumPy arrays well.
            Image.fromarray(mask).save(mask_path)
            processed_count += 1
            
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_file}. Skipping.")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\nSuccessfully created {processed_count} mask files out of {len(json_files)} JSON files.")

if __name__ == "__main__":
    convert_labelme_to_masks()