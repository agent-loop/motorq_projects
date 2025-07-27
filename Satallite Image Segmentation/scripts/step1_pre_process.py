import os
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import cv2

try:
    import rasterio
    from rasterio.plot import reshape_as_image
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: Rasterio not available. TIF processing will use alternative method.")

def preprocess_images(input_dir='data/raw_training_data/raw_data', output_dir='data/raw_training_data/images', 
                    target_size=(1024, 1024), output_format='png'):
    """
    Preprocess various image formats:
    1. Handles TIF (including multi-band) and PNG/JPG
    2. Resizes to target size
    3. Converts all to a standard format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} image files")
    
    for img_path in tqdm(image_files):
        try:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{filename}.{output_format}")
            if img_path.lower().endswith(('.tif', '.tiff')):
                if RASTERIO_AVAILABLE:
                    with rasterio.open(img_path) as src:
                        if src.count >= 3:
                            img_data = src.read([1, 2, 3])
                            img_data = reshape_as_image(img_data)
                        else:
                            img_data = src.read(1)
                            img_data = np.stack([img_data, img_data, img_data], axis=2)
                        if img_data.dtype != np.uint8:
                            img_data = (img_data / img_data.max() * 255).astype(np.uint8)
                else:
                    img_data = np.array(Image.open(img_path).convert('RGB'))
            else:
                img_data = cv2.imread(img_path)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            
            img_data = cv2.resize(img_data, target_size, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(output_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    preprocess_images(target_size=(1024, 1024), output_format='png')
    print("Preprocessing complete!")