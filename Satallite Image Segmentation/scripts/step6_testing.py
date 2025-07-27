# # save as predict_new.py
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import segmentation_models_pytorch as smp
# import glob
# import cv2
# import argparse

# # Define the colormap for visualization
# COLORMAP = np.array([
#     [0, 0, 0],       # Background - Black
#     [0, 128, 0],     # Forest - Green
#     [128, 128, 128], # Road - Gray
#     [255, 0, 0],     # Building - Red
#     [0, 0, 255],     # Water - Blue
#     [210, 180, 140]  # Land - Tan
# ], dtype=np.uint8)

# CLASS_NAMES = ['Background', 'Forest', 'Road', 'Building', 'Water', 'Land']

# def load_model(model_path, device, num_classes=6):
#     """Load a trained model from checkpoint"""
#     model = smp.UnetPlusPlus(
#         encoder_name="efficientnet-b0",
#         encoder_weights=None,  # We'll load the trained weights
#         in_channels=3,
#         classes=num_classes,
#         activation=None,
#     ).to(device)
    
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     return model

# def predict_image(model, image_path, output_dir, device, target_size=(512, 512)):
#     """Predict segmentation for a single image"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create transform
#     transform = A.Compose([
#         A.Resize(height=target_size[0], width=target_size[1]),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])
    
#     # Read image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     original_size = image.shape[:2]
    
#     # Apply transform
#     transformed = transform(image=image)
#     image_tensor = transformed['image'].unsqueeze(0).to(device)
    
#     # Get prediction
#     with torch.no_grad():
#         output = model(image_tensor)
#         pred = output.argmax(dim=1).cpu().numpy()[0]
    
#     # Resize prediction to original size
#     pred_original = cv2.resize(pred, (original_size[1], original_size[0]), 
#                              interpolation=cv2.INTER_NEAREST)
    
#     # Create colored mask
#     colored_pred = COLORMAP[pred_original]
    
#     # Calculate class percentages
#     class_pixels = {}
#     total_pixels = pred_original.size
#     for i, name in enumerate(CLASS_NAMES):
#         pixels = np.sum(pred_original == i)
#         percentage = (pixels / total_pixels) * 100
#         class_pixels[name] = percentage
    
#     # Create visualization
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(colored_pred)
#     plt.title("Segmentation")
#     plt.axis('off')
    
#     # Add legend
#     patches = []
#     legend_labels = []
#     for i, (color, name) in enumerate(zip(COLORMAP, CLASS_NAMES)):
#         patch = plt.Rectangle((0, 0), 1, 1, fc=color/255)
#         patches.append(patch)
#         legend_labels.append(f"{name}: {class_pixels[name]:.1f}%")
    
#     plt.figlegend(patches, legend_labels, loc='lower center', 
#                  ncol=3, bbox_to_anchor=(0.5, -0.05))
    
#     plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
#     # Save results
#     img_filename = os.path.basename(image_path)
#     output_path = os.path.join(output_dir, f"pred_{os.path.splitext(img_filename)[0]}.png")
#     plt.savefig(output_path)
    
#     # Also save just the colored mask
#     mask_path = os.path.join(output_dir, f"mask_{os.path.splitext(img_filename)[0]}.png")
#     cv2.imwrite(mask_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    
#     print(f"Prediction saved to {output_path}")
    
#     # Return class percentages
#     return class_pixels

# def predict_directory(model, input_dir, output_dir, device, target_size=(512, 512)):
#     """Process all images in a directory"""
#     # Find all images
#     image_files = []
#     for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
#         image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
#     print(f"Found {len(image_files)} images to process")
    
#     results = {}
#     for img_path in image_files:
#         img_filename = os.path.basename(img_path)
#         print(f"Processing {img_filename}...")
        
#         class_percentages = predict_image(model, img_path, output_dir, device, target_size)
#         results[img_filename] = class_percentages
    
#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Predict segmentation for satellite images')
#     parser.add_argument('--input', type=str, required=True, 
#                         help='Path to input image or directory')
#     parser.add_argument('--output', type=str, default='results/predictions', 
#                         help='Output directory')
#     parser.add_argument('--model', type=str, default='models/best_model_iou.pth', 
#                         help='Path to model file')
#     parser.add_argument('--size', type=int, default=512, 
#                         help='Size for processing (will be resized back to original)')
#     args = parser.parse_args()
    
#     # Check for GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Load model
#     model = load_model(args.model, device)
#     print(f"Model loaded from {args.model}")
    
#     # Process input
#     if os.path.isdir(args.input):
#         results = predict_directory(model, args.input, args.output, device, 
#                                    target_size=(args.size, args.size))
#     else:
#         predict_image(model, args.input, args.output, device, 
#                      target_size=(args.size, args.size))


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import cv2
import argparse
from pathlib import Path # Import Path for easier path manipulation

# Define the colormap for visualization
COLORMAP = np.array([
    [0, 0, 0],       # Background - Black
    [0, 128, 0],     # Forest - Green
    [128, 128, 128], # Road - Gray
    [255, 0, 0],     # Building - Red
    [0, 0, 255],     # Water - Blue
    [210, 180, 140]  # Land - Tan
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'Forest', 'Road', 'Building', 'Water', 'Land']
NUM_CLASSES_MODEL = len(CLASS_NAMES) # Model will output for these 6 classes

def load_model(model_path, device, num_classes=NUM_CLASSES_MODEL):
    """Load a trained model from checkpoint"""
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights=None,   # We'll load the trained weights
        in_channels=3,
        classes=num_classes,    # Should match the NUM_CLASSES used during training
        activation=None,        # None for CrossEntropyLoss
    ).to(device)
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle potential "module." prefix from DataParallel if trained on multiple GPUs
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model

def predict_image(model, image_path, output_dir, device, target_size=(512, 512)):
    """Predict segmentation for a single image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transform for inference
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Read image using PIL for consistency with training, then convert to OpenCV format for `cv2.resize` later
    # Using PIL for initial load ensures correct handling of PNGs as RGB
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil) # Convert PIL image to NumPy array for Albumentations
    
    original_size = image.shape[:2] # (height, width)
    
    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device) # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).cpu().numpy()[0] # Get predicted class IDs
    
    # Resize prediction back to original size using cv2.INTER_NEAREST for class IDs
    pred_original = cv2.resize(pred.astype(np.uint8), (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    # Ensure pred_original values don't exceed colormap size
    colored_pred = COLORMAP[np.clip(pred_original, 0, len(COLORMAP) - 1)]
    
    # Calculate class percentages
    class_pixels = {}
    total_pixels = pred_original.size
    for i, name in enumerate(CLASS_NAMES):
        pixels = np.sum(pred_original == i)
        percentage = (pixels / total_pixels) * 100
        class_pixels[name] = percentage
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(colored_pred)
    axes[1].set_title("Segmentation")
    axes[1].axis('off')
    
    # Add legend
    patches = []
    legend_labels = []
    for i, (color, name) in enumerate(zip(COLORMAP, CLASS_NAMES)):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color/255)
        patches.append(patch)
        legend_labels.append(f"{name}: {class_pixels[name]:.1f}%")
    
    # Using fig.legend for better placement flexibility
    fig.legend(handles=patches, labels=legend_labels, loc='lower center', 
               ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize='medium', frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for legend
    
    # Save visualization (image + prediction + legend)
    img_filename = os.path.basename(image_path)
    output_vis_path = os.path.join(output_dir, f"prediction_with_legend_{os.path.splitext(img_filename)[0]}.png")
    plt.savefig(output_vis_path, bbox_inches='tight') # bbox_inches='tight' helps include legend
    plt.close(fig) # Close the figure to free memory
    
    # Also save just the colored mask (segmentation only)
    mask_only_path = os.path.join(output_dir, f"mask_only_{os.path.splitext(img_filename)[0]}.png")
    # OpenCV expects BGR for writing PNGs, so convert from RGB
    cv2.imwrite(mask_only_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    
    print(f"Visualization saved to {output_vis_path}")
    print(f"Mask (segmentation only) saved to {mask_only_path}")
    
    # Print class percentages to console
    print(f"Class percentages for {img_filename}:")
    for name, percentage in class_pixels.items():
        print(f"  {name}: {percentage:.1f}%")
    
    return class_pixels

def predict_directory(model, input_dir, output_dir, device, target_size=(512, 512)):
    """Process all images in a directory"""
    # Find all images (support .png, .jpg, .jpeg, .tif)
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort files for consistent processing order
    image_files = sorted(image_files)

    if not image_files:
        print(f"No image files found in {input_dir}. Please check the directory and file extensions.")
        return {}
    
    print(f"Found {len(image_files)} images to process in {input_dir}")
    
    results = {}
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        print(f"\nProcessing {img_filename}...")
        
        class_percentages = predict_image(model, img_path, output_dir, device, target_size)
        results[img_filename] = class_percentages
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict segmentation for satellite images')
    parser.add_argument('--input', type=str, 
                        # Default to a placeholder for a single test image
                        default='data/final_data/test/images/sample_test_image.png', 
                        help='Path to input image file or directory containing images.')
    parser.add_argument('--output', type=str, 
                        # Default output directory for test results
                        default='data/final_data/test/output', 
                        help='Output directory where predictions will be saved.')
    parser.add_argument('--model', type=str, 
                        # Default model path, assuming best_model_iou.pth is saved
                        default='trained_model_outputs/models/best_model_iou.pth', 
                        help='Path to the trained model checkpoint file (e.g., best_model_iou.pth).')
    parser.add_argument('--size', type=int, default=512, 
                        help='Size (height and width) to resize images for model input.')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(args.model, device)
        print(f"Model loaded successfully from {args.model}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your model path is correct and the file exists.")
        print("Expected model path example: 'trained_model_outputs/models/best_model_iou.pth'")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit(1)
    
    # Process input: Check if it's a directory or a single file
    input_path = Path(args.input) # Use pathlib for robust path checks
    
    if input_path.is_dir():
        print(f"\nProcessing all images in directory: {input_path}")
        results = predict_directory(model, str(input_path), args.output, device, 
                                     target_size=(args.size, args.size))
        print("\nAll predictions complete for directory.")
        # Optionally, you can print aggregated results here
        # For example, average percentages across all images in the directory
    elif input_path.is_file():
        print(f"\nProcessing single image: {input_path}")
        predict_image(model, str(input_path), args.output, device, 
                      target_size=(args.size, args.size))
        print("\nPrediction complete for single image.")
    else:
        print(f"Error: Input path '{args.input}' is neither a valid file nor a directory.")
        exit(1)