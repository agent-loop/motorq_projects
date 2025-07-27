import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define expected classes
# IMPORTANT: If '__ignore__' (class 6) should truly be ignored during training and metrics,
# then NUM_CLASSES should be 6, and you should use ignore_index=6.
# If your masks actually contain values from 0-5, then max_class_value should be 5.
CLASS_NAMES = ['Background', 'Forest', 'Road', 'Building', 'Water', 'Land'] # Now 6 main classes
NUM_CLASSES = len(CLASS_NAMES) # This will be 6

# The ignore_index for CrossEntropyLoss, if applicable.
# This assumes that pixels with value 6 in your masks should be ignored in the loss calculation.
IGNORE_CLASS_INDEX = 6

def is_valid_mask(mask, max_class_value=5): # max_class_value should correspond to highest *valid* class ID (e.g., 5 for 0-5)
    """
    Check if mask contains only valid class values (0-max_class_value) OR the IGNORE_CLASS_INDEX.
    Returns: (is_valid, unique_values)
    """
    unique_values = np.unique(mask)
    # A mask is valid if all its unique values are within the defined range (0 to max_class_value)
    # OR if they are equal to the IGNORE_CLASS_INDEX.
    
    # Check if any value is outside the valid training range AND not the ignore index
    invalid_outside_training_range = np.any((unique_values < 0) | ((unique_values > max_class_value) & (unique_values != IGNORE_CLASS_INDEX)))

    return not invalid_outside_training_range, unique_values

# Custom loss function that ensures target is Long type and handles ignore_index
class TypeSafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        super(TypeSafeCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index if ignore_index is not None else -100, # Pass ignore_index to inner criterion
            reduction=reduction
        )
    
    def forward(self, input, target):
        # Ensure target is Long type
        if target.dtype != torch.long:
            target = target.long()
        return self.criterion(input, target)

# Dataset class with mask validation and proper type conversion
class SatelliteDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, max_class_value=5, filter_invalid=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.max_class_value = max_class_value # This refers to the highest *training* class value (e.g., 5)
        self.filter_invalid = filter_invalid
        
        # Get all image files (assuming .png for images based on previous scripts)
        # --- MODIFIED: Changed .jpg to .png for image files ---
        self.images = sorted(list(self.img_dir.glob("*.png"))) 
        
        # Filter out images with invalid masks
        if self.filter_invalid:
            self._filter_invalid_masks()
    
    def _filter_invalid_masks(self):
        """Filter out images with masks containing invalid class values
            (values not in 0-max_class_value and not IGNORE_CLASS_INDEX)
        """
        valid_images = []
        invalid_count = 0
        
        print(f"Checking {len(self.images)} images for valid masks...")
        for img_path in tqdm(self.images, desc="Validating masks"):
            # --- MODIFIED: Changed .tif to .png for mask files ---
            mask_path = self.mask_dir / (img_path.stem + '.png') 
            
            if not mask_path.exists():
                print(f"Skipping {img_path.name} - mask file {mask_path} not found")
                invalid_count += 1
                continue
            
            try:
                # Open mask in grayscale mode to ensure single channel values
                mask = np.array(Image.open(mask_path).convert("L")) # Convert to grayscale
                is_valid, unique_values = is_valid_mask(mask, self.max_class_value)
                
                if is_valid:
                    valid_images.append(img_path)
                else:
                    # Identify truly invalid values (not in 0-max_class_value and not IGNORE_CLASS_INDEX)
                    invalid_values_found = [
                        v for v in unique_values 
                        if (v < 0) or ((v > self.max_class_value) and (v != IGNORE_CLASS_INDEX))
                    ]
                    print(f"Skipping {img_path.name} - contains invalid classes: {invalid_values_found}")
                    invalid_count += 1
            except Exception as e:
                print(f"Error processing {mask_path}: {str(e)}")
                invalid_count += 1
        
        print(f"Filtered out {invalid_count} images with invalid masks. {len(valid_images)} remain.")
        self.images = valid_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # --- MODIFIED: Changed .tif to .png for mask files ---
        mask_path = self.mask_dir / (img_path.stem + '.png') 
        
        # Read image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        # Read mask in grayscale mode to ensure single channel values (class IDs)
        mask = np.array(Image.open(mask_path).convert("L")) 
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask values are within expected range (0 to max_class_value or IGNORE_CLASS_INDEX)
        # and convert mask to torch.LongTensor.
        # Clipping helps handle potential out-of-bounds values introduced by augmentations.
        mask = np.clip(mask, 0, max(self.max_class_value, IGNORE_CLASS_INDEX))
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# Data augmentation transforms
def get_transforms(height=512, width=512):
    train_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

# Calculate class weights to handle imbalanced classes
def calculate_class_weights(loader, num_classes=NUM_CLASSES, device='cuda', ignore_index=IGNORE_CLASS_INDEX):
    print("Calculating class weights...")
    all_labels = []
    for _, target in tqdm(loader, desc="Analyzing class distribution"):
        target_np = target.cpu().numpy().flatten()
        # Filter out ignore_index from class weight calculation if it's meant to be ignored
        if ignore_index is not None:
            target_np = target_np[target_np != ignore_index] # Remove ignore pixels before counting
        all_labels.append(target_np)
    
    all_labels = np.concatenate(all_labels)
    unique_values = np.unique(all_labels)
    print(f"Unique values in masks (for weight calculation, excluding ignore_index): {unique_values}")
    
    # Calculate class frequencies for actual training classes (0 to num_classes-1)
    class_counts = np.bincount(all_labels, minlength=num_classes)
    print(f"Class counts for classes 0 to {num_classes-1}: {class_counts}")
    
    # Ensure we have counts for all expected classes. Pad with 1 to avoid division by zero.
    if len(class_counts) < num_classes:
        print(f"Warning: Only found {len(class_counts)} classes in samples, expected {num_classes}. Padding with 1.")
        class_counts = np.pad(class_counts, (0, num_classes - len(class_counts)), 
                                 mode='constant', constant_values=1)
    
    # Handle classes with zero occurrences that might still exist after padding
    zero_counts_indices = np.where(class_counts == 0)[0]
    if len(zero_counts_indices) > 0:
        print(f"Warning: Classes {zero_counts_indices} have zero occurrences for weighting. Setting count to 1 to avoid NaN weights.")
        class_counts[zero_counts_indices] = 1 
    
    total_pixels = np.sum(class_counts)
    
    # Inverse frequency weighting
    class_weights = total_pixels / (class_counts * num_classes)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class weights shape: {class_weights.shape}")
    print(f"Class weights: {class_weights}")
    return class_weights

# IoU (Intersection over Union) metric
def iou_score(pred, target, n_classes=NUM_CLASSES, ignore_index=IGNORE_CLASS_INDEX):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Create a mask for pixels that are NOT the ignore_index in the target
    # This ensures ignored pixels don't influence IoU calculation for any class
    valid_pixels_mask = (target != ignore_index) if ignore_index is not None else torch.ones_like(target, dtype=torch.bool)

    # Filter pred and target to only include valid pixels for IoU calculation
    pred_filtered = pred[valid_pixels_mask]
    target_filtered = target[valid_pixels_mask]
    
    # Calculate IoU for each class (0 to n_classes-1)
    for cls in range(n_classes):
        pred_inds = pred_filtered == cls
        target_inds = target_filtered == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # No instances of this class in filtered prediction or target
        else:
            ious.append((intersection / union).item())
    
    return ious

# Visualize predictions during training
def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    """Generate and save visualization of predictions during training"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colormap for visualization - should match CLASS_NAMES (6 classes)
    # and include a distinct color for the IGNORE_CLASS_INDEX if it exists in masks.
    # colormap should have at least `max_class_id_seen + 1` entries for visualization.
    # If IGNORE_CLASS_INDEX is 6, colormap needs 7 entries.
    colormap = np.array([
        [0, 0, 0],       # 0: Background - Black
        [0, 128, 0],     # 1: Forest - Green
        [128, 128, 128], # 2: Road - Gray
        [255, 0, 0],     # 3: Building - Red
        [0, 0, 255],     # 4: Water - Blue
        [210, 180, 140], # 5: Land - Tan
        [255, 255, 255]  # 6: __ignore__ - White (distinct color for visualization, if present in masks)
    ], dtype=np.uint8)
    
    model.eval()
    images, masks, preds = [], [], []
    
    # Take samples from the start of the DataLoader for consistent visualization
    sample_iterator = iter(dataloader)

    with torch.no_grad():
        for _ in range(num_samples): # Try to get `num_samples` images
            try:
                img, mask = next(sample_iterator)
            except StopIteration:
                # No more samples in the dataloader
                break 

            img = img.to(device)
            output = model(img) 
            pred = output.argmax(dim=1)
            
            # Collect one sample (first from the batch)
            images.append(img[0].cpu())
            masks.append(mask[0].cpu())
            preds.append(pred[0].cpu())
            
            if len(images) >= num_samples:
                break
    
    if not images: # If no samples were collected
        print(f"No samples collected for visualization in epoch {epoch}. Skipping visualization.")
        model.train() # Set back to train mode
        return

    # Create figure
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5*len(images)))
    
    # Adjust axes if only one row is plotted
    if len(images) == 1:
        axes = axes[np.newaxis, :] # Make it 2D if only one sample is plotted
    
    # Prepare combined CLASS_NAMES for legend (including ignore if applicable)
    legend_class_names = CLASS_NAMES[:]
    if IGNORE_CLASS_INDEX is not None and IGNORE_CLASS_INDEX == len(CLASS_NAMES):
        legend_class_names.append('__ignore__')

    for i in range(len(images)): # Iterate through collected samples
        # Convert tensors to numpy arrays
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()
        pred = preds[i].numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Create colored masks - clip values to prevent index errors with colormap
        # Ensure mask and pred values don't exceed colormap size (e.g., if mask has 6, colormap needs 7 entries)
        colored_mask = colormap[np.clip(mask, 0, len(colormap) - 1)]
        colored_pred = colormap[np.clip(pred, 0, len(colormap) - 1)]
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(colored_mask)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colored_pred)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    # Add legend
    patches = []
    # Use legend_class_names and colormap consistent with its size
    for i, (color, name) in enumerate(zip(colormap, legend_class_names)):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color/255)
        patches.append(patch)
    
    fig.legend(patches, legend_class_names, loc='lower center', ncol=len(legend_class_names), bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'predictions_epoch_{epoch}.png'))
    plt.close()
    
    model.train() # Set model back to train mode

# Main training function
def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    num_epochs=50, 
    log_dir='trained_model_outputs/logs', 
    save_dir='trained_model_outputs/models', 
    vis_dir='trained_model_outputs/visualizations',
    mixed_precision=True
):
    """Train the model with all the bells and whistles needed for high accuracy"""
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # For mixed precision training (speeds up training with minimal accuracy loss)
    scaler = GradScaler() if mixed_precision else None
    
    best_val_loss = float('inf')
    best_mean_iou = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')):
            # Check tensor types in first batch for debugging
            if batch_idx == 0:
                print(f"Input tensor type: {data.dtype}, shape: {data.shape}")
                print(f"Target tensor type: {target.dtype}, shape: {target.shape}")
                print(f"Target unique values: {torch.unique(target).tolist()}")
            
            # Move data to device (target type conversion handled by dataset or TypeSafeCrossEntropyLoss)
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            if mixed_precision:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            # Log progress every 50 batches (optional, tqdm is also there)
            if (batch_idx + 1) % 50 == 0: # Adjusted to log after 50 batches
                 tqdm.write(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_ious = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Calculate IoU
                pred = output.argmax(dim=1)
                # Ensure pred and target are on CPU for numpy operations if not already
                batch_ious = iou_score(pred.cpu(), target.cpu(), n_classes=NUM_CLASSES, ignore_index=IGNORE_CLASS_INDEX) # Ensure cpu for numpy ops
                all_ious.append(batch_ious)
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Calculate mean IoU per class
        # Stack all_ious to a 2D array, then compute nanmean across batches for each class
        if all_ious: # Ensure all_ious is not empty
            mean_ious = np.nanmean(np.array(all_ious), axis=0) # This will have NaNs for ignored/missing classes
        else:
            mean_ious = np.full(NUM_CLASSES, np.nan) # If no batches, fill with NaN
        
        # Log per-class IoU to TensorBoard
        for i, class_iou in enumerate(mean_ious):
            if i < len(CLASS_NAMES): # Only log for the actual defined classes
                # Skip logging NaN to TensorBoard as it can cause issues or be misleading
                if not np.isnan(class_iou):
                    writer.add_scalar(f'IoU/{CLASS_NAMES[i]}', class_iou, epoch)
            
        # Calculate overall mean IoU, excluding NaN values (i.e., classes not present or ignored)
        mean_iou = np.nanmean(mean_ious)
        writer.add_scalar('IoU/mean', mean_iou, epoch)
        
        # Visualize predictions every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1: # Changed to (epoch + 1) % 5
            visualize_predictions(model, val_loader, device, epoch, vis_dir)
        
        # Log to console
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Mean IoU: {mean_iou:.4f}, Time: {epoch_time:.2f}s')
        
        # Log per-class IoU to console, handling NaN
        for i, class_name in enumerate(CLASS_NAMES): # Loop through 6 class names
            if i < len(mean_ious) and not np.isnan(mean_ious[i]):
                print(f'   {class_name} IoU: {mean_ious[i]:.4f}')
            else:
                print(f'   {class_name} IoU: N/A (No instances or ignored)')

        # Step the scheduler
        if scheduler: # Check if scheduler exists before stepping
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': avg_val_loss,
                'mean_iou': mean_iou,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best model by validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': avg_val_loss,
                'mean_iou': mean_iou,
            }, os.path.join(save_dir, 'best_model_loss.pth'))
            print(f'Model saved (best validation loss: {best_val_loss:.4f})')
        
        # Save best model by mean IoU
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': avg_val_loss,
                'mean_iou': mean_iou,
            }, os.path.join(save_dir, 'best_model_iou.pth'))
            print(f'Model saved (best mean IoU: {best_mean_iou:.4f})')
    
    writer.close()
    return model

if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths - assuming 'data/final_data' from your split_data.py
    # --- CONFIRMED: These paths are correct based on `split_data_into_sets_train_val_only.py` ---
    train_img_dir = 'data/final_data/train/images'
    train_mask_dir = 'data/final_data/train/masks'
    val_img_dir = 'data/final_data/val/images'
    val_mask_dir = 'data/final_data/val/masks'
    
    # Get transforms
    train_transform, val_transform = get_transforms(height=512, width=512)
    
    # Create datasets with filtering for invalid masks
    print("Creating training dataset...")
    train_dataset = SatelliteDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        max_class_value=5, # Highest actual class ID for training (0-5)
        filter_invalid=True
    )
    
    print("Creating validation dataset...")
    val_dataset = SatelliteDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        max_class_value=5, # Highest actual class ID for training (0-5)
        filter_invalid=True
    )
    
    # Adjust batch size based on your GPU memory
    batch_size = 6 # Use smaller batch size if you encounter memory issues
    
    # Create data loaders
    # num_workers: Adjust based on your CPU cores; 0 for debugging, more for faster loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check if datasets are valid (not empty after filtering)
    if len(train_dataset) == 0:
        print("Error: No valid training images found. Please check your data paths and mask integrity.")
        exit(1)
    
    if len(val_dataset) == 0:
        print("Error: No valid validation images found. Please check your data paths and mask integrity.")
        exit(1)
    
    # Calculate class weights
    try:
        class_weights = calculate_class_weights(train_loader, num_classes=NUM_CLASSES, device=device, ignore_index=IGNORE_CLASS_INDEX)
        # Use type-safe CrossEntropyLoss with calculated weights and ignore_index
        criterion = TypeSafeCrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_CLASS_INDEX)
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}")
        print("Using unweighted TypeSafeCrossEntropyLoss instead, but still ignoring class 6.")
        criterion = TypeSafeCrossEntropyLoss(ignore_index=IGNORE_CLASS_INDEX) # Still use ignore_index
    
    # Model selection - UNet++ with EfficientNet-B0 encoder
    # UNet++ is generally more robust for semantic segmentation.
    model = smp.UnetPlusPlus( 
        encoder_name="efficientnet-b0", # Lightweight but powerful encoder
        encoder_weights="imagenet",# Pre-trained on ImageNet
        in_channels=3, # RGB images
        classes=NUM_CLASSES, # Number of classes (6, as we handle ignore_index separately)
        activation=None,# None for CrossEntropyLoss
    ).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    # This scheduler often leads to better generalization.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Train the model
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50, # Adjust based on your dataset size and convergence
        log_dir='trained_model_outputs/logs', 
        save_dir='trained_model_outputs/models', 
        vis_dir='trained_model_outputs/visualizations', 
        mixed_precision=True # Use mixed precision for faster training
    )
    
    print("Training complete!")
    
    # Save final model
    torch.save({
        'epoch': 'final',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, os.path.join('trained_model_outputs/models', 'final_model.pth')) 
    
    print("Final model saved. For inference, 'best_model_iou.pth' or 'best_model_loss.pth' are typically preferred.")