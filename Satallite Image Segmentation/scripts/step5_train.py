# save as train.py
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
import cv2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define expected classes
CLASS_NAMES = ['Background', 'Forest', 'Road', 'Building', 'Water', 'Land','__ignore__']
NUM_CLASSES = len(CLASS_NAMES)

def is_valid_mask(mask, max_class_value=6):
    """
    Check if mask contains only valid class values (0-max_class_value)
    Returns: (is_valid, unique_values)
    """
    unique_values = np.unique(mask)
    return np.max(unique_values) <= max_class_value, unique_values

# Custom loss function that ensures target is Long type
class TypeSafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(TypeSafeCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index,
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
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.max_class_value = max_class_value
        self.filter_invalid = filter_invalid
        
        # Get all image files
        self.images = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        
        # Filter out images with invalid masks
        if self.filter_invalid:
            self._filter_invalid_masks()
    
    def _filter_invalid_masks(self):
        """Filter out images with masks containing invalid class values"""
        valid_images = []
        invalid_count = 0
        
        print(f"Checking {len(self.images)} images for valid masks...")
        for img_name in tqdm(self.images, desc="Validating masks"):
            mask_name = img_name  # Assuming mask has same filename
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                # Skip if mask doesn't exist
                print(f"Skipping {img_name} - mask file not found")
                invalid_count += 1
                continue
            
            try:
                # Load mask and check if valid
                mask = np.array(Image.open(mask_path))
                is_valid, unique_values = is_valid_mask(mask, self.max_class_value)
                
                if is_valid:
                    valid_images.append(img_name)
                else:
                    invalid_count += 1
                    print(f"Skipping {img_name} - contains invalid classes: {[v for v in unique_values if v > self.max_class_value]}")
            except Exception as e:
                print(f"Error processing {mask_path}: {str(e)}")
                invalid_count += 1
        
        print(f"Filtered out {invalid_count} images with invalid masks. {len(valid_images)} remain.")
        self.images = valid_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Mask has same filename
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Read image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Double-check validity (in case transformation introduced invalid values)
        if isinstance(mask, np.ndarray):
            is_valid, _ = is_valid_mask(mask, self.max_class_value)
            if not is_valid and self.filter_invalid:
                # If invalid after transform, clamp values to valid range
                mask = np.clip(mask, 0, self.max_class_value)
                
            # FIX: Explicitly convert mask to torch.LongTensor
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor) and mask.dtype != torch.long:
            # If mask is already a tensor but not Long type, convert it
            mask = mask.long()
        
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
def calculate_class_weights(loader, num_classes=6, device='cuda'):
    print("Calculating class weights...")
    all_labels = []
    for _, target in tqdm(loader, desc="Analyzing class distribution"):
        # Ensure target is on CPU and convert to numpy array
        target_np = target.cpu().numpy().flatten()
        all_labels.append(target_np)
    
    all_labels = np.concatenate(all_labels)
    unique_values = np.unique(all_labels)
    print(f"Unique values in masks: {unique_values}")
    
    # Calculate class frequencies
    class_counts = np.bincount(all_labels, minlength=num_classes)
    print(f"Class counts: {class_counts}")
    
    # Ensure we have counts for all classes
    if len(class_counts) < num_classes:
        print(f"Warning: Only found {len(class_counts)} classes, expected {num_classes}")
        # Pad with small values to avoid division by zero
        class_counts = np.pad(class_counts, (0, num_classes - len(class_counts)), 
                             mode='constant', constant_values=1)
    
    # Handle classes with zero occurrences
    zero_counts = np.where(class_counts == 0)[0]
    if len(zero_counts) > 0:
        print(f"Warning: Classes {zero_counts} have zero occurrences")
        class_counts[zero_counts] = 1  # Set to 1 to avoid division by zero
    
    total_pixels = np.sum(class_counts)
    
    # Inverse frequency weighting
    class_weights = total_pixels / (class_counts * num_classes)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class weights shape: {class_weights.shape}")
    print(f"Class weights: {class_weights}")
    return class_weights

# IoU (Intersection over Union) metric
def iou_score(pred, target, n_classes=6):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate IoU for each class
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # No instances of this class
        else:
            ious.append((intersection / union).item())
    
    return ious

# Visualize predictions during training
def visualize_predictions(model, dataloader, device, epoch, output_dir, num_samples=4):
    """Generate and save visualization of predictions during training"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colormap for visualization
    colormap = np.array([
        [0, 0, 0],       # Background - Black
        [0, 128, 0],     # Forest - Green
        [128, 128, 128], # Road - Gray
        [255, 0, 0],     # Building - Red
        [0, 0, 255],     # Water - Blue
        [210, 180, 140]  # Land - Tan
    ], dtype=np.uint8)
    
    model.eval()
    images, masks, preds = [], [], []
    
    with torch.no_grad():
        for img, mask in dataloader:
            if len(images) >= num_samples:
                break
                
            img = img.to(device)
            output = model(img)
            pred = output.argmax(dim=1)
            
            # Collect samples
            for i in range(min(num_samples - len(images), len(img))):
                images.append(img[i].cpu())
                masks.append(mask[i].cpu())
                preds.append(pred[i].cpu())
                
                if len(images) >= num_samples:
                    break
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Convert tensors to numpy arrays
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()
        pred = preds[i].numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Create colored masks
        colored_mask = colormap[mask]
        colored_pred = colormap[pred]
        
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
    for i, (color, name) in enumerate(zip(colormap, CLASS_NAMES)):
        patch = plt.Rectangle((0, 0), 1, 1, fc=color/255)
        patches.append(patch)
    
    fig.legend(patches, CLASS_NAMES, loc='lower center', ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'predictions_epoch_{epoch}.png'))
    plt.close()

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
            
            # Move data to device and ensure target is long type
            data = data.to(device)
            target = target.to(device).long()
            
            # Ensure target is long type (double check)
            if target.dtype != torch.long:
                target = target.long()
            
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
            
            # Log progress every 50 batches
            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
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
                target = target.to(device).long()
                
                # Ensure target is long type (double check)
                if target.dtype != torch.long:
                    target = target.long()
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Calculate IoU
                pred = output.argmax(dim=1)
                batch_ious = iou_score(pred, target)
                all_ious.append(batch_ious)
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Calculate mean IoU per class
        mean_ious = np.nanmean(np.array(all_ious), axis=0)
        
        for i, class_iou in enumerate(mean_ious):
            writer.add_scalar(f'IoU/{CLASS_NAMES[i]}', class_iou, epoch)
        
        # Log mean IoU across all classes
        mean_iou = np.nanmean(mean_ious)
        writer.add_scalar('IoU/mean', mean_iou, epoch)
        
        # Visualize predictions every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            visualize_predictions(model, val_loader, device, epoch, vis_dir)
        
        # Log to console
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Mean IoU: {mean_iou:.4f}, Time: {epoch_time:.2f}s')
        
        # Log per-class IoU to console
        for i, class_name in enumerate(CLASS_NAMES):
            print(f'  {class_name} IoU: {mean_ious[i]:.4f}')
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
    
    # Data paths
    train_img_dir = 'data/processed/train/images'
    train_mask_dir = 'data/processed/train/masks'
    val_img_dir = 'data/processed/val/images'
    val_mask_dir = 'data/processed/val/masks'
    
    # Get transforms
    train_transform, val_transform = get_transforms(height=512, width=512)
    
    # Create datasets with filtering for invalid masks
    print("Creating training dataset...")
    train_dataset = SatelliteDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        filter_invalid=True  # This will filter out images with invalid class values
    )
    
    print("Creating validation dataset...")
    val_dataset = SatelliteDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        filter_invalid=True
    )
    
    # Adjust batch size based on your GPU memory
    batch_size = 6  # Use smaller batch size if you encounter memory issues
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check if datasets are valid
    if len(train_dataset) == 0:
        print("Error: No valid training images found. Please check your annotations.")
        exit(1)
    
    if len(val_dataset) == 0:
        print("Error: No valid validation images found. Please check your annotations.")
        exit(1)
    
    # Try to calculate class weights or use equal weights if it fails
    try:
        class_weights = calculate_class_weights(train_loader, num_classes=NUM_CLASSES, device=device)
        # Use type-safe CrossEntropyLoss with calculated weights
        criterion = TypeSafeCrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}")
        print("Using unweighted CrossEntropyLoss instead")
        criterion = TypeSafeCrossEntropyLoss()  # No weights
    
    # Model selection - Use UNet with efficient encoder for better performance
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # Lightweight but powerful encoder
        encoder_weights="imagenet",      # Pre-trained on ImageNet
        in_channels=3,                   # RGB images
        classes=NUM_CLASSES,             # Number of classes
        activation=None,                 # None for CrossEntropyLoss
    ).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
        num_epochs=50,  # Adjust based on your dataset size
        log_dir='logs',
        save_dir='models',
        vis_dir='visualizations',
        mixed_precision=True  # Use mixed precision for faster training
    )
    
    print("Training complete!")
    
    # Save final model
    torch.save({
        'epoch': 'final',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, os.path.join('models', 'final_model.pth'))
    
    print("Final model saved. Use 'best_model_iou.pth' for inference as it typically performs best.")
