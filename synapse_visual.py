#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
import matplotlib.pyplot as plt
from utils.utils import DiceLoss
from torch.utils.data import DataLoader
from dataset.dataset_Synapse import Synapsedataset, RandomGenerator
import argparse
from tqdm import tqdm
import os
import json
from torchvision import transforms
from utils.test_Synapse import inference
from model.MTUNet import MTUNet
import numpy as np
from medpy.metric import dc, hd95
import time
from datetime import datetime

# Configure logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"training_log_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Create parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, help="batch size", type=int)
parser.add_argument("--lr", default=0.0001, help="learning rate", type=float)
parser.add_argument("--max_epochs", default=1, type=int)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--save_path", default="./checkpoint/Synapse/mtunet")
parser.add_argument("--n_gpu", default=1, type=int)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./dataset/lists_Synapse/lists_Synapse")
parser.add_argument("--root_dir", default="../dataset/project_TransUNet/data/Synapse")
parser.add_argument("--volume_path", default="../dataset/project_TransUNet/data/Synapse/test")
parser.add_argument("--z_spacing", default=10, type=int)
parser.add_argument("--num_classes", default=9, type=int)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument("--patches_size", default=16, type=int)
parser.add_argument("--n-skip", default=1, type=int)
parser.add_argument("--vis_dir", default="./visualizations", help="Directory to save visualizations")
parser.add_argument("--vis_sample_count", default=5, type=int, help="Number of test samples to visualize")
args = parser.parse_args()

# Create directories for saving results
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.test_save_dir, exist_ok=True)
os.makedirs(args.vis_dir, exist_ok=True)
os.makedirs(os.path.join(args.vis_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(args.vis_dir, "test"), exist_ok=True)

# Create a JSON log file for metrics
metrics_log_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
metrics = {
    "train_loss": [],
    "val_dice": [],
    "val_hd95": [],
    "lr": [],
    "epoch": [],
    "class_dice": [],
    "best_model_path": ""
}

# Define color map for segmentation visualization (matching Synapse dataset classes)
COLORMAP = {
    0: [0, 0, 0],       # Background: Black
    1: [255, 0, 0],     # Spleen: Red
    2: [0, 255, 0],     # Right Kidney: Green
    3: [0, 0, 255],     # Left Kidney: Blue
    4: [255, 255, 0],   # Gallbladder: Yellow
    5: [255, 0, 255],   # Liver: Magenta
    6: [0, 255, 255],   # Stomach: Cyan
    7: [255, 128, 0],   # Aorta: Orange
    8: [128, 0, 255]    # Inferior Vena Cava: Purple
}

CLASS_NAMES = [
    "Background", "Spleen", "Right Kidney", "Left Kidney", 
    "Gallbladder", "Liver", "Stomach", "Aorta", "Inferior Vena Cava"
]

def save_metrics_to_json():
    """Save current metrics to JSON file for later plotting"""
    with open(metrics_log_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def create_colored_mask(mask):
    """Convert a segmentation mask to a colored image"""
    if len(mask.shape) == 3:  # If mask is batch, height, width
        mask = mask[0]  # Take first slice
    
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Apply color for each class
    for class_idx, color in COLORMAP.items():
        rgb_mask[mask == class_idx] = color
        
    return rgb_mask

def visualize_batch(images, masks, predictions, epoch, batch_idx, phase="train"):
    """Save visualizations of images, ground truth masks, and predictions"""
    save_dir = os.path.join(args.vis_dir, phase)
    os.makedirs(save_dir, exist_ok=True)
    
    # Process only the first image in the batch for visualization
    image = images[0].cpu().numpy().transpose(1, 2, 0)
    
    # Normalize image for visualization (assuming it's already 0-1 normalized)
    image = (image - image.min()) / (image.max() - image.min()) 
    
    gt_mask = masks[0].cpu().numpy()
    pred_mask = torch.argmax(predictions[0], dim=0).cpu().numpy()
    
    # Create colored visualizations
    colored_gt = create_colored_mask(gt_mask)
    colored_pred = create_colored_mask(pred_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(colored_gt)
    axes[1].set_title("Ground Truth Segmentation")
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(colored_pred)
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis('off')
    
    # Add a color legend
    handles = []
    labels = []
    
    # Add the class legend below the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Add legend at the bottom
    legend_elements = []
    for i, name in enumerate(CLASS_NAMES):
        if i in COLORMAP:  # Skip if not in colormap
            color = [c/255 for c in COLORMAP[i]]  # Convert to 0-1 range for matplotlib
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=name))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(CLASS_NAMES)), frameon=False)
    
    # Save the figure
    file_name = f"{phase}_epoch{epoch}_batch{batch_idx}.png"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    logging.info(f"Saved visualization to {os.path.join(save_dir, file_name)}")

def visualize_test_case(image, gt_mask, pred_mask, case_name):
    """Visualize test case with ground truth and prediction"""
    save_dir = os.path.join(args.vis_dir, "test")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create colored visualizations
    colored_gt = create_colored_mask(gt_mask)
    colored_pred = create_colored_mask(pred_mask)
    
    # Calculate difference mask (red where prediction differs from ground truth)
    diff_mask = np.zeros_like(colored_gt)
    diff_mask[gt_mask != pred_mask] = [255, 0, 0]  # Red for differences
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot original image (if grayscale, convert to RGB)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Plot ground truth mask
    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title("Ground Truth Segmentation")
    axes[0, 1].axis('off')
    
    # Plot predicted mask
    axes[1, 0].imshow(colored_pred)
    axes[1, 0].set_title("Predicted Segmentation")
    axes[1, 0].axis('off')
    
    # Plot difference
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(diff_mask, alpha=0.5)
    axes[1, 1].set_title("Prediction Errors (Red)")
    axes[1, 1].axis('off')
    
    # Add a color legend
    handles = []
    labels = []
    
    # Add the class legend below the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Add legend at the bottom
    legend_elements = []
    for i, name in enumerate(CLASS_NAMES):
        if i in COLORMAP:  # Skip if not in colormap
            color = [c/255 for c in COLORMAP[i]]  # Convert to 0-1 range for matplotlib
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=name))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(legend_elements)), frameon=False)
    
    # Save the figure
    file_name = f"test_case_{case_name}.png"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    logging.info(f"Saved test visualization to {os.path.join(save_dir, file_name)}")

def plot_metrics():
    """Generate and save plots for training metrics"""
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["epoch"], metrics["train_loss"], 'b-', marker='o')
    plt.title("Training Loss vs Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.vis_dir, "train_loss.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot validation Dice score
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["epoch"][::args.n_skip], metrics["val_dice"], 'g-', marker='o')
    plt.title("Validation Dice Score vs Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dice Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.vis_dir, "val_dice.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot validation Hausdorff distance
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["epoch"][::args.n_skip], metrics["val_hd95"], 'r-', marker='o')
    plt.title("Validation Hausdorff Distance (HD95) vs Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("HD95", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.vis_dir, "val_hd95.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["epoch"], metrics["lr"], 'c-', marker='o')
    plt.title("Learning Rate vs Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.savefig(os.path.join(args.vis_dir, "learning_rate.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # If we have class-wise dice scores
    if metrics["class_dice"] and len(metrics["class_dice"]) > 0:
        class_dice = np.array(metrics["class_dice"])
        plt.figure(figsize=(14, 8))
        
        for i in range(1, args.num_classes):  # Skip background class
            if i < len(CLASS_NAMES):
                plt.plot(metrics["epoch"][::args.n_skip], class_dice[:, i], 
                        marker='o', label=CLASS_NAMES[i])
        
        plt.title("Class-wise Dice Score vs Epochs", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Dice Score", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(args.vis_dir, "class_wise_dice.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    logging.info(f"All metric plots saved to {args.vis_dir}")

def custom_inference(args, model, testloader, test_save_path):
    """Modified inference function that also captures visualization data"""
    model.eval()
    
    # Lists to store visualization data
    case_visualized = 0
    
    # Initialize metrics
    metric_list = 0.0
    metric_list_hd = 0.0
    metric_class_dc = np.zeros(args.num_classes)
    
    with torch.no_grad():
        for case_data in tqdm(testloader, total=len(testloader), desc="Testing"):
            # Extract case information
            image, label, case_name = case_data["image"], case_data["label"], case_data["case_name"][0]
            
            # Debug information
            logging.info(f"Processing case: {case_name}, image shape: {image.shape}, label shape: {label.shape}")
            
            # Handle different input formats - ensure we have the correct input shape
            # The error shows we're getting [1, 148, 512, 512] but expect [B, 3, H, W]
            if image.shape[1] != 1 and image.shape[1] != 3:
                logging.info(f"Reshaping input from {image.shape} for case {case_name}")
                # This is likely a 3D volume - we need to process each slice
                # Take the first slice or middle slice for visualization
                slice_idx = image.shape[1] // 2
                image_slice = image[:, slice_idx:slice_idx+1, :, :]
                label_slice = label[:, slice_idx:slice_idx+1, :, :]
                
                # Debug info
                logging.info(f"Using slice {slice_idx}, new shape: {image_slice.shape}")
                
                # Move to GPU
                image_slice, label_slice = image_slice.cuda(), label_slice.cuda()
                
                # Make prediction on single slice
                output = model(image_slice)
                prediction = torch.argmax(output, dim=1).cpu().numpy()
                label_np = label_slice.cpu().numpy()
            else:
                # Process normally if dimensions are as expected
                image, label = image.cuda(), label.cuda()
                output = model(image)
                prediction = torch.argmax(output, dim=1).cpu().numpy()
                label_np = label.cpu().numpy()
            
            # Calculate metrics
            dice_scores = []
            hd_scores = []
            
            for i in range(1, args.num_classes):
                organ_dice = dc(prediction == i, label_np == i)
                dice_scores.append(organ_dice)
                
                # Calculate HD95 only if both ground truth and prediction have the organ
                if np.sum(prediction == i) > 0 and np.sum(label_np == i) > 0:
                    try:
                        organ_hd95 = hd95(prediction == i, label_np == i)
                        hd_scores.append(organ_hd95)
                    except Exception as e:
                        logging.warning(f"HD95 calculation failed for class {i}: {e}")
                        hd_scores.append(0)
                else:
                    hd_scores.append(0)
                
                # Update class-wise metrics
                metric_class_dc[i] += organ_dice
            
            # Update case metrics
            avg_case_dice = np.mean(dice_scores)
            avg_case_hd = np.mean(hd_scores) if len(hd_scores) else 0
            
            metric_list += avg_case_dice
            metric_list_hd += avg_case_hd
            
            # Log case metrics
            logging.info(f"Case {case_name}: Dice={avg_case_dice:.4f}, HD95={avg_case_hd:.4f}")
            
            # Visualize sample test cases (limited number)
            if case_visualized < args.vis_sample_count:
                try:
                    # For visualization, ensure we're working with 2D data
                    if len(prediction.shape) > 3:  # If 4D: [B,C,H,W]
                        vis_image = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].cpu().numpy().transpose(1, 2, 0)
                        vis_label = label_np[0, 0]
                        vis_pred = prediction[0, 0]
                    elif len(prediction.shape) == 3:  # If 3D: [B,H,W]
                        vis_image = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].cpu().numpy().transpose(1, 2, 0)
                        vis_label = label_np[0]
                        vis_pred = prediction[0]
                    else:
                        # Handle unexpected dimensions gracefully
                        logging.warning(f"Unexpected prediction shape: {prediction.shape}, skipping visualization")
                        continue
                        
                    # Normalize image for visualization
                    vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
                    
                    visualize_test_case(vis_image, vis_label, vis_pred, f"{case_name}")
                    case_visualized += 1
                    logging.info(f"Visualized case {case_name}")
                except Exception as e:
                    logging.error(f"Visualization failed for case {case_name}: {e}")
    
    # Calculate average metrics
    avg_dcs = metric_list / len(testloader) if len(testloader) > 0 else 0
    avg_hd = metric_list_hd / len(testloader) if len(testloader) > 0 else 0
    class_dcs = metric_class_dc / len(testloader) if len(testloader) > 0 else metric_class_dc
    
    # Log metrics
    logging.info(f"Average Dice Score: {avg_dcs:.4f}")
    logging.info(f"Average HD95: {avg_hd:.4f}")
    for i in range(1, args.num_classes):
        logging.info(f"Class {i} ({CLASS_NAMES[i] if i < len(CLASS_NAMES) else 'Unknown'}): Dice={class_dcs[i]:.4f}")
    
    # Add class dice to metrics
    metrics["class_dice"].append(class_dcs.tolist())
    
    return avg_dcs, avg_hd

def visualize_test_case(image, gt_mask, pred_mask, case_name):
    """Visualize test case with ground truth and prediction"""
    save_dir = os.path.join(args.vis_dir, "test")
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle potential dimension issues
    if len(image.shape) > 2 and image.shape[0] == 3:  # If RGB image in format [C,H,W]
        image = np.transpose(image, (1, 2, 0))
    
    # Create colored visualizations
    colored_gt = create_colored_mask(gt_mask)
    colored_pred = create_colored_mask(pred_mask)
    
    # Calculate difference mask (red where prediction differs from ground truth)
    diff_mask = np.zeros_like(colored_gt)
    diff_mask[gt_mask != pred_mask] = [255, 0, 0]  # Red for differences
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot original image (if grayscale, convert to RGB)
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    else:
        rgb_image = image
        
    # Make sure image is in range [0,1] for matplotlib
    if rgb_image.max() > 1:
        rgb_image = rgb_image / 255.0
        
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Plot ground truth mask
    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title("Ground Truth Segmentation")
    axes[0, 1].axis('off')
    
    # Plot predicted mask
    axes[1, 0].imshow(colored_pred)
    axes[1, 0].set_title("Predicted Segmentation")
    axes[1, 0].axis('off')
    
    # Plot difference
    axes[1, 1].imshow(rgb_image)
    axes[1, 1].imshow(diff_mask, alpha=0.5)
    axes[1, 1].set_title("Prediction Errors (Red)")
    axes[1, 1].axis('off')
    
    # Add the class legend below the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Add legend at the bottom
    legend_elements = []
    for i, name in enumerate(CLASS_NAMES):
        if i in COLORMAP:  # Skip if not in colormap
            color = [c/255 for c in COLORMAP[i]]  # Convert to 0-1 range for matplotlib
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=name))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
              ncol=min(5, len(legend_elements)), frameon=False)
    
    # Save the figure
    file_name = f"test_case_{case_name}.png"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    logging.info(f"Saved test visualization to {os.path.join(save_dir, file_name)}")

# Add a standalone function to generate plots from saved JSON logs
def generate_plots_from_logs(log_file_path, output_dir="./visualizations"):
    """Generate plots from saved metrics JSON file"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load metrics from JSON
        with open(log_file_path, 'r') as f:
            metrics = json.load(f)
        
        logging.info(f"Loaded metrics from {log_file_path}")
        
        # Plot training loss
        if "train_loss" in metrics and len(metrics["train_loss"]) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(metrics["epoch"], metrics["train_loss"], 'b-', marker='o')
            plt.title("Training Loss vs Epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, "train_loss.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated training loss plot")
        
        # Plot validation Dice score
        if "val_dice" in metrics and len(metrics["val_dice"]) > 0:
            plt.figure(figsize=(12, 6))
            eval_epochs = metrics["epoch"][::args.n_skip] if len(metrics["epoch"]) >= len(metrics["val_dice"]) * args.n_skip else list(range(len(metrics["val_dice"])))
            plt.plot(eval_epochs, metrics["val_dice"], 'g-', marker='o')
            plt.title("Validation Dice Score vs Epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Dice Score", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, "val_dice.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated validation dice plot")
        
        # Plot validation Hausdorff distance
        if "val_hd95" in metrics and len(metrics["val_hd95"]) > 0:
            plt.figure(figsize=(12, 6))
            eval_epochs = metrics["epoch"][::args.n_skip] if len(metrics["epoch"]) >= len(metrics["val_hd95"]) * args.n_skip else list(range(len(metrics["val_hd95"])))
            plt.plot(eval_epochs, metrics["val_hd95"], 'r-', marker='o')
            plt.title("Validation Hausdorff Distance (HD95) vs Epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("HD95", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, "val_hd95.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated Hausdorff distance plot")
        
        # Plot learning rate
        if "lr" in metrics and len(metrics["lr"]) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(metrics["epoch"], metrics["lr"], 'c-', marker='o')
            plt.title("Learning Rate vs Epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')
            plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated learning rate plot")
        
        # If we have class-wise dice scores
        if "class_dice" in metrics and metrics["class_dice"] and len(metrics["class_dice"]) > 0:
            class_dice = np.array(metrics["class_dice"])
            plt.figure(figsize=(14, 8))
            
            num_classes = len(class_dice[0])
            for i in range(1, num_classes):  # Skip background class
                if i < len(CLASS_NAMES):
                    plt.plot(eval_epochs, [cd[i] for cd in class_dice], 
                            marker='o', label=CLASS_NAMES[i])
                else:
                    plt.plot(eval_epochs, [cd[i] for cd in class_dice], 
                            marker='o', label=f"Class {i}")
            
            plt.title("Class-wise Dice Score vs Epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Dice Score", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(output_dir, "class_wise_dice.png"), dpi=150, bbox_inches='tight')
            plt.close()
            logging.info(f"Generated class-wise dice score plot")
        
        logging.info(f"All metric plots saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error generating plots from logs: {e}")
        return False

# Main training code
def main():
    # Initialize model
    model = MTUNet(args.num_classes)
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        logging.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create datasets and loaders
    train_dataset = Synapsedataset(
        args.root_dir, 
        args.list_dir, 
        split="train", 
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )
    
    Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    db_test = Synapsedataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    # Move model to GPU if available
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model = model.cuda()
    model.train()
    
    # Initialize loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    
    # Set save interval
    save_interval = args.n_skip
    
    # Initialize training variables
    iterator = tqdm(range(0, args.max_epochs), ncols=70)
    iter_num = 0
    Best_dcs = 0
    
    # Calculate max iterations
    max_iterations = args.max_epochs * len(Train_loader)
    base_lr = args.lr
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    # Start training loop
    for epoch in iterator:
        model.train()
        train_loss = 0
        
        # Train batch loop
        for i_batch, sampled_batch in enumerate(Train_loader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # Forward pass
            outputs = model(image_batch)
            
            # Calculate loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
            loss = loss_dice * 0.5 + loss_ce * 0.5
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            
            # Log current iteration
            logging.info(f'iteration {iter_num} : loss : {loss.item():.4f} lr_: {lr_:.6f}')
            train_loss += loss.item()
            
            # Visualize sample training batch
            if i_batch % 50 == 0:  # Adjust frequency as needed
                visualize_batch(image_batch, label_batch, outputs, epoch, i_batch, "train")
        
        # Calculate average loss for the epoch
        avg_train_loss = train_loss / len(Train_loader)
        logging.info(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")
        
        # Update metrics
        metrics["train_loss"].append(avg_train_loss)
        metrics["lr"].append(lr_)
        metrics["epoch"].append(epoch)
        
        # Save metrics to JSON after each epoch
        save_metrics_to_json()
        
        # Run validation
        if (epoch + 1) % save_interval == 0:
            # Run custom inference with visualization
            avg_dcs, avg_hd = custom_inference(args, model, testloader, args.test_save_dir)
            
            # Update metrics
            metrics["val_dice"].append(avg_dcs)
            metrics["val_hd95"].append(avg_hd)
            
            # Save model if it's the best so far
            if avg_dcs > Best_dcs:
                save_mode_path = os.path.join(
                    args.save_path, 
                    f'epoch{epoch}_lr{lr_:.6f}_dice{avg_dcs:.4f}_hd{avg_hd:.4f}.pth'
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"New best model! Saved to {save_mode_path}")
                
                # Update best metrics
                Best_dcs = avg_dcs
                metrics["best_model_path"] = save_mode_path
            
            # Generate and save plots after each validation
            plot_metrics()
            
            # Save updated metrics to JSON
            save_metrics_to_json()
        
        # Save final model
        if epoch >= args.max_epochs - 1:
            save_mode_path = os.path.join(
                args.save_path, 
                f'final_epoch{epoch}_lr{lr_:.6f}_dice{avg_dcs:.4f}.pth'
            )
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Training finished. Final model saved to {save_mode_path}")
            iterator.close()
            break
    
    # Final visualization of metrics
    plot_metrics()
    
    # Generate a summary report
    with open(os.path.join(args.vis_dir, "training_summary.txt"), "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"==============\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of epochs: {args.max_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Initial learning rate: {args.lr}\n")
        f.write(f"Final learning rate: {lr_:.6f}\n")
        f.write(f"Best validation Dice score: {Best_dcs:.4f}\n")
        f.write(f"Best model saved at: {metrics['best_model_path']}\n")

if __name__ == "__main__":
    main()
