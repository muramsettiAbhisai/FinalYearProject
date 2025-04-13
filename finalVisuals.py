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
from dataset.dataset_ACDC import ACDCdataset, RandomGenerator
import argparse
from tqdm import tqdm
import os
import csv
from torchvision import transforms
from utils.test_ACDC import inference
from model.MTUNet import MTUNet
import numpy as np
from medpy.metric import dc, hd95
import nibabel as nib
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=2)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./checkpoint")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="../dataset/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="../dataset/ACDC/")
parser.add_argument("--volume_path", default="../dataset/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--visualization_dir', default='./visualizations', help='saving visualizations')
parser.add_argument('--log_dir', default='./logs', help='saving logs')
parser.add_argument("--patches_size", default=16)
parser.add_argument("--n_skip", default=1)
args = parser.parse_args()

# Create directories if they don't exist
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.test_save_dir, exist_ok=True)
os.makedirs(args.visualization_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s',
                   handlers=[
                       logging.FileHandler(os.path.join(args.log_dir, 'training.log')),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger()

# Create CSV log file for metrics
csv_path = os.path.join(args.log_dir, 'metrics.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_DSC', 'Val_HD95', 'LR'])

# Create model
model = MTUNet(args.num_classes)  # 4

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))
    logging.info(f"Loaded checkpoint from {args.checkpoint}")

# Create datasets
train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    model = nn.DataParallel(model)

model = model.cuda()
model.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

train_losses = []
val_dsc_scores = []
val_hd95_scores = []
lr_values = []

Best_dsc = 0.1
max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

# Color mapping for visualization
# Define a colormap for segmentation visualization
class_colors = {
    0: [0, 0, 0],      # Background - black
    1: [255, 0, 0],    # Class 1 - red (RV)
    2: [0, 255, 0],    # Class 2 - green (MYO)
    3: [0, 0, 255]     # Class 3 - blue (LV)
}

def create_colored_mask(mask):
    """
    Create a colored version of the segmentation mask
    """
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        colored_mask[mask == class_idx] = color
    return colored_mask

def compute_metrics(outputs, labels):
    """
    Compute DSC and HD95 for all classes
    """
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()
    
    class_dsc = []
    class_hd = []
    
    # Skip background class (0)
    for c in range(1, args.num_classes):
        pred_c = (outputs == c).astype(int)
        label_c = (labels == c).astype(int)
        
        if np.sum(label_c) > 0:
            dsc_score = dc(pred_c, label_c)
            class_dsc.append(dsc_score)
            
            try:
                hd_score = hd95(pred_c, label_c)
                class_hd.append(hd_score)
            except:
                # When there's a problem computing HD95, use a large value
                class_hd.append(100)
    
    avg_dsc = np.mean(class_dsc) if class_dsc else 0
    avg_hd = np.mean(class_hd) if class_hd else 100
    
    return avg_dsc, avg_hd

def val():
    """
    Validate the model on validation set
    """
    logging.info("Validation ===>")
    model.eval()
    
    val_dsc_sum = 0
    val_hd_sum = 0
    
    with torch.no_grad():
        for i, val_sampled_batch in enumerate(valloader):
            val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
            val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
            val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)

            val_outputs = model(val_image_batch)
            val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1)
            
            dsc, hd = compute_metrics(val_outputs, val_label_batch.squeeze(1))
            val_dsc_sum += dsc
            val_hd_sum += hd

    avg_dsc = val_dsc_sum / len(valloader)
    avg_hd = val_hd_sum / len(valloader)
    
    logging.info(f"Validation - Avg DSC: {avg_dsc:.4f}, Avg HD95: {avg_hd:.4f}")
    return avg_dsc, avg_hd

def visualize_predictions(epoch, best=False):
    """
    Create visualizations of ground truth vs predictions for a few validation samples
    """
    logging.info("Creating visualizations...")
    model.eval()
    
    # Choose a few samples to visualize (here we'll take the first 3)
    num_samples = min(3, len(valloader))
    viz_dir = os.path.join(args.visualization_dir, f"epoch_{epoch}")
    os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, val_sampled_batch in enumerate(valloader):
            if i >= num_samples:
                break
                
            val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
            val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
            val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda()
            
            # Get predictions
            val_outputs = model(val_image_batch)
            val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1)
            
            # Convert to numpy for visualization
            val_image = val_image_batch.squeeze().cpu().numpy()
            val_label = val_label_batch.squeeze().cpu().numpy()
            val_pred = val_outputs.squeeze().cpu().numpy()
            
            # Create colored masks
            colored_gt = create_colored_mask(val_label)
            colored_pred = create_colored_mask(val_pred)
            
            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(val_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(colored_gt)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(colored_pred)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in class_colors[0]], label='Background'),
                plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in class_colors[1]], label='RV'),
                plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in class_colors[2]], label='MYO'),
                plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in class_colors[3]], label='LV')
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=4)
            
            plt.tight_layout()
            
            # Save the visualization
            suffix = "_best" if best else ""
            plt.savefig(os.path.join(viz_dir, f"sample_{i}{suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            if best:
                # For best model, save overlay visualization too
                plt.figure(figsize=(10, 8))
                plt.imshow(val_image, cmap='gray')
                plt.imshow(colored_pred, alpha=0.5)
                plt.title('Best Model Prediction Overlay')
                plt.axis('off')
                
                # Add legend
                plt.legend(handles=legend_elements, loc='lower center', ncol=4)
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"sample_{i}_overlay{suffix}.png"), dpi=300, bbox_inches='tight')
                plt.close()

def calculate_val_dsc_per_class():
    """
    Calculate DSC for each class on validation set
    """
    model.eval()
    class_dsc = [[] for _ in range(args.num_classes)]
    
    with torch.no_grad():
        for i, val_sampled_batch in enumerate(valloader):
            val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
            val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
            val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)

            val_outputs = model(val_image_batch)
            val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1)
            
            # Convert to numpy
            outputs = val_outputs.cpu().numpy()
            labels = val_label_batch.squeeze(1).cpu().numpy()
            
            # Calculate DSC for each class (skip background)
            for c in range(1, args.num_classes):
                pred_c = (outputs == c).astype(int)
                label_c = (labels == c).astype(int)
                
                if np.sum(label_c) > 0:
                    dsc_score = dc(pred_c, label_c)
                    class_dsc[c].append(dsc_score)
    
    # Calculate average DSC for each class
    avg_class_dsc = []
    for c in range(1, args.num_classes):
        if class_dsc[c]:
            avg_class_dsc.append(np.mean(class_dsc[c]))
        else:
            avg_class_dsc.append(0)
    
    class_names = ['RV', 'MYO', 'LV']
    return class_names, avg_class_dsc

def plot_metrics():
    """
    Generate and save plots for all metrics
    """
    logging.info("Generating metrics plots...")
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation DSC plot
    plt.figure(figsize=(10, 6))
    epochs = range(0, len(val_dsc_scores) * save_interval, save_interval)
    plt.plot(epochs, val_dsc_scores, 'g-', label='Validation DSC')
    plt.title('Validation Dice Similarity Coefficient vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'val_dsc_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Validation HD95 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_hd95_scores, 'r-', label='Validation HD95')
    plt.title('Validation Hausdorff Distance 95% vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'val_hd95_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_values, 'k-')
    plt.title('Learning Rate vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined metrics plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_dsc_scores, 'g-')
    plt.title('Validation DSC Score')
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_hd95_scores, 'r-')
    plt.title('Validation HD95')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, lr_values, 'k-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class-wise DSC for best model (if applicable)
    if len(val_dsc_scores) > 0:
        # Load best model first
        best_epoch_idx = val_dsc_scores.index(max(val_dsc_scores))
        best_epoch = best_epoch_idx * save_interval
        
        # Calculate class-wise DSC
        class_names, class_dsc = calculate_val_dsc_per_class()
        
        # Plot class-wise DSC
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, class_dsc, color=['red', 'green', 'blue'])
        plt.title('Class-wise Dice Similarity Coefficient (Best Model)')
        plt.xlabel('Class')
        plt.ylabel('DSC')
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(args.log_dir, 'class_wise_dsc.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Main training loop
for epoch in iterator:
    model.train()
    train_loss = 0
    
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        loss = loss_dice * 0.5 + loss_ce * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        logging.info(f'Iteration {iter_num}: Loss: {loss.item():.4f}, LR: {lr_:.6f}')
        train_loss += loss.item()
    
    # Calculate average training loss for this epoch
    avg_train_loss = train_loss / len(Train_loader)
    train_losses.append(avg_train_loss)
    
    logging.info(f"Epoch {epoch+1}/{args.max_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    # Validation
    if (epoch + 1) % save_interval == 0:
        val_dsc, val_hd = val()
        val_dsc_scores.append(val_dsc)
        val_hd95_scores.append(val_hd)
        lr_values.append(lr_)
        
        # Run inference on test set (but don't plot it)
        test_dsc, test_hd = inference(args, model, testloader, args.test_save_dir)
        
        # Save metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_dsc, val_hd, lr_])
        
        # Check if this is the best model
        if val_dsc > Best_dsc:
            Best_dsc = val_dsc
            Best_epoch = epoch
            Best_hd = val_hd
            save_mode_path = os.path.join(args.save_path, f'epoch={epoch}_dsc={val_dsc:.4f}_hd={val_hd:.4f}_best.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved best model to {save_mode_path}")
            
            # Create visualizations for best model
            visualize_predictions(epoch, best=True)
        else:
            save_mode_path = os.path.join(args.save_path, f'epoch={epoch}_dsc={val_dsc:.4f}_hd={val_hd:.4f}.pth')
            torch.save(model.state_dict(), save_mode_path)
            
            # Create regular visualizations
            visualize_predictions(epoch)
        
        # Generate plots
        plot_metrics()
        
    # Save final model
    if epoch >= args.max_epochs - 1:
        final_val_dsc, final_val_hd = val()
        save_mode_path = os.path.join(args.save_path, f'final_epoch={epoch}_dsc={final_val_dsc:.4f}_hd={final_val_hd:.4f}.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info(f"Saved final model to {save_mode_path}")
        
        # Final visualization
        visualize_predictions(epoch)
        
        # Final plots
        plot_metrics()
        
        # Print best model information
        logging.info(f"Best model was at epoch {Best_epoch} with DSC={Best_dsc:.4f} and HD95={Best_hd:.4f}")
        
        iterator.close()
        break

logging.info("Training completed!")
