#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from utils.utils import DiceLoss
from torch.utils.data import DataLoader
from dataset.dataset_ACDC import ACDCdataset, RandomGenerator
import argparse
from tqdm import tqdm
import os
from torchvision import transforms
import numpy as np
from medpy.metric import dc, hd95
import cv2
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=100)
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
parser.add_argument('--visual_dir', default='./visualizations', help='directory for saving visualizations')
parser.add_argument("--patches_size", default=16)
parser.add_argument("--n_skip", default=1)
args = parser.parse_args()

# Create visualization directories if they don't exist
os.makedirs(args.visual_dir, exist_ok=True)
os.makedirs(os.path.join(args.visual_dir, 'inference'), exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Defining colormap for segmentation visualization
colors = ['black', 'red', 'green', 'blue']
cmap = ListedColormap(colors[:args.num_classes])

def inference(args, model, testloader, test_save_dir):
    model.eval()
    os.makedirs(test_save_dir, exist_ok=True)
    
    dice_list = []
    hd_list = []
    
    for idx, test_sampled_batch in enumerate(testloader):
        test_image_batch, test_label_batch = test_sampled_batch["image"], test_sampled_batch["label"]
        test_image_batch, test_label_batch = test_image_batch.type(torch.FloatTensor), test_label_batch.type(torch.FloatTensor)
        test_image_batch, test_label_batch = test_image_batch.cuda().unsqueeze(1), test_label_batch.cuda().unsqueeze(1)
        
        with torch.no_grad():
            test_outputs = model(test_image_batch)
            test_pred = torch.argmax(torch.softmax(test_outputs, dim=1), dim=1).squeeze(0)
            
        # Calculate metrics
        test_pred_np = test_pred.cpu().data.numpy()
        test_label_np = test_label_batch.squeeze(0).squeeze(0).cpu().data.numpy()
        
        dice_score = dc(test_pred_np, test_label_np)
        hd_score = 0
        try:
            hd_score = hd95(test_pred_np, test_label_np)
        except:
            pass
        
        dice_list.append(dice_score)
        hd_list.append(hd_score)
        
        # Visualization of results
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(test_image_batch.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(test_label_np, cmap=cmap, vmin=0, vmax=args.num_classes-1)
        plt.title(f'Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 3, 3)
        plt.imshow(test_pred_np, cmap=cmap, vmin=0, vmax=args.num_classes-1)
        plt.title(f'Prediction\nDice: {dice_score:.4f}, HD95: {hd_score:.4f}')
        plt.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(args.visual_dir, 'inference', f'case_{idx+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f'Case {idx+1} - Dice: {dice_score:.4f}, HD95: {hd_score:.4f}')
    
    avg_dice = np.mean(dice_list)
    avg_hd = np.mean(hd_list)
    logger.info(f'Average Dice: {avg_dice:.4f}, Average HD95: {avg_hd:.4f}')
    
    return avg_dice, avg_hd

def calculate_metrics_batch(outputs, labels):
    """Calculate Dice and HD95 for a batch"""
    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    dice_scores = []
    hd_scores = []
    
    for i in range(outputs.shape[0]):
        pred = outputs[i].cpu().data.numpy()
        label = labels[i].cpu().data.numpy()
        dice_scores.append(dc(pred, label))
        try:
            hd_scores.append(hd95(pred, label))
        except:
            hd_scores.append(0)
    
    return np.mean(dice_scores), np.mean(hd_scores)

def val():
    logger.info("Validation ===>")
    model.eval()
    dice_list = []
    hd_list = []
    
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        
        with torch.no_grad():
            val_outputs = model(val_image_batch)
            val_pred = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
            
        dice_score = dc(val_pred.cpu().data.numpy(), val_label_batch.squeeze(0).squeeze(0).cpu().data.numpy())
        dice_list.append(dice_score)
        
        # Calculate HD95 if possible
        try:
            hd_score = hd95(val_pred.cpu().data.numpy(), val_label_batch.squeeze(0).squeeze(0).cpu().data.numpy())
            hd_list.append(hd_score)
        except:
            hd_list.append(0)
    
    avg_dice = np.mean(dice_list)
    avg_hd = np.mean(hd_list)
    logger.info(f"Validation - Average Dice: {avg_dice:.4f}, Average HD95: {avg_hd:.4f}")
    return avg_dice, avg_hd

# Initialize model
model = MTUNet(args.num_classes)

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

# Data Loaders
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

# Loss functions
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip

# Iterator for epochs
iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

# Metrics tracking
train_losses = []
train_dice_scores = []
train_hd_scores = []
val_dice_scores = []
val_hd_scores = []
test_dice_scores = []
test_hd_scores = []
epochs = []

Best_dice = 0.0
max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)

# Training loop
for epoch in iterator:
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_hd = 0
    train_samples = 0
    
    # Track current epoch
    epochs.append(epoch + 1)
    
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

        # Calculate learning rate
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        
        # Calculate metrics for this batch
        dice_score, hd_score = calculate_metrics_batch(outputs, label_batch)
        
        # Accumulate batch statistics
        batch_size = image_batch.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_dice += dice_score * batch_size
        epoch_hd += hd_score * batch_size
        train_samples += batch_size
        
        if i_batch % 10 == 0:
            logger.info(f'Iteration {iter_num}: Loss: {loss.item():.4f}, Dice: {dice_score:.4f}, HD95: {hd_score:.4f}, LR: {lr_:.6f}')

    # Average metrics for the epoch
    avg_loss = epoch_loss / train_samples
    avg_dice = epoch_dice / train_samples
    avg_hd = epoch_hd / train_samples
    
    # Save metrics
    train_losses.append(avg_loss)
    train_dice_scores.append(avg_dice)
    train_hd_scores.append(avg_hd)
    
    logger.info(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.4f}, Train HD95: {avg_hd:.4f}')

    # Validation and Testing at intervals
    if (epoch + 1) % save_interval == 0:
        # Run validation
        val_dice, val_hd = val()
        val_dice_scores.append(val_dice)
        val_hd_scores.append(val_hd)
        
        # Run testing
        test_dice, test_hd = inference(args, model, testloader, args.test_save_dir)
        test_dice_scores.append(test_dice)
        test_hd_scores.append(test_hd)
        
        # Update checkpoint if validation dice improved
        if val_dice > Best_dice:
            save_mode_path = os.path.join(args.save_path, f'epoch={epoch+1}_dice={val_dice:.4f}_hd={val_hd:.4f}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info(f"Model saved to {save_mode_path}")
            Best_dice = val_dice
        
        # Generate visualizations after each validation interval
        
        # 1. Training Loss Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
        plt.title('Training Loss vs Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(args.visual_dir, 'train_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Training Dice Score Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_dice_scores, 'r-', marker='o', label='Training Dice')
        plt.title('Training Dice Score vs Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Dice Score', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(args.visual_dir, 'train_dice.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Training HD95 Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_hd_scores, 'g-', marker='o', label='Training HD95')
        plt.title('Training HD95 vs Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('HD95 Score', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(args.visual_dir, 'train_hd95.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Validation Metrics
        val_epochs = list(range(save_interval, epoch+2, save_interval))
        if len(val_epochs) > 0:
            # 4.1 Validation Dice Score
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, val_dice_scores, 'm-', marker='s', label='Validation Dice')
            plt.title('Validation Dice Score vs Epochs', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Dice Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'val_dice.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4.2 Validation HD95 Score
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, val_hd_scores, 'c-', marker='s', label='Validation HD95')
            plt.title('Validation HD95 vs Epochs', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('HD95 Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'val_hd95.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Test Metrics
            # 5.1 Test Dice Score
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, test_dice_scores, 'y-', marker='d', label='Test Dice')
            plt.title('Test Dice Score vs Epochs', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Dice Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'test_dice.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5.2 Test HD95 Score
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, test_hd_scores, 'k-', marker='d', label='Test HD95')
            plt.title('Test HD95 vs Epochs', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('HD95 Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'test_hd95.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. Combined Metrics
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_dice_scores, 'r-', marker='o', label='Train Dice')
            plt.plot(val_epochs, val_dice_scores, 'm-', marker='s', label='Val Dice')
            plt.plot(val_epochs, test_dice_scores, 'y-', marker='d', label='Test Dice')
            plt.title('Dice Score Comparison', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Dice Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'combined_dice.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_hd_scores, 'g-', marker='o', label='Train HD95')
            plt.plot(val_epochs, val_hd_scores, 'c-', marker='s', label='Val HD95')
            plt.plot(val_epochs, test_hd_scores, 'k-', marker='d', label='Test HD95')
            plt.title('HD95 Score Comparison', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('HD95 Score', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.savefig(os.path.join(args.visual_dir, 'combined_hd95.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Save final model
    if epoch >= args.max_epochs - 1:
        save_mode_path = os.path.join(args.save_path, f'final_epoch={epoch+1}_dice={val_dice:.4f}_hd={val_hd:.4f}.pth')
        torch.save(model.state_dict(), save_mode_path)
        logger.info(f"Final model saved to {save_mode_path}")
        iterator.close()
        break

logger.info("Training completed!")
