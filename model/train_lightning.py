import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from torchvision import transforms
from process_data import FullDataset
from dt_net import DTNet

from tqdm import tqdm

import time

import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

import argparse
"""

Note: This is testing code and not final version of training code

"""


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, pred, target):
        eps = 1e-7
        pred_log = torch.log(pred + eps)
        target_log = torch.log(target + eps)
        diff_log = pred_log - target_log

        diff_log_sq = diff_log ** 2
        N = torch.numel(diff_log)
        first_term = diff_log_sq.mean()

        second_term = (diff_log.sum() ** 2) / (N ** 2)

        return first_term - second_term

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_dy, pred_dx = self.gradient(pred)
        target_dy, target_dx = self.gradient(target)
        grad_diff_y = torch.abs(pred_dy - target_dy)
        grad_diff_x = torch.abs(pred_dx - target_dx)
        return grad_diff_y.mean() + grad_diff_x.mean()

    def gradient(self, x):
        D_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return D_dy, D_dx


class LightningDTModel(L.LightningModule):
    def __init__(self, model_name='base', learning_rate=1e-3, output_mask=None):
        super(LightningDTModel, self).__init__()
        
        if model_name == 'densenet':
            self.model = DTNet(n_heads=1, head_output_channels=7, encoder='densenet')
        elif model_name == 'resnet':
            self.model = DTNet(n_heads=1, head_output_channels=7, encoder='resnet')
 
        self.criterion = nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.learning_rate = learning_rate
        self.file_num = 0
        self.output_mask = output_mask
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model(X)
        # outputs = torch.cat(outputs, dim=1)
        if self.output_mask is not None:
            outputs = outputs * self.output_mask.to(outputs.device)
            
        loss = self.criterion(outputs, y)
        mse = self.mse(outputs, y)
        l1 = self.l1_loss(outputs, y)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_l1', l1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # get error between non zero values in y and corresponding values in output
        y_non_zero = y[y != 0]
        output_non_zero = outputs[y != 0]
        return_loss = self.mse(output_non_zero, y_non_zero)
        self.log('val_mse_non_zero_loss', return_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if batch_idx == 0:
            # take images and diff from batch
            deformed_img = X[0][:3].cpu().detach().numpy()
            undeformed_img = X[0][3:6].cpu().detach().numpy()
            img_diff = X[0][6].cpu().detach().numpy()
            
            # Normalize the images if necessary
            def normalize_img(img):
                    if img.dtype != np.uint8:
                        img = 255 * img
                        img = img.astype(np.uint8)
                    return img

            deformed_img = normalize_img(deformed_img)
            undeformed_img = normalize_img(undeformed_img)
            img_diff = normalize_img(img_diff)
            
            # Add channel dimension to img_diff to make it [1, H, W]
            img_diff = np.expand_dims(img_diff, axis=0)
            
            # Convert images to tensors
            deformed_img_tensor = torch.tensor(deformed_img)
            undeformed_img_tensor = torch.tensor(undeformed_img)
            img_diff_tensor = torch.tensor(img_diff)
            
            # Log images to TensorBoard
            self.logger.experiment.add_image('val_deformed_image', deformed_img_tensor, self.file_num)
            self.logger.experiment.add_image('val_undeformed_image', undeformed_img_tensor, self.file_num)
            self.logger.experiment.add_image('val_image_difference', img_diff_tensor, self.file_num)
            
            output_1 = outputs[0][0].cpu().detach().numpy()
            # Ensure the image tensor is in the correct format and normalize if necessary
            image_tensor = torch.tensor(output_1).unsqueeze(0)  # Add channel dimension
            if image_tensor.dtype != torch.uint8:
                image_tensor = (255 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())).byte()
                
            gt_depth_1 = y[0][0].cpu().detach().numpy()
            gt_depth_tensor = torch.tensor(gt_depth_1).unsqueeze(0)
            if gt_depth_tensor.dtype != torch.uint8:
                gt_depth_tensor = (255 * (gt_depth_tensor - gt_depth_tensor.min()) / (gt_depth_tensor.max() - gt_depth_tensor.min())).byte()
            
            # Add image to TensorBoard
            self.logger.experiment.add_image('val_output_depth_displacement', image_tensor, self.file_num)
            self.logger.experiment.add_image('val_gt_depth_displacement', gt_depth_tensor, self.file_num)
                
            self.file_num += 1
        
        return loss
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model(X)
        
        # outputs = torch.cat(outputs, dim=1)
        loss = self.criterion(outputs, y)
        mse = self.mse(outputs, y)
        l1 = self.l1_loss(outputs, y)
        
        if self.output_mask is not None:
            outputs = outputs * self.output_mask.to(outputs.device)
                  
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_l1', l1, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        y_non_zero = y[y != 0]
        output_non_zero = outputs[y != 0]
        return_loss = self.mse(output_non_zero, y_non_zero)
        self.log('train_mse_non_zero_loss', return_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # log sample input and output at each epoch
        if batch_idx == 0:
            # take images and diff from batch
            deformed_img = X[0][:3].cpu().detach().numpy()
            undeformed_img = X[0][3:6].cpu().detach().numpy()
            img_diff = X[0][6].cpu().detach().numpy()
            
            # Normalize the images if necessary
            def normalize_img(img):
                    if img.dtype != np.uint8:
                        img = 255 * img
                        img = img.astype(np.uint8)
                    return img

            deformed_img = normalize_img(deformed_img)
            undeformed_img = normalize_img(undeformed_img)
            img_diff = normalize_img(img_diff)
            
            # Add channel dimension to img_diff to make it [1, H, W]
            img_diff = np.expand_dims(img_diff, axis=0)
            
            # Convert images to tensors
            deformed_img_tensor = torch.tensor(deformed_img)
            undeformed_img_tensor = torch.tensor(undeformed_img)
            img_diff_tensor = torch.tensor(img_diff)
            
            # Log images to TensorBoard
            self.logger.experiment.add_image('deformed_image', deformed_img_tensor, self.file_num)
            self.logger.experiment.add_image('undeformed_image', undeformed_img_tensor, self.file_num)
            self.logger.experiment.add_image('image_difference', img_diff_tensor, self.file_num)
            
            output_1 = outputs[0][0].cpu().detach().numpy()
            # Ensure the image tensor is in the correct format and normalize if necessary
            image_tensor = torch.tensor(output_1).unsqueeze(0)  # Add channel dimension
            if image_tensor.dtype != torch.uint8:
                image_tensor = (255 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())).byte()
                
            gt_depth_1 = y[0][0].cpu().detach().numpy()
            gt_depth_tensor = torch.tensor(gt_depth_1).unsqueeze(0)
            if gt_depth_tensor.dtype != torch.uint8:
                gt_depth_tensor = (255 * (gt_depth_tensor - gt_depth_tensor.min()) / (gt_depth_tensor.max() - gt_depth_tensor.min())).byte()
            
            # Add image to TensorBoard
            self.logger.experiment.add_image('output_depth_displacement', image_tensor, self.file_num)
            self.logger.experiment.add_image('gt_depth_displacement', gt_depth_tensor, self.file_num)
            if output_1.dtype != np.uint8:
                output_1 = cv2.normalize(output_1, None, 0, 255, cv2.NORM_MINMAX)
                output_1 = np.uint8(output_1)
                # save as image with colormap
                depth_colormap = cv2.applyColorMap(output_1, cv2.COLORMAP_JET)
                cv2.imwrite(f'viz_lightning/{self.file_num}_output_resnet_skip.png', depth_colormap)
                
            self.file_num += 1
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Args for DenseTact Calibration Training.')
    
    parser.add_argument('-m', '--model-save-folder', type=str, required=True, help='Name of the user', default='models/dt_model_1/')
    parser.add_argument('-s', '--samples_dir', type=str, required=True, help='Path to samples directory', default='real_world_data/')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    model_save_folder = args.model_save_folder
    samples_dir = args.samples_dir
    print(f"Model save folder: {model_save_folder}")
    
    train = True
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # default transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    
    dataset = FullDataset(transform=transform, samples_dir=samples_dir)
    output_mask = dataset.get_output_mask()
    output_mask = transform(output_mask)
    
    print(len(dataset), 'length')
    full_dataset_length = len(dataset)
    # full_dataset_length = 132546
    
    # take only 10 percent of dataset for train and test
    dataset_length = int(1 * full_dataset_length)
    train_size = int(0.8 * dataset_length)
    val_size = dataset_length - train_size
    
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}")
    
    train_batch_size = 16
    
    dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=24)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=12)
    
    calibration_model = LightningDTModel(model_name='densenet', learning_rate=1e-3, output_mask=output_mask)
    # trainer = L.Trainer(limit_train_batches=1000, max_epochs=1000)
    log_dir = 'lightning_logs'
    name = 'dt_densenet_depth'
    logger = TensorBoardLogger(save_dir=log_dir, name=name)
    
    trainer = L.Trainer(limit_train_batches=2000, max_epochs=1000, strategy='ddp_find_unused_parameters_true', logger=logger)
    print("Starting...")
    
    trainer.fit(model=calibration_model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)