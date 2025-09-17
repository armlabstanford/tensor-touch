import os
import os.path as osp
import cv2
import math
import numpy as np
import argparse

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as F
from process_data import FullDataset
from lightning.pytorch.loggers import TensorBoardLogger

import lightning as L
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from models import pretrain_dict
from util.loss_util import ssim
from util.scheduler_util import LinearWarmupCosineAnnealingLR

from configs import get_cfg_defaults


class LightningDTModel(L.LightningModule):
    def __init__(self, 
                 model_name='base', 
                 num_epochs=100,
                 total_steps=1e6,
                 mask_ratio=0.75,
                 learning_rate=1e-3
        ):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()

        if model_name in pretrain_dict:
            if 'vit' in model_name:
                self.model = pretrain_dict[model_name](img_size=256, in_chans=6)
            else:
                self.model = pretrain_dict[model_name](input_size=256, in_chans=6)
        else:
            print("Model {} not found in pretrain_dict".format(model_name))
            print("Available models: {}".format(pretrain_dict.keys()))
            exit()

        self.criterion = nn.L1Loss()
        self.mse = nn.MSELoss()
        print(self.model)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.total_steps = total_steps
        self.mask_ratio = mask_ratio
        
        # Color jitter parameters
        self.brightness = 0.25
        self.contrast = 0.25
        self.saturation = 0.25
        self.hue = 0.02
        
        # Random crop transform
        self.random_crop = transforms.RandomResizedCrop(
            size=(256, 256), 
            scale=(0.3, 1.0), 
            ratio=(0.9, 1.1),
            antialias=True
        )

    def setup(self, stage=None):
        # populate only once
        if self.global_rank == 0:
            # Heavy I/O into SHARED_DATA
            SHARED_DATA.zero_()                      # or copy_ from disk, etc.
        # ensure others wait until it's ready
        self.trainer.strategy.barrier()
        
    def apply_paired_color_jitter(self, x):
        """Apply different color jitter params to each 3-channel half of 6-channel input"""
        # x shape: (N, 6, H, W)
        N, C, H, W = x.shape
        
        # Split into deformed and undeformed
        deformed = x[:, :3]  # (N, 3, H, W)
        undeformed = x[:, 3:]  # (N, 3, H, W)
        
        # Sample different jitter parameters for each half
        # Deformed image parameters
        def_brightness = torch.empty(N, device=x.device).uniform_(1-self.brightness, 1+self.brightness)
        def_contrast = torch.empty(N, device=x.device).uniform_(1-self.contrast, 1+self.contrast)
        def_saturation = torch.empty(N, device=x.device).uniform_(1-self.saturation, 1+self.saturation)
        def_hue = torch.empty(N, device=x.device).uniform_(-self.hue, self.hue)
        
        # Undeformed image parameters (different from deformed)
        undef_brightness = torch.empty(N, device=x.device).uniform_(1-self.brightness, 1+self.brightness)
        undef_contrast = torch.empty(N, device=x.device).uniform_(1-self.contrast, 1+self.contrast)
        undef_saturation = torch.empty(N, device=x.device).uniform_(1-self.saturation, 1+self.saturation)
        undef_hue = torch.empty(N, device=x.device).uniform_(-self.hue, self.hue)
        
        # Apply transforms with different parameters for each half
        for i in range(N):
            # Different random order for each image
            ops = ['brightness', 'contrast', 'saturation', 'hue']
            def_order = torch.randperm(4)
            undef_order = torch.randperm(4)
            
            # Apply to deformed image
            for idx in def_order:
                op = ops[idx]
                if op == 'brightness':
                    deformed[i] = F.adjust_brightness(deformed[i], def_brightness[i].item())
                elif op == 'contrast':
                    deformed[i] = F.adjust_contrast(deformed[i], def_contrast[i].item())
                elif op == 'saturation':
                    deformed[i] = F.adjust_saturation(deformed[i], def_saturation[i].item())
                elif op == 'hue':
                    deformed[i] = F.adjust_hue(deformed[i], def_hue[i].item())
            
            # Apply to undeformed image with different parameters
            for idx in undef_order:
                op = ops[idx]
                if op == 'brightness':
                    undeformed[i] = F.adjust_brightness(undeformed[i], undef_brightness[i].item())
                elif op == 'contrast':
                    undeformed[i] = F.adjust_contrast(undeformed[i], undef_contrast[i].item())
                elif op == 'saturation':
                    undeformed[i] = F.adjust_saturation(undeformed[i], undef_saturation[i].item())
                elif op == 'hue':
                    undeformed[i] = F.adjust_hue(undeformed[i], undef_hue[i].item())
        
        # Clamp to valid range and concatenate
        deformed = torch.clamp(deformed, 0.0, 1.0)
        undeformed = torch.clamp(undeformed, 0.0, 1.0)
        
        return torch.cat([deformed, undeformed], dim=1)
    
    def save_augmented_images(self, X_original, X_augmented, batch_idx):
        """Save original and augmented images side by side for visualization"""
        import matplotlib.pyplot as plt
        
        # Create augmented_images directory if it doesn't exist
        os.makedirs('augmented_images', exist_ok=True)
        
        # Get first sample from batch for visualization
        orig_deform = X_original[0, :3, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        orig_undeform = X_original[0, 3:6, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        
        aug_deform = X_augmented[0, :3, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        aug_undeform = X_augmented[0, 3:6, :, :].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(orig_deform)
        axes[0, 0].set_title('Original Deformed')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orig_undeform)
        axes[0, 1].set_title('Original Undeformed')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(aug_deform)
        axes[1, 0].set_title('Augmented Deformed')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(aug_undeform)
        axes[1, 1].set_title('Augmented Undeformed')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'augmented_images/augmented_batch_{batch_idx}_epoch_{self.current_epoch}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def apply_random_crop(self, x):
        """Apply random resized crop efficiently using torchvision"""
        # x shape: (N, 6, H, W)
        return self.random_crop(x)
        
    def training_step(self, batch, batch_idx):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, _ = batch 
        X_original = X.clone()  # Save original for comparison
        
        # Apply random crop first, then color jitter for data augmentation
        X = self.apply_random_crop(X)
        X = self.apply_paired_color_jitter(X)
        
        N, C, H, W = X.shape 
        loss, pred, mask = self.model(X, self.mask_ratio) 

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        with torch.no_grad():
            pred = pred.detach().clone()
            X = X.detach().clone()
            
            pred = self.model.reconstruct_img(pred)

            # compute PSNR
            mse = self.mse(pred, X)
            psnr = 10 * torch.log10(1 / mse)
            self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            ssim_loss = ssim(pred, X)
            self.log('train/ssim', ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
            # visualize the reconstructed images and augmented inputs
            if batch_idx % 20 == 0:
                # Save augmented images for visualization
                self.save_augmented_images(X_original, X, batch_idx)
                # reshape pred to (N, C, H, W)
                deform_color = pred[0, :3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                undeform_color = pred[0, 3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()
                
                # log images in pytorch lightning
                self.logger.experiment.add_image('reconstruct/deform_color', deform_color, self.global_step)
                self.logger.experiment.add_image('reconstruct/undeform_color', undeform_color, self.global_step)

                # gt images
                gt_deform_color = X[0, :3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                gt_undeform_color = X[0, 3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()

                self.logger.experiment.add_image('gt/deform_color', gt_deform_color, self.global_step)
                self.logger.experiment.add_image('gt/undeform_color', gt_undeform_color, self.global_step)

                # spatial tokens
                spatial_num = mask.shape[1]
                # assume square image
                num = int(math.sqrt(spatial_num))
                mask_shape = H // num
                mask = mask.reshape(N, num, num)
                mask = mask.repeat_interleave(mask_shape, dim=1).repeat_interleave(mask_shape, dim=2).unsqueeze(1) # to (N, 1, H, W)
                mask = mask[0].detach().cpu().numpy()
                self.logger.experiment.add_image('gt/mask', mask, self.global_step)

        
        return {"loss": loss, "train_psnr": psnr, "train_ssim": ssim_loss}
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=10, T_max=self.num_epochs)

        return {"optimizer": optimizer}

        # return {"optimizer": optimizer, 
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "interval": "epoch"
        #         }}




SHARED_DATA = torch.empty(10000, 6, 256, 256, dtype=torch.float16).share_memory_()


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset_ratio', type=float, default=1.0)
    arg.add_argument('--dataset_dir', type=str, default="/arm/u/maestro/Desktop/DenseTact-Model/es4t/es4t/dataset_local/")
    arg.add_argument('--epochs', type=int, default=200)
    arg.add_argument('--config', type=str, default="configs/QHiera_disp.yaml")
    arg.add_argument('--gpus', type=int, default=1)
    
    arg.add_argument('--model', type=str, default="mae_hiera_large_256", help="Model Architecture, choose either hiera or vit")
    arg.add_argument('--batch_size', type=int, default=64)
    arg.add_argument('--num_workers', type=int, default=24)
    arg.add_argument('--mask_ratio', type=float, default=0.7)
    arg.add_argument('--exp_name', type=str, default="mae_random_crop_hiera")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--real_world', action='store_true')
    
    opt = arg.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        ])

    extra_samples_dirs = ['/arm/u/maestro/Desktop/DenseTact-Model/es1t/dataset_local/', 
                          '/arm/u/maestro/Desktop/DenseTact-Model/es2t/es2t/dataset_local/',
                          '/arm/u/maestro/Desktop/DenseTact-Model/es3t/es3t/dataset_local/']
    dataset = FullDataset(cfg, transform=transform, samples_dir=opt.dataset_dir, 
                          extra_samples_dirs=extra_samples_dirs,
                          is_real_world=opt.real_world, is_mae=True)
    print("Dataset total samples: {}".format(len(dataset)))


    full_dataset_length = len(dataset)
    
    # take only 10 percent of dataset for train and test
    dataset_length = int(opt.dataset_ratio * full_dataset_length)
    train_size = int(0.95 * dataset_length)
    test_size = dataset_length - train_size
    
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, full_dataset_length - dataset_length])
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    X, y = dataset[1000]
    deform_color = X[:3, :, :].detach().cpu().numpy()
    undeform_color = X[3:6, :, :].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(deform_color.transpose(1, 2, 0))
    ax[0].set_title("Deformed Image")
    ax[1].imshow(undeform_color.transpose(1, 2, 0))
    ax[1].set_title("Undeformed Image")
    plt.savefig("sample_image_123.png")
    plt.close()
    
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=12, 
                                 prefetch_factor=2)
    
    calibration_model = LightningDTModel(
        model_name=opt.model, 
        num_epochs=opt.epochs,
        total_steps=opt.epochs * len(train_dataset),
        mask_ratio=opt.mask_ratio,
        learning_rate=8e-4
    )


    logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='train/loss_epoch',
        dirpath=osp.join(opt.exp_name, 'checkpoints/'),
        filename='dt_model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )

    # log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    strategy = "ddp" if opt.gpus > 1 else "auto"

    # add callbacks
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=opt.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy=strategy, profiler="simple")
    

    trainer.fit(model=calibration_model, train_dataloaders=dataloader, ckpt_path=opt.ckpt_path)
