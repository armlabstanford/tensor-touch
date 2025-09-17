import os
import os.path as osp
import cv2
import torch.distributed as dist
import shutil
import numpy as np
import argparse
from typing import Dict
from weakref import proxy
from copy import deepcopy
import matplotlib as mpl
import wandb
import datetime
from termcolor import cprint

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch.nn.functional as F

from torchvision.transforms import functional as Fun

from torchvision import transforms
from process_data import FullDataset, FEAT_CHANNEL

from torchmetrics.functional import structural_similarity_index_measure

from torch.cuda.amp import autocast


import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from configs import get_cfg_defaults

from models import build_model, replace_LoRA, MonkeyPatchLoRALinear, HieraDPT
from util.loss_util import ssim
from util.scheduler_util import LinearWarmupCosineAnnealingLR

from models.dense import DecoderNHead

from tqdm import tqdm
from copy import deepcopy

def apply_colormap(depth_map, cmap='viridis'):  
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 0.00001)  # Normalize to [0, 1]
    colormap = plt.get_cmap(cmap)
    depth_map_colored = colormap(depth_map_normalized)[:, :, :3]  # Get RGB values, ignore alpha channel
    depth_map_colored = (depth_map_colored * 255).astype(np.uint8)  # Convert to 8-bit image
    return depth_map_colored

class LightningDTModel(L.LightningModule):
    OUTPUT_ORDER = ["depth", "stress_x", "stress_y", "stress_z"]
    
    def __init__(self, cfg):
        """ MAE Model for training on the DT dataset """
        super(LightningDTModel, self).__init__()
        self.cfg = cfg

        self.model = build_model(cfg)

        # if cfg.model.LoRA:
        #     self.model.replace_LoRA(cfg.model.LoRA_rank, cfg.model.LoRA_scale)
        if len(self.cfg.model.pretrained_model) > 0:
            print("Load pretrained model")
            # self.model.load_from_pretrained_model(self.cfg.model.pretrained_model, load_decoder=True) 
            self.model.load_from_pretrained_model(self.cfg.model.pretrained_model)   

            if cfg.model.name != "DenseNetV2":
                # todo: see if we need to freeze encoder
                self.model.freeze_encoder()
            
                # use LoRA finetune
                if cfg.model.LoRA:
                    self.model.replace_LoRA(self.cfg.model.LoRA_rank, self.cfg.model.LoRA_scale)
        
        if cfg.model.loss == "L1":
            self.criterion = nn.L1Loss()
        elif cfg.model.loss == "L2":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("Loss not implemented {}".format(cfg.model.loss))

        # get channel information
        total_output_channels = sum(self.cfg.model.out_chans)
        dataset_output_type = cfg.dataset.output_type
        expected_output_channels = sum([FEAT_CHANNEL[t] for t in dataset_output_type])
        
        assert total_output_channels == expected_output_channels, \
                    f"Output channels mismatch {total_output_channels} != {expected_output_channels}"

        self.output_names = []
        if cfg.dataset.contiguous_on_direction:
            # depth always comes first
            if "depth" in dataset_output_type:
                self.output_names.append("depth")

            # add the other channels
            for d in ["x", "y", "z"]:
                for t in dataset_output_type:
                    if t == "depth":
                        continue
                    else:
                        # extend the three channels
                        self.output_names.append(f"{t}_{d}")
        
        else:
            for t in dataset_output_type:
                if t == "depth":
                    self.output_names.append("depth")
                else:
                    for d in ["x", "y", "z"]:
                        self.output_names.append(f"{t}_{d}")
        
        # MSE for validation
        self.mse = nn.MSELoss()
        
        self.num_epochs = cfg.epochs
        self.total_steps = cfg.total_steps
        self.validation_stats = {}
        self.smooth_l1_loss = nn.SmoothL1Loss()
        # this should be set in the config
        self.beta = 1

        self.z_curr_mean = None
        self.z_curr_std = None

        # Color jitter parameters
        self.brightness = 0.25
        self.contrast = 0.25
        self.saturation = 0.25
        self.hue = 0.02

    def set_teacher_encoders(self, teacher_encoders_dict):
        self.teacher_encoders_dict = teacher_encoders_dict
        
        
    def set_student_encoders(self, student_encoders):
        self.student_encoders = student_encoders
    
    def on_train_start(self):
        # only load pre-trained model at the first epoch and if lora is not used
        if len(self.cfg.model.pretrained_model) > 0 and self.trainer.current_epoch == 0 and not self.cfg.model.LoRA:
            print("Load pretrained model")
            self.model.load_from_pretrained_model(self.cfg.model.pretrained_model)
        
        # set if student encoders are available
        if hasattr(self, "student_encoders"):
            for student_encoder in self.student_encoders:
                student_encoder.eval().to(next(self.model.parameters()).device)

        if hasattr(self, "teacher_encoders_dict"):
            for enc_key in self.teacher_encoders_dict:
                self.teacher_encoders_dict[enc_key].eval().to(next(self.model.parameters()).device)

    def on_train_epoch_start(self):
        """ 
        I decide to always freeze the encoder when loading from a pretrained model 
        If you want to finetune the encoder, you can load from a checkpoint and set the pretrained_model to ""
        """
        epoch = self.current_epoch
        if len(self.cfg.model.pretrained_model) > 0 and False:
            print("Freezing encoder when loading from pretrained model")
            self.model.freeze_encoder()
        else:
            # unfreeze the encoder
            print("Unfreezing encoder")
            self.model.unfreeze_encoder()
            # else:
            # self.model.unfreeze_encoder()

        # Calculate trainable parameters
        # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
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
                    deformed[i] = Fun.adjust_brightness(deformed[i], def_brightness[i].item())
                elif op == 'contrast':
                    deformed[i] = Fun.adjust_contrast(deformed[i], def_contrast[i].item())
                elif op == 'saturation':
                    deformed[i] = Fun.adjust_saturation(deformed[i], def_saturation[i].item())
                elif op == 'hue':
                    deformed[i] = Fun.adjust_hue(deformed[i], def_hue[i].item())

            # Apply to undeformed image with different parameters
            for idx in undef_order:
                op = ops[idx]
                if op == 'brightness':
                    undeformed[i] = Fun.adjust_brightness(undeformed[i], undef_brightness[i].item())
                elif op == 'contrast':
                    undeformed[i] = Fun.adjust_contrast(undeformed[i], undef_contrast[i].item())
                elif op == 'saturation':
                    undeformed[i] = Fun.adjust_saturation(undeformed[i], undef_saturation[i].item())
                elif op == 'hue':
                    undeformed[i] = Fun.adjust_hue(undeformed[i], undef_hue[i].item())
        
        # Clamp to valid range and concatenate
        deformed = torch.clamp(deformed, 0.0, 1.0)
        undeformed = torch.clamp(undeformed, 0.0, 1.0)
        
        return torch.cat([deformed, undeformed], dim=1)
       
            
    def training_step(self, batch, batch_idx):
        X, Y = batch 

        # Apply paired color jitter on GPU
        X = self.apply_paired_color_jitter(X)
        
        # # force Negative samples to be 0
        # Y = torch.where(torch.abs(Y) < self.cfg.metric.PN_thresh, torch.zeros_like(Y), Y)
        N, C, H, W = X.shape 

        if self.cfg.model.encoder == "densenet" or self.cfg.model.encoder == "resnet":
            pred, z = self.model(X)
        else:
            if self.cfg.model.hiera.return_encoder_output:
                pred, z = self.model(X) 
            else:
                pred = self.model(X)

        # get the output of each encoder
        z_loss = 0
        if hasattr(self, "student_encoders"):
            for student_encoder in self.student_encoders:
                student_z = student_encoder(X)
                
                # get a combination of smooth L1 loss and cosine similarity
                cosine_loss = (1 - F.cosine_similarity(student_z, z, dim=1).mean())
                smooth_l1_loss = self.smooth_l1_loss(student_z, z)
                
                # get l1 loss between student_pred and z
                z_loss += 1 * (self.beta * cosine_loss + (1 - self.beta) * smooth_l1_loss)
                
        if self.cfg.model.hiera.return_encoder_output:
            # go through the teachers and compute similarity
            z_loss = 0.0  # Initialize z_loss

            # Theia loss of cosine similarity and L1 loss
            # Make sure features are normalized!
            with autocast():
                for key in self.teacher_encoders_dict.keys():
                    teacher_model = self.teacher_encoders_dict[key]
                    # Run inference on teacher model
                    _, teacher_z = teacher_model(X)

                    # shape is torch.Size([8, 96, 64, 64])
                    # t_mean = teacher_z.mean(dim=[0, 2, 3], keepdim=True)  # Shape: [1, 96, 1, 1]
                    # t_std = teacher_z.std(dim=[0, 2, 3], keepdim=True)    # Shape: [1, 96, 1, 1]
                    # norm_teacher_z = (teacher_z - t_mean) / (t_std + 1e-5)

                    # # remove gradients from normal_teacher_z
                    # norm_teacher_z = norm_teacher_z.detach()
                             
                    # Compute losses
                    cosine_loss = (1 - F.cosine_similarity(teacher_z, z, dim=-1).mean())
                    smooth_l1_loss = self.smooth_l1_loss(teacher_z, z)
                    
                    # Accumulate loss
                    z_loss += 1 * (self.beta * cosine_loss + (1 - self.beta) * smooth_l1_loss)
                    
        
        z_loss *= 10

        # log the total z loss
        self.log(f'train/z_loss', z_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        loss = z_loss
        total_loss = z_loss
        predict_dict = self._process_prediction(pred)
        label_dict = self._process_prediction(Y)

        for name in self.cfg.dataset.output_type:
            scale = 1
            if 'disp' in name:
                weight = self.cfg.loss.disp_weight
                scale = self.cfg.scales.disp
            elif 'stress1' in name:
                weight = self.cfg.loss.stress_weight
                scale = self.cfg.scales.stress
            elif 'stress2' in name:
                weight = self.cfg.loss.stress2_weight
                scale = self.cfg.scales.stress2
            elif 'depth' in name:
                weight = self.cfg.loss.depth_weight
                scale = self.cfg.scales.depth
            elif 'cnorm' in name:
                weight = self.cfg.loss.cnorm_weight
                scale = self.cfg.scales.cnorm
            elif 'shear' in name:
                weight = self.cfg.loss.area_shear_weight
                scale = self.cfg.scales.area_shear
            
            if name == 'depth':
                # if the label_dict is below a very small threshold, have the loss be weighted small
                loss = 0
                for item in label_dict[name]:
                    # weight is based on the max value of the label
                    if torch.abs(item).max() > 0.01:
                        loss += 100 * self.criterion(predict_dict[name], label_dict[name] * self.cfg.scale)
                    else:
                        loss += 1 * self.criterion(predict_dict[name], label_dict[name] * self.cfg.scale)

            else:
                channels = [f"{name}_{d}" for d in ["x", "y", "z"]]
                pred_vec = torch.stack([predict_dict[c] for c in channels], dim=1)
                label_vec = torch.stack([label_dict[c] for c in channels], dim=1)

                loss = self.criterion(pred_vec, label_vec * scale)

            total_loss += weight * loss
            self.log(f'train/{name}/loss', weight * loss / scale, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.log('train/loss', total_loss / self.cfg.scale , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"loss": total_loss}

    def reset_validation_stats(self):
        """ resetting stats for val/test """
        self.validation_stats = {
            "mse": [],
            "cosine_sim_0": [],
            "cosine_sim_1": [],
            "cosine_sim_2": [],
            "l1_loss_0": [],
            "l1_loss_1": [],
            "l1_loss_2": [],
            "mean_ssim": [],
        }
        
        for name in self.cfg.dataset.output_type:
            self.validation_stats[name] = dict(NegativeNum=0, 
                                                FNNum=0, PosNum=0, num_images = 0, mse = 0,
                                                error_cureve=np.zeros(11), abs_error_curve=np.zeros(11))
    
    def on_validation_epoch_start(self):
        self.reset_validation_stats()

    @torch.no_grad()
    def compute_metric(self, prediction:torch.Tensor, 
                       label:torch.Tensor,
                       PN_thresh:float=1e-6,
                       TF_rel_error_rate:float=0.1,
                       TF_abs_error_thresh:float=1e-3):
        """ Compute the prediction metric 
        
        Args:
            prediction: (N, C, H, W)
            label: (N, C, H, W)
            
            PN_thresh: threshold for negative samples
            TF_rel_error_rate: threshold for relative error
            TF_abs_error_thresh: threshold for absolute error

        Returns:
            dict: dictionary of metrics
                NegativeNum: Number of negative samples
                FNNum: Number of false negative samples
                PosNum: Number of positive samples
                error_curve: Error curve (11)
                    [<0.1, <0.2, ..., <0.9, <1.0, >1.0]
                abs_error_curve: Absolute error curve
                    [<0.1mm, <0.2mm, ..., <0.9mm, <1.0mm, >1.0mm]
        """
        prediction_flatten = prediction.flatten()
        label_flatten = label.flatten()

        num_bins = self.cfg.metric.num_bins
        interval = TF_rel_error_rate / num_bins
        threshold_ratio = [k * interval for k in range(1, num_bins + 1)]
        # normally set TF_abs_error_thresh to 1e-3 (=1mm)
        abs_interval = TF_abs_error_thresh / num_bins
        threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]

        # Here, negative samples are the pixels with value less than 1e-6
        # positive samples are the pixels with value greater than 1e-6
        eps = PN_thresh

        # compute the negative pixels
        Neg_mask = torch.abs(label_flatten) < eps
        Pos_mask = torch.abs(label_flatten) >= eps

        # compute the false negative
        NegativeNum = torch.sum(Neg_mask).item()
        FNNum = torch.sum(Neg_mask * (torch.abs(prediction_flatten) >= eps)).item()

        # compute the error curve
        gt_pos_items = label_flatten[Pos_mask]
        pred_pos_items = prediction_flatten[Pos_mask]

        PosNum = torch.sum(Pos_mask).item()
        error_curve = []
        abs_error_curve = []
        if PosNum > 0:
            # compute the relative error
            rel_error = torch.abs(gt_pos_items - pred_pos_items) / torch.abs(gt_pos_items)

            # compute the threshold ratio
            for ratio in threshold_ratio:
                error_curve.append(torch.sum(rel_error < ratio).item())

            error_curve.append(torch.sum(rel_error > TF_rel_error_rate).item())
            error_curve = np.array(error_curve)

            # compute abs error
            abs_error = torch.abs(gt_pos_items - pred_pos_items)

            for thresh in threshold_abs:
                abs_error_curve.append(torch.sum(abs_error < thresh).item())

            abs_error_curve.append(torch.sum(abs_error > TF_abs_error_thresh).item())
            abs_error_curve = np.array(abs_error_curve)
        
        return dict(NegativeNum=NegativeNum, FNNum=FNNum, num_images=prediction.shape[0], mse = self.mse(prediction, label).item(),
                        PosNum=PosNum, error_cureve=error_curve, abs_error_curve=abs_error_curve)    
    
    
    @torch.no_grad()
    def compute_metric_vector(self, prediction:torch.Tensor, 
                       label:torch.Tensor,
                       PN_thresh:float=1e-6,
                       TF_rel_error_rate:float=0.1,
                       TF_abs_error_thresh:float=1e-3):
        """ Compute the prediction metric 
        
        Args:
            prediction: (N, 3, H, W)
            label: (N, 3, H, W)
            
            PN_thresh: threshold for negative samples
            TF_rel_error_rate: threshold for relative error
            TF_abs_error_thresh: threshold for absolute error

        Returns:
            dict: dictionary of metrics
                NegativeNum: Number of negative samples
                FNNum: Number of false negative samples
                PosNum: Number of positive samples
                error_curve: Error curve (11)
                    [<0.1, <0.2, ..., <0.9, <1.0, >1.0]
                abs_error_curve: Absolute error curve
                    [<0.1mm, <0.2mm, ..., <0.9mm, <1.0mm, >1.0mm]
        """
        prediction = prediction.permute(0, 2, 3, 1).reshape(-1, 3) # (N, 3)
        label = label.permute(0, 2, 3, 1).reshape(-1, 3) # (N, 3)

        num_bins = self.cfg.metric.num_bins
        interval = TF_rel_error_rate / num_bins
        threshold_ratio = [k * interval for k in range(1, num_bins + 1)]
        # normally set TF_abs_error_thresh to 1e-3 (=1mm)
        abs_interval = TF_abs_error_thresh / num_bins
        threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]

        # Here, negative samples are the pixels with value than 1e-6
        # positive samples are the pixels with value greater than 1e-6
        eps = PN_thresh

        # compute the negative pixels
        prediction_length = torch.norm(prediction, dim=1)
        label_length = torch.norm(label, dim=1)

        Neg_mask = label_length < eps
        Pos_mask = label_length >= eps

        # compute the false negative
        NegativeNum = torch.sum(Neg_mask).item()
        FNNum = torch.sum(Neg_mask * (prediction_length >= eps)).item()

        # compute the error curve
        gt_pos_items = prediction[Pos_mask]
        gt_pos_length = label_length[Pos_mask]

        pred_pos_items = label[Pos_mask]
        error_term = torch.norm(gt_pos_items - pred_pos_items, dim=1)

        PosNum = torch.sum(Pos_mask).item()
        error_curve = []
        abs_error_curve = []
        if PosNum > 0:
            # compute the relative error
            rel_error = error_term / gt_pos_length

            # compute the threshold ratio
            for ratio in threshold_ratio:
                error_curve.append(torch.sum(rel_error < ratio).item())

            error_curve.append(torch.sum(rel_error > TF_rel_error_rate).item())
            error_curve = np.array(error_curve)

            # compute abs error
            abs_error = error_term

            for thresh in threshold_abs:
                abs_error_curve.append(torch.sum(abs_error < thresh).item())

            abs_error_curve.append(torch.sum(abs_error > TF_abs_error_thresh).item())
            abs_error_curve = np.array(abs_error_curve)
        
        return dict(NegativeNum=NegativeNum, FNNum=FNNum, num_images=prediction.shape[0], mse = self.mse(prediction, label).item(),
                        PosNum=PosNum, error_cureve=error_curve, abs_error_curve=abs_error_curve)
    
    def update_metric(self, cumulative_metric_dict:dict, current_step_dict:dict):
        for key in ['NegativeNum', 'FNNum', 'PosNum','error_cureve', 'abs_error_curve']:
            cumulative_metric_dict[key] += current_step_dict[key] 
        
        # update mse
        fraction = cumulative_metric_dict["num_images"] / (cumulative_metric_dict["num_images"] + current_step_dict["num_images"])
        cumulative_metric_dict["mse"] = fraction * cumulative_metric_dict["mse"] + (1 - fraction) * current_step_dict["mse"]
        cumulative_metric_dict["num_images"] += current_step_dict["num_images"]
            
    def compute_cosine_similarity_per_group(self, pred, target):
        """
        Given pred and target of shape B x C x H x W, compute the cosine similarity for each vector in groups of 3,
        and then compute the mean cosine similarity per vector group.
        
        Args:
            pred: Tensor of shape B x C x H x W (predicted vector field)
            target: Tensor of shape B x C x H x W (target vector field)
        
        Returns:
            mean_cosine_sim_per_group: Tensor of shape (group_dim,) representing the mean cosine similarity per vector group.
        """
        assert pred.shape == target.shape, "pred and target must have the same shape"
        assert pred.shape[1] % 3 == 0, "C dimension must be divisible by 3"
        B, C, H, W = pred.shape
        group_dim = C // 3

        pred = pred.view(B, group_dim, 3, H, W)
        target = target.view(B, group_dim, 3, H, W)

        dot_product = torch.sum(pred * target, dim=2)
        pred_magnitude = torch.sqrt(torch.sum(pred ** 2, dim=2) + 1e-8)
        target_magnitude = torch.sqrt(torch.sum(target ** 2, dim=2) + 1e-8)

        cosine_similarity = dot_product / (pred_magnitude * target_magnitude)
        mean_cosine_sim_per_group = torch.mean(cosine_similarity, dim=(0, 2, 3))  # Mean per group across batch and spatial dimensions

        return mean_cosine_sim_per_group
    
    def compute_mean_l1_loss_per_group(self, pred, target):
        """
        Given pred and target of shape B x C x H x W, compute the L1 loss for each vector in groups of 3,
        and then compute the mean L1 loss per vector group.
        
        Args:
            pred: Tensor of shape B x C x H x W (predicted vector field)
            target: Tensor of shape B x C x H x W (target vector field)
        
        Returns:
            mean_l1_loss_per_group: Tensor of shape (group_dim,) representing the mean L1 loss per vector group.
        """
        assert pred.shape == target.shape, "pred and target must have the same shape"
        assert pred.shape[1] % 3 == 0, "C dimension must be divisible by 3"
        B, C, H, W = pred.shape
        group_dim = C // 3

        pred = pred.view(B, group_dim, 3, H, W)
        target = target.view(B, group_dim, 3, H, W)

        l1_loss = torch.abs(pred - target)
        vector_l1_loss = torch.sum(l1_loss, dim=2)  # Sum across vector components
        mean_l1_loss_per_group = torch.mean(vector_l1_loss, dim=(0, 2, 3))  # Mean per group across batch and spatial dimensions

        return mean_l1_loss_per_group
    
    def _process_prediction(self, prediction:torch.Tensor) -> dict:
        """ Process the prediction and label for computing the metric """
        output = dict()
        for idx in range(len(self.output_names)):
            output[self.output_names[idx]] = prediction[:, idx, :, :]
        
        return output
    
    def _run_val_test_step(self, batch, batch_idx, name="val"):
        # X - (N, C1, H, W); Y - (N, C2, H, W)
        X, Y = batch 

        X = self.apply_paired_color_jitter(X)

        N, C, H, W = X.shape 

        # if model is densenet, get the z value
        if self.cfg.model.encoder == "densenet" or self.cfg.model.encoder == "resnet":
            pred, z = self.model(X)
        else: 
            if self.cfg.model.hiera.return_encoder_output:
                pred, z = self.model(X) 
            else:
                pred = self.model(X)

        with torch.no_grad():
            # visualize the reconstructed images
            # visualize first y and pred
            if batch_idx % 10 == 0 and name == "test":
                for i in range(N):
                    X0 = X[i].detach().clone()
                    gt_def = X0[:3, :, :].clamp_(min=0., max=1.).detach().cpu().numpy()
                    gt_undef = X0[3:6, :, :].clamp_(min=0, max=1.).detach().cpu().numpy()
                    self.logger.experiment.add_image(f'{name}/gt/deform_color_{i}', gt_def, self.global_step)
                    self.logger.experiment.add_image(f'{name}/gt/undeform_color_{i}', gt_undef, self.global_step)
                    gt_depth = Y[i].detach().clone()
                    gt_depth = gt_depth.cpu().numpy()
                    pred_depth = pred[i].detach().clone()
                    pred_depth = pred_depth.cpu().numpy()
                    # plot both
                    fig, axes = plt.subplots(2, len(self.output_names), figsize=(20, 10))
                    pred_ax = axes[0]
                    
                    for name, ax, p in zip(self.output_names, pred_ax, pred_depth):
                        ax.imshow(p)
                        ax.set_title(name)
                        # Remove axes labels - choose one of these options:
                        ax.set_xticks([])  # Remove x-axis ticks
                        ax.set_yticks([])  # Remove y-axis ticks
                        # OR use this to remove all axes elements:
                        ax.axis('off')
                    
                    gt_ax = axes[1]
                    for name, ax, g in zip(self.output_names, gt_ax, gt_depth):
                        ax.imshow(g)
                        ax.set_title(name)
                        # Remove axes labels - choose one of these options:
                        ax.set_xticks([])  # Remove x-axis ticks
                        ax.set_yticks([])  # Remove y-axis ticks
                        # OR use this to remove all axes elements:
                        ax.axis('off')
                    
                    fig.savefig(osp.join(self.logger.save_dir, f"{name}_prediction_{batch_idx}_{i}.png"))
                    plt.close(fig)
        
        # unscale the prediction by each output scale
        pred_unit_scale = pred.clone()
        Y_unit_scale = Y.clone()
        for idx, name in enumerate(self.output_names):
            scale = 1
            if 'disp' in name:
                scale = self.cfg.scales.disp
                unit_scale = self.cfg.unit_scales.disp
            elif 'stress1' in name:
                scale = self.cfg.scales.stress
                unit_scale = self.cfg.unit_scales.stress
            elif 'stress2' in name:
                scale = self.cfg.scales.stress2
                unit_scale = self.cfg.unit_scales.stress2
            elif 'depth' in name:
                scale = self.cfg.scales.depth
                unit_scale = self.cfg.unit_scales.depth
            elif 'cnorm' in name:
                scale = self.cfg.scales.cnorm
                unit_scale = self.cfg.unit_scales.cnorm
            elif 'shear' in name:
                scale = self.cfg.scales.area_shear
                unit_scale = self.cfg.unit_scales.area_shear
            
            pred[:, idx, :, :] /= scale
            
            pred_unit_scale[:, idx, :, :] = pred[:, idx, :, :] / unit_scale
            Y_unit_scale[:, idx, :, :] = Y[:, idx, :, :] / unit_scale
        mse = self.mse(pred_unit_scale, Y_unit_scale)
        
            
        # compute ssim per channel
        ssim_per_channel = []
        for i in range(len(pred[0])):
            ssim_val = structural_similarity_index_measure(pred[:, i, :, :].unsqueeze(1), Y[:, i, :, :].unsqueeze(1))
            # get item_value
            ssim_per_channel.append(ssim_val.item())
            
        # get the mean ssim
        mean_ssim = np.mean(ssim_per_channel)
        
        self.validation_stats["mean_ssim"].append(mean_ssim)
        
        # compute lpips
        
        self.log(f'{name}/mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.validation_stats["mse"].append(mse.item())

        predict_dict, label_dict = self._process_prediction(pred_unit_scale), self._process_prediction(Y_unit_scale)
        
        # compute force metric
        for name in self.cfg.dataset.output_type:
            if name == 'depth':
                metric_dict = self.compute_metric(predict_dict[name] / (self.cfg.scale / 10), label_dict[name] / (self.cfg.scale / 10),
                                                  PN_thresh=self.cfg.metric.PN_thresh,
                                                  TF_rel_error_rate=self.cfg.metric.TF_rel_error_rate,
                                                  TF_abs_error_thresh=self.cfg.metric.TF_abs_error_thresh)
            else:
                channels = [f"{name}_{d}" for d in ["x", "y", "z"]]
                pred_vec = torch.stack([predict_dict[c] for c in channels], dim=1)
                label_vec = torch.stack([label_dict[c] for c in channels], dim=1)

                # compute vector metric
                metric_dict = self.compute_metric_vector(pred_vec, label_vec,
                                                        PN_thresh=self.cfg.metric.PN_thresh,
                                                        TF_rel_error_rate=self.cfg.metric.TF_rel_error_rate,
                                                        TF_abs_error_thresh=self.cfg.metric.TF_abs_error_thresh)
                
            self.update_metric(self.validation_stats[name], metric_dict)

    def validation_step(self, batch, batch_idx):
        self._run_val_test_step(batch, batch_idx, name="val")
    
    def summarize_metric(self, name="val"):
        avg_mse = np.array(self.validation_stats["mse"])
        avg_mse = np.mean(avg_mse)
        self.log(f'{name}/avg_mse', avg_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        psnr = 10 * np.log10(1.0 / avg_mse)
        self.log(f'{name}_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        avg_ssim = np.mean(self.validation_stats["mean_ssim"])
        avg_ssim = np.mean(avg_ssim)
        self.log(f'{name}/avg_ssim', avg_ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # compute the ssim for the outputs

        if dist.is_initialized():
            # summarize the results from multiple GPUs
            output = [None for _ in range(dist.get_world_size())]

            # gather the results from all GPUs
            dist.all_gather_object(output, self.validation_stats)

            if self.global_rank == 0:
                # sum to the validation stats
                for stats in output[1:]:
                    
                    for key, item in stats.items():
                        if isinstance(item, dict):
                            self.update_metric(self.validation_stats[key], item)
                        else:
                            self.validation_stats[key] += stats[key]
        
        # compute the metric in the rank = 0
        depth_AUC_abs = 0.
        if self.global_rank == 0:
            for item in self.cfg.dataset.output_type:
                # log mse
                mse = self.validation_stats[item]["mse"]
                self.log(f'{name}/{item}/mse', mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                # compute FNR
                FNR = self.validation_stats[item]["FNNum"] / self.validation_stats[item]["NegativeNum"]

                # here FPR is defined as the rel error > 100 %
                FPR_rel = self.validation_stats[item]["error_cureve"][-1] / self.validation_stats[item]["PosNum"]

                # here FPR is defined as the abs error > TF_abs_error_thresh
                FPR_abs = self.validation_stats[item]["abs_error_curve"][-1] / self.validation_stats[item]["PosNum"]

                # log
                self.log(f'{name}/{item}/FNR', FNR, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{name}/{item}/FPR_rel', FPR_rel, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                self.log(f'{name}/{item}/FPR_abs', FPR_abs, on_step=False, on_epoch=True, prog_bar=False, logger=True)

                # draw the curve
                error_curve = self.validation_stats[item]["error_cureve"][:-1] / self.validation_stats[item]["PosNum"]
                # compute the Area under curve
                AUC_rel = np.mean(error_curve)
                self.log(f'{name}/{item}/AUC_rel', AUC_rel, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
                abs_error_curve = self.validation_stats[item]["abs_error_curve"][:-1] / self.validation_stats[item]["PosNum"]
                AUC_abs = np.mean(abs_error_curve)
                self.log(f'{name}/{item}/AUC_abs', AUC_abs, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
                # select depth as final metric
                if item == "depth":
                    depth_AUC_abs = AUC_abs
                
                # for ckpt name
                # self.log('AUC', AUC_abs * 100, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
                # disable plotting when training in DDP
                if not dist.is_initialized():

                    num_bins = self.cfg.metric.num_bins
                    interval = self.cfg.metric.TF_rel_error_rate / num_bins
                    threshold_ratio = [k * interval for k in range(1, num_bins + 1)]

                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # set x-y limit
                    ax.set_xlim([0, threshold_ratio[-1]])
                    ax.set_ylim([0, 1])
                    # draw the plot of the error curve
                    ax.plot(threshold_ratio, error_curve, label="Error Curve")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Error Rate")
                    if isinstance(self.logger, TensorBoardLogger):
                        self.logger.experiment.add_figure(f"{name}/{item}/Rel Error Curve", fig, self.global_step)
                    # else:
                    #     fig.savefig(osp.join(self.logger.save_dir, f"{name}_{item}_error_curve_{self.global_step}.png"))
                    # fig.clf()

                    # draw the plot of the abs error curve
                    abs_interval = self.cfg.metric.TF_abs_error_thresh / num_bins
                    threshold_abs = [k * abs_interval for k in range(1, num_bins + 1)]
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    # set x-y limit
                    ax.set_xlim([0, threshold_abs[-1]])
                    ax.set_ylim([0, 1])
                    # draw the plot of the error curve
                    ax.plot(threshold_abs, abs_error_curve, label="Abs Error Curve")
                    ax.set_xlabel("Threshold")
                    ax.set_ylabel("Error Rate")
                    if isinstance(self.logger, TensorBoardLogger):
                        self.logger.experiment.add_figure(f"{name}/{item}/Abs Error Curve", fig, self.global_step)
                    # else:
                    #     fig.savefig(osp.join(self.logger.save_dir, f"{name}_{item}_abs_error_curve_{self.global_step}.png"))

        sync_data = depth_AUC_abs * 100 if self.global_rank == 0 else 0
        if dist.is_initialized():
            # boardcast to all GPUS
            output = [None for _ in range(dist.get_world_size())]

            # gather the results from all GPUs
            dist.all_gather_object(output, sync_data)
            
            # debug purpose
            print(f"collected AUC {output} rank {self.global_rank}") 

            AUC = sum(output)
            self.log('AUC', AUC, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            # normal case
            self.log('AUC', sync_data, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.summarize_metric(name="val")

    def on_test_epoch_start(self):
        self.reset_validation_stats()

    def test_step(self, batch, batch_idx):
        self._run_val_test_step(batch, batch_idx, name="test")
    
    def on_test_epoch_end(self):
        stats = self.summarize_metric(name="test")
        if stats is not None:
            printable = {key: value for key, value in stats.items() if key != "fig"}

            from pprint import pprint
            pprint(printable)

            stats["fig"].savefig(osp.join(self.logger.save_dir, "test_error_curve.png"))
            
    def configure_optimizers(self):
        
        if isinstance(self.model, HieraDPT):
            optimizer = self.model.configure_optimizers(self.cfg)
        else:
            if self.cfg.optimizer.name == "Adam":
                optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg.optimizer.lr)
            elif self.cfg.optimizer.name == "AdamW":
                optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg.optimizer.lr)

        opt_config = dict(optimizer = optimizer)
        
        T_max = self.total_steps // (2 * self.cfg.scheduler.cycle_k + 1)
        if cfg.scheduler.name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.optimizer.eta_min)
            opt_config["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
        elif cfg.scheduler.name == "linear_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps=cfg.scheduler.warmup, T_max=T_max, eta_min=cfg.optimizer.eta_min)
            opt_config["lr_scheduler"] = dict(scheduler=scheduler, interval="step")
        elif cfg.scheduler.name == "none":
            print("no scheduler")
            pass

        return opt_config

def construct_student_encoders(encoder_paths, calibration_model):
    student_encoders = []
    for encoder_path in encoder_paths:
        model_state = torch.load(encoder_path)
        
        encoder_state_dict = {k: v for k, v in model_state["state_dict"].items() if k.startswith("model.encoder.")}
        
        calibration_model_copy = deepcopy(calibration_model)
        
        calibration_model_state_dict = calibration_model_copy.state_dict()
        calibration_model_state_dict.update(encoder_state_dict)
        calibration_model_copy.load_state_dict(calibration_model_state_dict)
        
        # create copy of calibration model, delete the decoders, use z vector
        decoder_N_head_info = {'heads': 1, 'channels_per_head': 3}
        head = DecoderNHead(64, **decoder_N_head_info)
        calibration_model_copy.model.decoders[0].head = head
        
        del calibration_model_copy.model.decoders
        # move to cuda
        calibration_model_copy.model.to(device)
        student_encoders.append(calibration_model_copy.model)
    return student_encoders

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--gpus', type=int, default=4)
    arg.add_argument('--exp_name', type=str, default="exp/base")
    arg.add_argument('--ckpt_path', type=str, default=None)
    arg.add_argument('--config', type=str, default="configs/densenet_real_all.yaml")
    arg.add_argument('--logger', type=str, default="tensorboard")
    arg.add_argument('--dataset_dir', type=str, default="/arm/u/maestro/Desktop/DenseTact-Model/es4t/es4t/dataset_local/")
    arg.add_argument('--eval', action='store_true')
    arg.add_argument('--finetune', action='store_true')
    arg.add_argument('--match_features_from_encoders', action='store_true')
    arg.add_argument('--encoder_paths', nargs='+', type=str, help="List of encoder paths")
    arg.add_argument('--real_world', action='store_true')
    opt = arg.parse_args()    

    # load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config)
    # copy config file
    os.makedirs(opt.exp_name, exist_ok=True)
    
    if opt.config != osp.join(opt.exp_name, 'config.yaml'):
        shutil.copy(opt.config, osp.join(opt.exp_name, 'config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])

    X_transform = transforms.Compose([
        transforms.Resize((cfg.model.img_size, cfg.model.img_size), antialias=True),
    ])

    extra_samples_dirs = ['/arm/u/maestro/Desktop/DenseTact-Model/es1t/dataset_local/', 
                          '/arm/u/maestro/Desktop/DenseTact-Model/es2t/es2t/dataset_local/',
                          '/arm/u/maestro/Desktop/DenseTact-Model/es3t/es3t/dataset_local/']
    
    dataset = FullDataset(cfg, transform=transform, X_transform=X_transform,
                          extra_samples_dirs=extra_samples_dirs,
                          samples_dir=opt.dataset_dir, is_real_world=opt.real_world)

    print("Dataset total samples: {}".format(len(dataset)))
    full_dataset_length = len(dataset)
    # go through some samples in the dataset
    X, y = dataset[100]

    dataset_length = int(cfg.dataset_ratio * full_dataset_length)
    train_size = int(0.85 * dataset_length)
    test_size = dataset_length - train_size
    cfg.total_steps = train_size * cfg.epochs // (cfg.batch_size * opt.gpus)
    
    train_dataset, test_dataset, _ = random_split(dataset, 
                                                  [train_size, test_size, full_dataset_length - dataset_length],
                                                  generator=torch.Generator().manual_seed(cfg.seed))
    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                            pin_memory=True,
                            pin_memory_device="cuda" if torch.cuda.is_available() else "",
                            persistent_workers=True,
                            prefetch_factor=8)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=12)

    calibration_model = LightningDTModel(cfg)


    X, y = dataset[1000]
    deform_color = X[:3, :, :].detach().cpu().numpy()
    undeform_color = X[3:6, :, :].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(deform_color.transpose(1, 2, 0))
    ax[0].set_title("Deformed Image")
    ax[1].imshow(undeform_color.transpose(1, 2, 0))
    ax[1].set_title("Undeformed Image")
    plt.savefig("sample_image_matt.png")
    plt.close()
    

    # get date
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = "{}-{}".format(opt.exp_name.split("/")[-1], date)
   
    if opt.logger == "tensorboard":
        logger = TensorBoardLogger(osp.join(opt.exp_name, 'tb_logs/'), name="lightning_logs")
    if opt.logger == "wandb":
        logger = WandbLogger(project="DenseTact", name=name, config=cfg)
    
    # create callback to save checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_psnr',
        dirpath=osp.join(opt.exp_name, 'checkpoints/'),
        filename='dt_model-{epoch:02d}-{val_psnr:.2f}',
        save_top_k=3,
        verbose=True,
        save_last=True,
        mode='max',
    )

    # log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # add callbacks
    strategy = "ddp_find_unused_parameters_true" if opt.gpus < 1 else "auto"
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = L.Trainer(max_epochs=cfg.epochs, callbacks=callbacks, logger=logger,
                        accelerator="gpu", devices=opt.gpus, strategy=strategy,
                        gradient_clip_val=cfg.gradient_clip_val, gradient_clip_algorithm=cfg.gradient_clip_algorithm)


    # load the hiera encoders if desired
    teacher_models = {}

    if cfg.model.hiera.return_encoder_output:
        for output in tqdm(cfg.dataset.output_type, desc="Loading teacher models", unit="item"):
            # get the encoder path
            if output == "cnorm":
                encoder_path = cfg.teacher_encoders.cnorm_path
            elif output == "disp":
                encoder_path = cfg.teacher_encoders.disp_path
            elif output == "shear":
                encoder_path = cfg.teacher_encoders.area_shear_path
            elif output == "stress1":
                encoder_path = cfg.teacher_encoders.stress_path
            elif output == "stress2":
                encoder_path = cfg.teacher_encoders.stress2_path

            if len(encoder_path) == 0:
                # no need to supervise on this model
                continue

            # create copy of the cfg
            cfg_teacher = deepcopy(cfg)
            cfg_teacher.model.out_chans = [3]

            # load only the encoder
            cprint(f"Loading {output} model", "green")
            teacher_model = build_model(cfg_teacher)
            if cfg.model.LoRA:
                teacher_model.replace_LoRA(cfg.model.LoRA_rank, cfg.model.LoRA_scale)
            teacher_model.load_from_pretrained_model(encoder_path, load_decoder=False)   

            # we can delete the decoder head
            teacher_models[output] = teacher_model

        # add teacher encoders!
        calibration_model.set_teacher_encoders(teacher_models)

    # only load states for finetuning and model is densenet
    if opt.finetune and "densenet" in cfg.model.name:
        model_state = torch.load(opt.ckpt_path)
        encoder_state_dict = {k: v for k, v in model_state["state_dict"].items() if k.startswith("model.encoder.")}
        
        calibration_model_state_dict = calibration_model.state_dict()
        calibration_model_state_dict.update(encoder_state_dict)
        calibration_model.load_state_dict(calibration_model_state_dict)
        
        # get out_channels from the config
        heads = len(cfg.model.out_chans)
        decoder_N_head_info = {'heads': heads, 'channels_per_head': 3}
        head = DecoderNHead(64, **decoder_N_head_info)
        # delete the old head
        del calibration_model.model.decoders[0].head
        calibration_model.model.decoders[0].head = head
        opt.ckpt_path = None
    
    if opt.match_features_from_encoders:
        # TODO: update this for densetact
        student_encoders = construct_student_encoders(opt.encoder_paths, calibration_model)
        calibration_model.set_student_encoders(student_encoders)
    if opt.eval:
        trainer.test(model=calibration_model, dataloaders=test_dataloader, ckpt_path=opt.ckpt_path)
    else:
        trainer.fit(model=calibration_model, train_dataloaders=dataloader, 
                     val_dataloaders=test_dataloader)
        
        trainer.test(model=calibration_model, dataloaders=test_dataloader)