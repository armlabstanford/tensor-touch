# 2024 ARMLab - DenseTact Calibration

import time
import json
import os
import pickle
import argparse

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms

import pandas as pd

from tqdm import tqdm

import OpenEXR
import Imath

import h5py

from multiprocessing import Pool, Manager

import random
import torchvision.transforms.functional as F

def sample_color_jitter_params(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02):
    b = random.uniform(max(0, 1 - brightness), 1 + brightness)
    c = random.uniform(max(0, 1 - contrast), 1 + contrast)
    s = random.uniform(max(0, 1 - saturation), 1 + saturation)
    h = random.uniform(-hue, hue)
    order = [0, 1, 2, 3]
    random.shuffle(order)
    return b, c, s, h, order

def apply_color_jitter_tensor(img, params):
    b, c, s, h, order = params
    for o in order:
        if o == 0:
            img = F.adjust_brightness(img, b)
        elif o == 1:
            img = F.adjust_contrast(img, c)
        elif o == 2:
            img = F.adjust_saturation(img, s)
        elif o == 3:
            img = F.adjust_hue(img, h)
    return img


DEPTH_CONSTANT = 17.437069

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

NAME_TO_RGB_MAPPINGS = {
    'CNORM': ('CNORMF-CNORMF1', 'CNORMF-CNORMF2', 'CNORMF-CNORMF3'),
    'S11': ('S-S11', 'S-S22', 'S-S33'),
    'S12': ('S-S12', 'S-S13', 'S-S23'),
    'UU1': ('U-U1', 'U-U2', 'U-U3'),
    'CNAREA': ('CNAREA', 'CSHEAR1', 'CSHEAR2'),
}

FEAT_CHANNEL = {
    "depth": 1, 
    "disp": 3,
    "stress1": 3,
    "stress2": 3,
    "shear": 3,
    "cnorm": 3
}

MAX = 0
MIN = 0

manager = Manager()
sample_id = manager.Value('i', 0)
sample_id_lock = manager.Lock()


DISP_MEANS = [-0.17508025467395782, -0.8888664841651917, -0.12558214366436005]
DISP_STDS = [0.5062088370323181, 1.4978240728378296, 0.44729936122894287]

FORCE_MEANS = [-0.0026451176963746548, -0.010370887815952301, -0.002843874040991068]
FORCE_STDS =  [0.0008145869709551334, 0.0002247917652130127, 0.00009318174794316292]

STRESS1_MEANS =  [-0.017115609720349312, -0.02853190153837204, -0.01735331304371357]
STRESS1_STDS = [0.03661240264773369, 0.05918154492974281, 0.03791118785738945]


def normalize_item(item, channel_means, channel_stds):
    """
    Normalize a single item with shape (H, W, C).
    """
    # normalized_item = (item - channel_means[None, None, :]) / channel_stds[None, None, :]
    normalized_item = (item - channel_means[None, None, :])

    return normalized_item

def write_data(file_path, data, is_X = True, bounds_dict=None):
    global MAX
    if is_X:
        deformed_img_norm, undeformed_img_norm, image_diff = data
        
        # save all to png
        cv2.imwrite(f'{file_path}/deformed.png', deformed_img_norm * 255)
        cv2.imwrite(f'{file_path}/undeformed.png', undeformed_img_norm * 255)
        image_diff = (image_diff).astype(np.uint16)
        cv2.imwrite(f'{file_path}/diff.png', image_diff)
        
    else:
        cnorm_img, stress1_img, stress2_img, displacement_img, area_shear_img = data
        
        # save each to png
        cv2.imwrite(f'{file_path}/cnorm.png', cnorm_img)
        cv2.imwrite(f'{file_path}/stress1.png', stress1_img)
        cv2.imwrite(f'{file_path}/stress2.png', stress2_img)
        cv2.imwrite(f'{file_path}/displacement.png', displacement_img)
        cv2.imwrite(f'{file_path}/area_shear.png', area_shear_img)
        
        # save bounds to json
        with open(f'{file_path}/bounds.json', 'w') as f:
            json.dump(bounds_dict, f, indent=4)
            
            
class CombinedMAEDataset(Dataset):
    """class for a combined dataset for Hiera/ViT pretraining."""
    def __init__(self, dir1, dir2, transform=None):
        """
        Args:
            dir1 (str): Path to the first image directory.
            dir2 (str): Path to the second image directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        
        # get list of directories in each dir
        self.dirs1 = [os.path.join(dir1, d) for d in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, d)) and 'X' in d]
        self.dirs2 = [os.path.join(dir2, d) for d in os.listdir(dir2) if os.path.isdir(os.path.join(dir2, d)) and 'X' in d]

        # Combine the lists
        self.dir_paths = self.dirs1 + self.dirs2
        
        
    def __len__(self):
        """length of the dataset"""
        return len(self.dir_paths)

    def __getitem__(self, idx):
        """gets an image for pretraining."""
        # Get the image path
        dir_path = self.dir_paths[idx]
        img_path = f'{dir_path}/deformed.png'
        
        # Open the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image
        
class FullDataset(Dataset):
    OUTPUT_TYPES = ['depth', 'stress1', 'stress2', 'disp', 'shear', 'cnorm']
    def __init__(self,  opt, transform=None, X_transform=None,
                 samples_dir='../Documents/Dataset/sim_dataset', 
                 root_dir=None,
                 extra_samples_dirs=['../Documents/Dataset/sim_dataset'],
                 is_real_world=False, is_mae=False):
        """
        Dataset for DenseTact Calibration task

        Args:
            transform (torchvision.transforms.Compose): Transform to apply to the data
            samples_dir (str): path to the processed dataset
            output_type (str): Type of output to get from the dataset
            root_dir (str) Optional: Root directory of the original dataset, 
                    This argument is only needed when pre-processing the data
            is_real_world (bool): Flag to indicate if it is real world data
        """
        self.samples_dir = samples_dir
        self.root_dir = root_dir
        self.transform = transform 
        self.X_transform = X_transform 


        self.output_type = opt.dataset.output_type
        self.normalization = opt.dataset.normalization
        self.extra_samples_dirs = extra_samples_dirs
        self.is_mae = is_mae

        self.folder_with_idx = []
        self.folder_with_idx.append((0, self.samples_dir))
        length = sum(os.path.isdir(os.path.join(self.samples_dir, name)) for name in os.listdir(self.samples_dir)) / 2
        for dir in extra_samples_dirs:
            self.folder_with_idx.append((int(length), dir))
            length += (sum(os.path.isdir(os.path.join(dir, name)) for name in os.listdir(dir)) / 2)

        self.color_jitter = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.02
        )

        for t in opt.dataset.output_type:
            assert t in self.OUTPUT_TYPES, f"Output type must be one of {self.OUTPUT_TYPES}, \
                                                Input was {t}"
        self.is_real_world = is_real_world
        self.opt = opt
        
        if self.is_real_world:
            # check if real_blender_info.json exists
            if os.path.exists('output.json'):
                self.blender_info = self.read_blender_info_json('output.json')
            else:
                # read from output.json
                self.blender_info = self.read_blender_info_json('real_blender_info.json')
        else:
            self.blender_info = self.read_blender_info_json('blender_info.json')
            
        if not self.is_real_world:
            # get output mask to only worry about DT
            self.output_mask = self.get_output_mask()
        else:
            # 256 by 256 mask
            self.output_mask = np.ones((512, 512))

        self.output_mask = self.get_output_mask()
        self.min = np.inf
        self.max = -np.inf

        self.folders = sorted(self.folder_with_idx)[::-1]

        self.cache = []

        # self.cached = False
        # self.create_cache()
        # self.cached = True


    def construct_dataset(self):
        self._construct_dataset_from_json()
            
    def create_cache(self):
        self.cache = []
        for i in tqdm(range(15000), desc="Creating cache"):
            res = self[i]
            self.cache.append(res)
            # add idx to cache
            
    def read_blender_info_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    def read_csv(self, csv_file):
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        return df
    
    def get_data_from_frame_and_name(self, root, frame, name, data):
        frame = int(frame)
        img_path = f'{root}/img_{name}_{frame}.png'
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # remove duplicates from data, data is a dataframe
        data = data.drop_duplicates(subset=['field'])
        
        field1 = NAME_TO_RGB_MAPPINGS[name][0]
        field2 = NAME_TO_RGB_MAPPINGS[name][1]
        field3 = NAME_TO_RGB_MAPPINGS[name][2]
        
        data1 = data[data['field'] == field1]
        data2 = data[data['field'] == field2]
        data3 = data[data['field'] == field3]
        
        # get max and min values
        min_val_r = data1['min'].values[0]
        max_val_r = data1['max'].values[0]
        
        min_val_g = data2['min'].values[0]
        max_val_g = data2['max'].values[0]
        
        min_val_b = data3['min'].values[0]
        max_val_b = data3['max'].values[0]
        
        # get values from image, which are linearly scaled
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        
        bounds_dict = {
            name: {
                'max_val_r': max_val_r,
                'min_val_r': min_val_r,
                'max_val_g': max_val_g,
                'min_val_g': min_val_g,
                'max_val_b': max_val_b,
                'min_val_b': min_val_b,
            }
        }
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # scale values
        channel1 = ((img_r / 255.0) * (max_val_r - min_val_r)) + min_val_r
        channel2 = ((img_g / 255.0) * (max_val_g - min_val_g)) + min_val_g
        channel3 = ((img_b / 255.0) * (max_val_b - min_val_b)) + min_val_b
        
        # combine channels into one image
        combined = np.stack([channel1, channel2, channel3], axis=2)
        return combined, img, bounds_dict
    
    def _construct_dataset_from_json(self):
        global MAX
        global MIN
        total_samples = 0
        for item in tqdm(self.blender_info, desc="Processing Blender Info"):
            # go through each sequence of presses
            path = item['path']
            # get last two directories
            path_split = path.split('/')
            configuration_dir = path_split[-2]
            folder_name = path_split[-1]
            combined = os.path.join(self.root_dir, configuration_dir, folder_name)
            files = os.listdir(combined)
            
            # files with csv
            csv_files = [f for f in files if f.endswith('.csv')]
            
            # undeformed_img_path = os.path.join(combined, f'img_0.png')
            # this is the undeformed image
            undeformed_img_path = os.path.join(combined, f'img_1.png')
            
            undeformed_img = self.read_image(undeformed_img_path)
            if self.is_real_world:
                undeformed_img = crop_image(undeformed_img, 165, 115, 25, 25)
            else:
                undeformed_img = center_crop(undeformed_img, 50)
            
            if len(csv_files) == 1:
                df = self.read_csv(os.path.join(combined, csv_files[0]))
                # remove cases where field is a string with length of 0
                df = df[df['field'].str.len() > 0]
                # remove rows where frame is na or inf or empty 
                df = df[~df['frame'].isna()]
                df = df[~df['frame'].isin([np.inf, -np.inf])]
                
                print(f"Processing {combined}")
                df['frame'] = df['frame'].astype(int)
                grouped = df.groupby('frame')
                
                pool = Pool(processes=24)
                
                tasks = [(group, frame, combined, undeformed_img, self.transform) for frame, group in grouped]
                for _ in tqdm(pool.imap_unordered(self.process_group, tasks), total=len(tasks), desc="Processing files in one press"):
                    pass

                pool.close()
                pool.join()
            elif self.output_type == 'depth':
                # no csv files and depth. 
                print(f"Processing {combined}")
                pool = Pool(processes=24)
                
                png_files = [f for f in files if f.endswith('.png')]
                
                tasks = [(combined, undeformed_img, self.transform) for _ in range(1)]
                
                for _ in tqdm(pool.imap_unordered(self.process_depth_group, tasks), total=len(tasks), desc="Processing files in one press"):
                    pass
                
    def process_depth_group(self, args):
        pass
    
    def process_group(self, args):
        global sample_id, sample_id_lock
        
        # try:
        combined_dict = {}
        
        group, frame, combined, undeformed_img, transform  = args

        group['field'] = group['field'].str.replace(' ', '')
        
        cnorm_data = group[(group['field'] == 'CNORMF-CNORMF1') | 
                        (group['field'] == 'CNORMF-CNORMF2') | 
                        (group['field'] == 'CNORMF-CNORMF3')]

        cnorm_output, cnorm_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'CNORM', cnorm_data)
        combined_dict.update(bounds)

        stress_data1 = group[(group['field'] == 'S-S11') | 
                            (group['field'] == 'S-S22') | 
                            (group['field'] == 'S-S33')]
        stress_data1_output, stress1_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'S11', stress_data1)
        combined_dict.update(bounds)

        stress_data2 = group[(group['field'] == 'S-S12') |
                            (group['field'] == 'S-S13') | 
                            (group['field'] == 'S-S23')]
        stress_data2_output, stress2_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'S12', stress_data2)
        combined_dict.update(bounds)

        displacement_data = group[(group['field'] == 'U-U1') | 
                        (group['field'] == 'U-U2') | 
                        (group['field'] == 'U-U3')]
        displacement_data_output, displacement_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'UU1', displacement_data)
        combined_dict.update(bounds)

        area_shear_data = group[(group['field'] == 'CNAREA') | 
                                (group['field'] == 'CSHEAR1') | 
                                (group['field'] == 'CSHEAR2')]
        area_shear_data_output, area_shear_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'CNAREA', area_shear_data)
        combined_dict.update(bounds)

        X  = self.construct_rgb(undeformed_img, frame + 1, combined)
        
        y = (cnorm_img, stress1_img, stress2_img, displacement_img, area_shear_img)
        
        with sample_id_lock:
            sample_id.value += 1
            current_sample_id = sample_id.value
            
        os.makedirs(f'{self.samples_dir}/X{current_sample_id}', exist_ok=True)
        os.makedirs(f'{self.samples_dir}/y{current_sample_id}', exist_ok=True)

        write_data(f'{self.samples_dir}/X{current_sample_id}', X, is_X = True)
        write_data(f'{self.samples_dir}/y{current_sample_id}', y, is_X = False, bounds_dict=combined_dict)
        # except Exception as e: 
        #     print("Skipping because of error...", e)
        
    def construct_rgb(self, undeformed_img, frame, combined):
        """
        Construct RGB inputs and depth output.

        Args:
            undeformed_img (_type_): _description_
            undeformed_depth (_type_): _description_
            frame (_type_): _description_
            combined (_type_): _description_

        Returns:
            _type_: _description_
        """
        deform_img_path = os.path.join(combined, f'img_{frame}.png')
        deformed_img = self.read_image(deform_img_path)
        if self.is_real_world:
            deformed_img = crop_image(deformed_img, 165, 115, 25, 25)
        
        else:
            deformed_img = center_crop(deformed_img, 50)
        # undeformed_img = center_crop(undeformed_img, 90)
        
        # resize images to 512 by 512
        deformed_img = cv2.resize(deformed_img, (512, 512))
        undeformed_img = cv2.resize(undeformed_img, (512, 512))
        
        # convert both to rgb
        deformed_img = cv2.cvtColor(deformed_img, cv2.COLOR_BGR2RGB)
        undeformed_img = cv2.cvtColor(undeformed_img, cv2.COLOR_BGR2RGB)

            
        hsv_img1 = cv2.cvtColor(deformed_img, cv2.COLOR_RGB2HSV)
        hsv_img2 = cv2.cvtColor(undeformed_img, cv2.COLOR_RGB2HSV)

        deformed_img_norm = deformed_img / 255.0
        undeformed_img_norm = undeformed_img / 255.0
        image_diff = cv2.subtract(hsv_img1[:,:,2], hsv_img2[:,:,2])
        
        X = (deformed_img_norm, undeformed_img_norm, image_diff)
        
        return X
        
    def __len__(self):
        """length of the dataset

        Returns:
            (int): length of the dataset (number of samples)
        """
        # length of files in the data folder
        
        length = sum(os.path.isdir(os.path.join(self.samples_dir, name)) for name in os.listdir(self.samples_dir)) / 2
        
        for dir in self.extra_samples_dirs:
            # add to the length
            length = int(length)
            length += (sum(os.path.isdir(os.path.join(dir, name)) for name in os.listdir(dir)) / 2)
        # augment data with rotations by 90 degrees
        self.length = int(length)
        
        return int(length)
    
    def read_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_output_mask(self):
        # output mask is based on the second value in the dataset
        # if output_mask.png does not exist, create it
        if os.path.exists('output_mask.png'):
            self.output_mask = cv2.imread('output_mask.png', cv2.IMREAD_ANYDEPTH)
            self.output_mask = self.output_mask / 255.0
            return self.output_mask
        
        cnorm_img = cv2.imread(f'{self.samples_dir}/y111/cnorm.png')
        cnorm_img = cv2.cvtColor(cnorm_img, cv2.COLOR_BGR2RGB)
        
        stress1_img = cv2.imread(f'{self.samples_dir}/y111/stress1.png')
        stress1_img = cv2.cvtColor(stress1_img, cv2.COLOR_BGR2RGB)
        
        diff = stress1_img - cnorm_img
        mask = np.all(diff == [0, 0, 0], axis=-1).astype(np.uint8)
        # opposite mask
        mask = 1 - mask
        
        # save mask as png
        cv2.imwrite(f'output_mask.png', mask * 255)
        
        return mask
        
    def __getitem__(self, idx):
        """Get item in dataset. Will get either depth or whole output suite"""

        # if self.cached and idx < len(self.cache):
        #     # if cached, return from cache
        #     return self.cache[idx]

        # based on sample idx, compute which folder to look in
        for folder_item in self.folders:
            folder_start_idx = folder_item[0]
            folder_name = folder_item[1]

            if idx >= folder_start_idx:
                # got the folder, get the sample num in the folder now
                sample_num = idx - folder_start_idx
                samples_dir = folder_name
                break

        sample_num = sample_num + 1
        
        # read deformed and undeformed images
        deformed_img_norm = cv2.imread(f'{samples_dir}/X{sample_num}/deformed.png')
        undeformed_img_norm = cv2.imread(f'{samples_dir}/X{sample_num}/undeformed.png')

        deformed = torch.from_numpy(cv2.cvtColor(deformed_img_norm, cv2.COLOR_BGR2RGB)).float() / 255.0
        undeformed = torch.from_numpy(cv2.cvtColor(undeformed_img_norm, cv2.COLOR_BGR2RGB)).float() / 255.0
        deformed = deformed.permute(2,0,1)
        undeformed = undeformed.permute(2,0,1)

        data_pack = []
        
        # return data to avoid extra file reads.
        if self.is_mae:
            deformed = self.transform(deformed).float()  # if transform accepts tensors
            undeformed = self.transform(undeformed).float()  # if transform accepts tensors

            X = torch.cat([deformed, undeformed], dim=0)  # shape (6,H,W)
            y = torch.tensor([0], dtype=torch.long)
            return X, y
        
        # load bounds json only if it exists
        bounds_exists = os.path.exists(f'{samples_dir}/y{sample_num}/bounds.json')
        if bounds_exists:
            with open(f'{samples_dir}/y{sample_num}/bounds.json') as f:
                bounds = json.load(f)

        for t in self.output_type:
           
            if t == 'depth':
                # process depth
                relative_depth = cv2.imread(f'{samples_dir}/y{sample_num}/depth.png', cv2.IMREAD_ANYDEPTH)
                relative_depth = relative_depth / 10000.0
                relative_depth = relative_depth - 1 
                y = relative_depth
                y = y[:,:,np.newaxis]
                data_pack.append(y)

            elif t == 'cnorm':
                cnorm_img = cv2.imread(f'{samples_dir}/y{sample_num}/cnforce_local.png')
                cnorm_img = cv2.cvtColor(cnorm_img, cv2.COLOR_BGR2RGB)
                x1 = ((cnorm_img[:,:,0] / 255.0) * (bounds['cnforce_local']['max_val_r'] - bounds['cnforce_local']['min_val_r'])) + bounds['cnforce_local']['min_val_r']
                x2 = ((cnorm_img[:,:,1] / 255.0) * (bounds['cnforce_local']['max_val_g'] - bounds['cnforce_local']['min_val_g'])) + bounds['cnforce_local']['min_val_g']
                x3 = ((cnorm_img[:,:,2] / 255.0) * (bounds['cnforce_local']['max_val_b'] - bounds['cnforce_local']['min_val_b'])) + bounds['cnforce_local']['min_val_b']
                cnorm = np.stack([x1, x2, x3], axis=2)

                # apply self.mask to cnorm
                # cnorm = cnorm * self.output_mask[:,:,np.newaxis]
                if self.normalization:
                    cnorm = normalize_item(cnorm, np.array(FORCE_MEANS), np.array(FORCE_STDS))

                data_pack.append(cnorm)
        
            elif t == 'stress1':
                # process stress1
                stress1_img = cv2.imread(f'{samples_dir}/y{sample_num}/nforce_local.png')
                stress1_img = cv2.cvtColor(stress1_img, cv2.COLOR_BGR2RGB)
                x1 = ((stress1_img[:,:,0] / 255.0) * (bounds['nforce_local']['max_val_r'] - bounds['nforce_local']['min_val_r'])) + bounds['nforce_local']['min_val_r']
                x2 = ((stress1_img[:,:,1] / 255.0) * (bounds['nforce_local']['max_val_g'] - bounds['nforce_local']['min_val_g'])) + bounds['nforce_local']['min_val_g']
                x3 = ((stress1_img[:,:,2] / 255.0) * (bounds['nforce_local']['max_val_b'] - bounds['nforce_local']['min_val_b'])) + bounds['nforce_local']['min_val_b']
                stress1 = np.stack([x1, x2, x3], axis=2)
                
                # perform normalization if desired
                # if self.normalization:
                #     stress1 = normalize_item(stress1, np.array(DISP_MEANS), np.array(DISP_STDS))
                
                data_pack.append(stress1)
            
            elif t == 'stress2':
                # process stress2
                stress2_img = cv2.imread(f'{samples_dir}/y{sample_num}/sforce_local.png')
                stress2_img = cv2.cvtColor(stress2_img, cv2.COLOR_BGR2RGB)
                x1 = ((stress2_img[:,:,0] / 255.0) * (bounds['sforce_local']['max_val_r'] - bounds['sforce_local']['min_val_r'])) + bounds['sforce_local']['min_val_r']
                x2 = ((stress2_img[:,:,1] / 255.0) * (bounds['sforce_local']['max_val_g'] - bounds['sforce_local']['min_val_g'])) + bounds['sforce_local']['min_val_g']
                x3 = ((stress2_img[:,:,2] / 255.0) * (bounds['sforce_local']['max_val_b'] - bounds['sforce_local']['min_val_b'])) + bounds['sforce_local']['min_val_b']
                stress2 = np.stack([x1, x2, x3], axis=2)
                # stress2 = stress2 * self.output_mask[:, :, np.newaxis]
                data_pack.append(stress2)

            elif t == 'disp':
                # process displacement
                # displacement_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/displacement.png')
                displacement_img = cv2.imread(f'{samples_dir}/y{sample_num}/disp_local.png')
                displacement_img = cv2.cvtColor(displacement_img, cv2.COLOR_BGR2RGB)
                # x1 = ((displacement_img[:,:,0] / 255.0) * (bounds['UU1']['max_val_r'] - bounds['UU1']['min_val_r'])) + bounds['UU1']['min_val_r']
                # x2 = ((displacement_img[:,:,1] / 255.0) * (bounds['UU1']['max_val_g'] - bounds['UU1']['min_val_g'])) + bounds['UU1']['min_val_g']
                # x3 = ((displacement_img[:,:,2] / 255.0) * (bounds['UU1']['max_val_b'] - bounds['UU1']['min_val_b'])) + bounds['UU1']['min_val_b']
                x1 = ((displacement_img[:,:,0] / 255.0) * (bounds['disp_local']['max_val_r'] - bounds['disp_local']['min_val_r'])) + bounds['disp_local']['min_val_r']
                x2 = ((displacement_img[:,:,1] / 255.0) * (bounds['disp_local']['max_val_g'] - bounds['disp_local']['min_val_g'])) + bounds['disp_local']['min_val_g']
                x3 = ((displacement_img[:,:,2] / 255.0) * (bounds['disp_local']['max_val_b'] - bounds['disp_local']['min_val_b'])) + bounds['disp_local']['min_val_b']
                displacement = np.stack([x1, x2, x3], axis=2)

                if self.normalization:
                    displacement = normalize_item(displacement, np.array(DISP_MEANS), np.array(DISP_STDS))

                data_pack.append(displacement)

            elif t == "shear":
                # process area shear
                area_shear_img = cv2.imread(f'{samples_dir}/y{sample_num}/csforce_local.png')
                area_shear_img = cv2.cvtColor(area_shear_img, cv2.COLOR_BGR2RGB)
                x1 = ((area_shear_img[:,:,0] / 255.0) * (bounds['csforce_local']['max_val_r'] - bounds['csforce_local']['min_val_r'])) + bounds['csforce_local']['min_val_r']
                x2 = ((area_shear_img[:,:,1] / 255.0) * (bounds['csforce_local']['max_val_g'] - bounds['csforce_local']['min_val_g'])) + bounds['csforce_local']['min_val_g']
                x3 = ((area_shear_img[:,:,2] / 255.0) * (bounds['csforce_local']['max_val_b'] - bounds['csforce_local']['min_val_b'])) + bounds['csforce_local']['min_val_b']
                area_shear = np.stack([x1, x2, x3], axis=2)
                data_pack.append(area_shear)

        if len(data_pack) > 0:
            y = np.concatenate(data_pack, axis=2) # (H, W, C)
            H, W, _ = y.shape

            if self.opt is not None and self.opt.dataset.contiguous_on_direction:
                if "depth" in self.output_type:
                    depth = y[:, :, [0]]
                    directions = y[:, :, 1:]
                else:
                    depth = None
                    directions = y

                directions = directions.reshape(H, W, -1, 3)
                directions = directions.transpose(0, 1, 3, 2) 
                directions = directions.reshape(H, W, -1) #  contiguous on direction
                y = np.concatenate([f for f in [depth, directions] if f is not None], axis=2)
            
            # apply transform
            y = self.transform(y).float()  
            
            resized_mask = cv2.resize(self.output_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
            y = y  * resized_mask

        else:
            y = [0]

        deformed = self.X_transform(deformed).float()  # if transform accepts tensors
        undeformed = self.X_transform(undeformed).float()  # if transform accepts tensors
        X = torch.cat([deformed, undeformed], dim=0)  # shape (6,H,W)

        return X, y
    
    def preprocess_depth(self, depth_image):
        if depth_image.ndim == 2:
            x = depth_image[:, :, np.newaxis]
            depth_image = np.copy(x)
        depth_image[depth_image[:,:,0]>50] = 15
        depth_image = depth_image[:,:,0]
        
        return depth_image
    
    def __repr__(self) -> str:
        return f'DenseTact Dataset with {self.__len__()} samples'


def center_crop(image, crop_px):
    # Read the image
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    
    # Calculate cropping coordinates
    left = crop_px
    top = crop_px
    right = width - crop_px
    bottom = height - crop_px
    
    # Perform the crop
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def read_exr_depth(filename, depth_channel='V'):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to determine the image dimensions
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the depth channel
    if depth_channel in header['channels']:
        depth_data = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
    else:
        # raise ValueError(f"Depth channel '{depth_channel}' not found in the EXR file.")
        return None

    # Convert depth data to a numpy array
    depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))
    # depth_array = np.zeros((height, width))
    return depth_array

def read_exr_cv(filename):
    # Open the EXR file
    exr_image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return exr_image

def crop_image(image, left, right, top, bottom):
    shape = image.shape
    if len(shape) == 2:
        height, width = shape
    else:
        height, width, _ = shape
    
    cropped_image = image[top:height - bottom, left:width - right]
    return cropped_image   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set dataset parameters dynamically')
    parser.add_argument('--samples_dir', type=str, default='real_world_dataset', help='Directory of the samples')
    parser.add_argument('--root_dir', type=str, default='../DenseTact-Calibration-M/real_world_data_v2', help='Root directory of original dataset')
    parser.add_argument('--is_real_world', type=bool, default=True, help='Flag to indicate if it is real world data')

    args = parser.parse_args()

    samples_dir = args.samples_dir
    root_dir = args.root_dir
    is_real_world = args.is_real_world
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])
    
    
    dataset = FullDataset(transform=transform, samples_dir=samples_dir, 
                          root_dir=root_dir, is_real_world=is_real_world, output_type='full')
    
    print()
    full_max = 0
    full_min = 0
    
    idxes = np.random.randint(0, len(dataset), 1000)
    for i in (idxes):
        X, y = dataset[i]
        
        y = y.permute(1, 2, 0).numpy()
        # visualize first channel
        plt.imshow(y[:,:,0])
        plt.show()
        
        # shape of y is 3 by 256 by 256

    
    # dataset.construct_dataset()
    
    ax[1, 2].imshow(displacement_img[:,:,0])
    ax[1, 2].set_title("Displacement Image")
    plt.show()
    
    # use this line to construct the dataset if it has not been constructed
    # dataset.construct_dataset()
    # get random idxes
    # idxes = np.random.randint(0, len(dataset), 1000)
    # for i in (idxes):
    #     X, y = dataset[i]
    #     # plot the second channel
    #     y = y.permute(1, 2, 0).numpy()
    #     print(y.shape)
    #     # plot the 2nd channel of y
    #     plt.imshow(y[:,:,1])
    #     plt.show()

    #     deformed_img_norm = X[0:3]
    #     undeformed_img_norm = X[3:6]
    #     image_diff = X[6]
        
    #     # reshape to python image format
    #     deformed_img_norm = deformed_img_norm.permute(1, 2, 0).numpy()
    #     undeformed_img_norm = undeformed_img_norm.permute(1, 2, 0).numpy()
    #     image_diff = image_diff.numpy()
        
    #     # plot images and diff
    #     fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    #     ax[0].imshow(deformed_img_norm)
    #     ax[0].set_title("Deformed Image")
    #     ax[1].imshow(undeformed_img_norm)
    #     ax[1].set_title("Undeformed Image")
    #     ax[2].imshow(image_diff, cmap='gray')
    #     ax[2].set_title("Image Difference")
    #     plt.show()
        
        
    #     plt.imshow(relative_depth)
    #     plt.show()
        
    #     x1 = y[0].numpy()
    #     y1 = y[1].numpy()
    #     z1 = y[2].numpy()
    #     # plot all three
    #     fig, ax = plt.subplots(1, 3, figsize=(12, 8))   
    #     ax[0].imshow(relative_depth)
    #     ax[0].set_title("Relative Depth")
    #     ax[1].imshow(y1)
    #     ax[1].set_title("Y")
    #     ax[2].imshow(z1)
    #     ax[2].set_title("Z")
    #     plt.show()

    #     # visualize the images
    #     cnorm_img = y[0:3]
    #     stress1_img = y[3:6]
    #     stress2_img = y[6:9]
    #     displacement_img = y[9:12]
    #     area_shear_img = y[12:15]
        
    #     # convert to numpy
    #     cnorm_img = cnorm_img.permute(1, 2, 0).numpy()
    #     stress1_img = stress1_img.permute(1, 2, 0).numpy()
    #     stress2_img = stress2_img.permute(1, 2, 0).numpy()
    #     displacement_img = displacement_img.permute(1, 2, 0).numpy()
    #     area_shear_img = area_shear_img.permute(1, 2, 0).numpy()
        
        
    #     # visualize the images
    #     fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    #     ax[0, 0].imshow(deformed_img_norm)
    #     ax[0, 0].set_title("Deformed Image")
    #     ax[0, 1].imshow(undeformed_img_norm)
    #     ax[0, 1].set_title("Undeformed Image")
        
    #     ax[0, 2].imshow(image_diff, cmap='gray')
    #     ax[0, 2].set_title("Image Difference")
        
    #     # show one channel 
    #     ax[1, 0].imshow(cnorm_img[:,:,0])
    #     ax[1, 0].set_title("CNORM Image")
        
    #     ax[1, 1].imshow(stress1_img[:,:,0])
    #     ax[1, 1].set_title("Stress1 Image")
        
    #     ax[1, 2].imshow(displacement_img[:,:,0])
    #     ax[1, 2].set_title("Displacement Image")
    #     plt.show()
