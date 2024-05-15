#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:08:05 2023

@author: ruchak
"""

import pandas as pd
import os
import numpy as np
import shutil
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import utils
import torch.nn as nn
import timm
import torch.nn.functional as F
from vision_transformer_pytorch import VisionTransformer

class ViTBase16(nn.Module):
    def __init__(self, pretrained = True):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained = True)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
def template_matching(path):
    '''
    Returns the cropped image based on our template

    Parameters
    ----------
    path : String
            The path to the input image

    Returns
    -------
    resulting_img : Cropped image
            The cropped image

    '''
    try:
        original_array = cv2.imread(path)
        image_array = cv2.cvtColor(original_array, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(cv2.imread('/home/ruchak/Desktop/RDF/scripts/rdf_template.png'), cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(image_array, template, cv2.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
        # Draw a rectangle around the matched region
        h, w = template.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image_array, top_left, bottom_right, (0, 255, 0), 2)
        resulting_img = original_array[top_left[1]: top_left[1] + h, top_left[0]: top_left[0] + w]

        return resulting_img
    except:
        return np.empty((0,0,3))

class CreateDataset(Dataset):
    def __init__(self, df):
        self.df = df 
        
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df['Path'].iloc[idx]
        img = Image.open(row).convert('RGB')
        
        
        img = self.transforms(img)
        
        return img

def get_df(waferlist, folder_substring):
    urls = []
    df = pd.DataFrame()
    for wafer in waferlist:
        all_folders = sorted(os.listdir('/extrastg/mdfs14/hamr-analysis/image_repo/proc_DSCVRIX/images/nfs/' + wafer + '/'))
        folder_subset = [query for query in all_folders if any(sub in query for sub in folder_substring)]
        for i in range(len(folder_subset)):
            all_images = sorted(os.listdir('/extrastg/mdfs14/hamr-analysis/image_repo/proc_DSCVRIX/images/nfs/' + wafer + '/' + folder_subset[i] + '/'))
            for j in range(len(all_images)):
                urls.append('/extrastg/mdfs14/hamr-analysis/image_repo/proc_DSCVRIX/images/nfs/' + wafer + '/' + folder_subset[i] + '/' + all_images[j])
    df['URL'] = urls
    
    return df

def check_valid_template(df):
    to_drop = []
    urls = []
    urls_orig = []
    for i in range(len(df)):
        path = df['URL'].iloc[i]
        out = template_matching(path)
        flag = 0
        for j in range(3):
            if out.shape[j] == 0: 
                flag = 1
                to_drop.append(i)
                break
        if not(flag):
            urls_orig.append(path.replace('/extrastg/mdfs14/hamr-analysis', 'ida.seagate.com:9081'))
            storage_loc, dest_folder = path.split('proc_DSCVRIX')
            dest = storage_loc + 'cropped_images_RDF/proc_DSCVRIX' + dest_folder
            urls.append(dest.replace('/extrastg/mdfs14/hamr-analysis', 'ida.seagate.com:9081'))
            
            os.makedirs(os.path.dirname(dest), mode = 0o777, exist_ok=True)
            
            cv2.imwrite(dest, out)
                
    df = df.drop(to_drop, axis = 0).reset_index(drop = True)
    df['URL to cropped image'] = urls
    df['URL to actual image'] = urls_orig
            
    return df
def feature_extraction(df):
    dataset = CreateDataset(df)
    train_loader = DataLoader(dataset = dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ViTBase16()
    model = model.to(device)



    model.eval()
    test_feature_space = []
    with torch.no_grad():
        for i, (imgs) in enumerate(train_loader):
            images = imgs.to(device)

            features = model(images)
            features = torch.squeeze(features)
            
            test_feature_space.append(features)

        test_feature_space = torch.cat(test_feature_space, dim = 0).contiguous().cpu().numpy()
    
    return test_feature_space

def main(df):
    
    test_feature_space = feature_extraction(df)
    return df, test_feature_space
    
    