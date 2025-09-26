# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:14:34 2025

@author: root
"""
import spectral.io.envi as envi
from pathlib import Path
import sys
import os
import random
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from dimensionality_reduction import PCAnalysis

class dataset(Dataset): 
    def __init__(self, images, labels, info_indices): 
        self.images = images
        self.labels = labels
        self.info_indices = info_indices
  
    def __len__(self): 
        return len(self.images) 
  
    def __getitem__(self, index): 
        image = torch.tensor(self.images[index]).float()
        #image dimensions (H, W, D)
        image = torch.transpose(image, 0, 2) #(D, W, H)
        image = torch.transpose(image, 1, 2) #(D, H, W); ordering compatible with network

        labels = torch.tensor(self.labels[index])

        info_indices =  torch.tensor(self.info_indices[index])
        

        return image, labels, info_indices
    20
class Loader():
    def __init__(self, n_components):
        self.n_components = n_components

    def open_datasets(self, set_split_type: str):
        images = None
        labels = None
        info_indices = None
        with open('./datasets/split_data/' + set_split_type + '/images.pkl', 'rb') as f:
            images = pickle.load(f)
                           
        with open('./datasets/split_data/' + set_split_type + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            
        with open('./datasets/split_data/' + set_split_type + '/info_indices.pkl', 'rb') as f:
            info_indices = pickle.load(f)

        return images, labels, info_indices
    
    def form_dataset(self, set_split_type: str):
        images, labels, info_indices = self.open_datasets(set_split_type)
        data = dataset(images, labels, info_indices)
        return data


    def load_data(self):
        print("loading datasets...")
        
        data_train = self.form_dataset("train")
        data_val = self.form_dataset("validation")
        data_test = self.form_dataset("test")
                
        return data_train, data_val, data_test

##to-do
#generate script with metadata dictionary
#standardization

def main():
    loader = Loader()
    loader.load_data()
    
if __name__ == "__main__":
    main()


