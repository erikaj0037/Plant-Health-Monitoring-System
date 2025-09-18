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

class Loader():
    def __init__(self, n_components):
        self.n_components = n_components

    def open_datasets(self, set_split_type: str):
        images, labels, info_indices = None
        with open('./datasets/split_data/' + set_split_type + '/images.pkl', 'rb') as f:
            images = pickle.load(f)
                           
        with open('./datasets/split_data/' + set_split_type + '/labels.pkl', 'rb') as f:
            labels = pickle.dump(labels, f)
            
        with open('./datasets/split_data/' + set_split_type + '/info_indices.pkl', 'rb') as f:
            info_indices = pickle.dump(info_indices, f)

        return images, labels, info_indices
    
    def form_dataset(self, set_split_type: str):
        images, labels, info_indices = self.open_data_sets(set_split_type)
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


