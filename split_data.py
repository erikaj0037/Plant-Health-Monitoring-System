# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:13:23 2025

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
from gather_raw_data import GatherData

class SplitData(GatherData):
    def __init__(self, n_components):
        self.image_shape = np.array([512, 512, n_components])
        self.train_set_images, self.train_set_labels, self.train_set_info = self.get_empty_sets(self.image_shape)
        self.val_set_images, self.val_set_labels, self.val_set_info = self.get_empty_sets(self.image_shape)
        self.test_set_images, self.test_set_labels, self.test_set_info = self.get_empty_sets(self.image_shape)

    def shuffle_sets(self, images:np.ndarray, labels:list, info:list):
        assert len(images)==len(labels), "Length of image and labels sets are unequal!"
        assert len(images)==len(info), "Length of image and info sets are unequal!"

        indices = np.arange(len(images))
        random.shuffle(indices)
        
        images = images[indices]
        labels = [labels[i] for i in indices]
        info = [info[i] for i in indices]

        return images, labels, info


    def shuffle_save_sets(self, images:np.ndarray, labels:list, info:list, set_split_type: str): #set split types: train, validation, test
        images, labels, info = self.shuffle_sets(images[1:], labels, info) #image[1:] removes dummy element at index 0

        with open('./datasets/split_data/' + set_split_type + '/images.pkl', 'wb') as f:
            pickle.dump(images, f)
                           
        with open('./datasets/split_data/' + set_split_type + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
            
        with open('./datasets/split_data/' + set_split_type + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)

        with open('./datasets/split_data/' + set_split_type + '/info_indices.pkl', 'wb') as f: #will be passed through dataloader, which only accepts numerical values, not strings
            info_indices = list(np.arange(len(info)))
            pickle.dump(info_indices, f)

    def standardize_train_set(self, training_set):
        mean = np.mean(training_set)
        std = np.std(training_set)
        training_set_standardized = (training_set - mean) / std

        with open('./datasets/split_data/standardization_values.pkl', 'wb') as f:
            pickle.dump({"mean": mean, "std": std}, f)
        return training_set_standardized
    
    def standardize(self, set):
        std_values = None
        with open('./datasets/split_data/standardization_values.pkl', 'rb') as f:
            std_values = pickle.load(f)
        set_standardized = (set - std_values['mean']) / std_values['std']
        return set_standardized
    
    def normalize_train_set(self, training_set):
        min = np.min(training_set)
        max = np.max(training_set)
        with open('./datasets/split_data/normalization_values.pkl', 'wb') as f:
            pickle.dump({"max": max, "min": min}, f)
        training_set_normalized = (training_set - min) / (max - min)
        return training_set_normalized
    
    def normalize(self, set):
        norm_values = None
        with open('./datasets/split_data/normalization_values.pkl', 'rb') as f:
            norm_values = pickle.load(f)
        set_normalized = (set - norm_values['min']) / (norm_values['max'] - norm_values['min'])
        return set_normalized

    def split_set(self, images:np.ndarray, labels:list, info:list, threshold_ratio: float, shuffle: bool):
        if shuffle:
            images, labels, info = self.shuffle_sets(images, labels, info)

        test_thresh =int(np.floor(threshold_ratio*len(images)))
        images_split = [images[:test_thresh], images[test_thresh:]]
        labels_split = [labels[:test_thresh], labels[test_thresh:]]
        info_split = [info[:test_thresh], info[test_thresh:]]

        return images_split[0], labels_split[0], info_split[0], images_split[1], labels_split[1], info_split[1]

    def append_set(self, image_set:np.ndarray, label_set:list, info_set:list, images:np.ndarray, labels:list, info:list):
        image_set = np.append(image_set, images, axis = 0)
        label_set += labels
        info_set += info
        
        return image_set, label_set, info_set

    def split_data(self):
        print("splitting data...")
    
        path = Path(r'./datasets/reduced_data/')
        for month_folder in sorted(path.iterdir()):
            if month_folder.name[0] == ".":
                    continue
            print("month:", month_folder.name)
            
            images, labels, info = self.open_gathered_data(month_folder)

            # images = self.standardize(images)
            split_thresh_ratio = 0.8 # 80/20% train/test, 80/20% train/val
            images_train, labels_train, info_train, images_test, labels_test, info_test = self.split_set(images, labels, info, split_thresh_ratio, shuffle = 'True')
            images_train, labels_train, info_train, images_val, labels_val, info_val = self.split_set(images_train, labels_train, info_train, split_thresh_ratio, shuffle = 'False')
            
            self.train_set_images, self.train_set_labels, self.train_set_info = self.append_set(self.train_set_images, self.train_set_labels, self.train_set_info, images_train, labels_train, info_train)
            self.val_set_images, self.val_set_labels, self.val_set_info = self.append_set(self.val_set_images, self.val_set_labels, self.val_set_info, images_val, labels_val, info_val)
            self.test_set_images, self.test_set_labels, self.test_set_info = self.append_set(self.test_set_images, self.test_set_labels, self.test_set_info, images_test, labels_test, info_test)

            self.train_set_images = self.normalize_train_set(self.train_set_images)
            self.val_set_images = self.normalize(self.val_set_images)
            self.test_set_images = self.normalize(self.test_set_images)

            print("training subset size:",  len(self.train_set_images))
            print("validation subset:", len(self.val_set_images))
            print("test subset:", len(self.test_set_images))
        
        #insert standardization
        self.shuffle_save_sets(self.train_set_images, self.train_set_labels, self.train_set_info, "train")
        self.shuffle_save_sets(self.val_set_images, self.val_set_labels, self.val_set_info, "validation")
        self.shuffle_save_sets(self.test_set_images, self.test_set_labels, self.test_set_info, "test")
               
def main():
    data_splitter = SplitData(n_components=3)
    data_splitter.split_data()
    
if __name__ == "__main__":
    main()