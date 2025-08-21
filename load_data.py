# import spectral as spy
import spectral.io.envi as envi
from pathlib import Path
import sys
import random
import pickle
import pandas as pd
import numpy as np

# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.datasets import load_digits


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
import umap
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import skcuda.linalg as linalg
# from skcuda.linalg import PCA as cuPCA
from dimensionality_reduction import PCAnalysis

class dataset(Dataset): 
    def __init__(self, images, labels, info): 
        self.images = images
        self.labels = labels
        self.info = info
  
    def __len__(self): 
        return len(self.images) 
  
    def __getitem__(self, index): 
        image = torch.tensor(self.images[index]).float()
        #image dimensions (H, W, D)
        image = torch.transpose(image, 0, 2) #(D, W, H)
        image = torch.transpose(image, 1, 2) #(D, H, W); ordering necessary for network

        labels = torch.tensor(self.labels[index])

        info =  self.info[index]
        info_ascii = self.string_to_ascii(info)
        info = tuple([torch.tensor(item) for item in info_ascii])
        info = torch.squeeze(pad_sequence(torch.unsqueeze(pad_sequence(info, batch_first=True, padding_value=0), dim=2), batch_first=True, padding_value=0))
        

        return image, labels, info
    
    def string_to_ascii(self, string_list):
        ascii_list = []
        for string in string_list:
            ascii_list.append([ord(char) for char in string])
        # ascii_padded = self.pad_ascii(ascii_list)
        
        return ascii_list
    
    def pad_ascii(self, ascii_list):
        max_str_len = 14
#         max_ = 0
#         for ascii_ in ascii_list:
#             if len(ascii_) > max_:
#                 max_ = len(ascii_)
        ascii_padded = np.zeros((1,max_str_len))
        for ascii_ in ascii_list:
            ascii_padded = np.append(ascii_padded, np.array([np.pad(np.array(ascii_), (0, max_str_len - len(ascii_)))]), axis = 0)

        return ascii_padded[1:].astype(int)

#     # ASCII to String
#     def ascii_to_string(ascii_list):
#         return ''.join(chr(char) for char in ascii_list)

class GatherData():
    def __init__(self):
        
        self.metadata = {'description': 'Data recorded with Specim IQ', 'samples': '256', 'lines': '256', 'bands': '204', 'header offset': '0', 'file type': 'ENVI', 'data type': '4', 'interleave': 'BIL', 'sensor type': 'SPECIM IQ', 'byte order': '0', 'default bands': ['70', '53', '19'], 'latitude': '0.00000000', 'longitude': '0.00000000', 'acquisition date': '07-04-2021', 'errors': 'none', 'binning': ['1', '1'], 'tint': '17', 'fps': '58.8235', 'wavelength': ['397.32', '400.20', '403.09', '405.97', '408.85', '411.74', '414.63', '417.52', '420.40', '423.29', '426.19', '429.08', '431.97', '434.87', '437.76', '440.66', '443.56', '446.45', '449.35', '452.25', '455.16', '458.06', '460.96', '463.87', '466.77', '469.68', '472.59', '475.50', '478.41', '481.32', '484.23', '487.14', '490.06', '492.97', '495.89', '498.80', '501.72', '504.64', '507.56', '510.48', '513.40', '516.33', '519.25', '522.18', '525.10', '528.03', '530.96', '533.89', '536.82', '539.75', '542.68', '545.62', '548.55', '551.49', '554.43', '557.36', '560.30', '563.24', '566.18', '569.12', '572.07', '575.01', '577.96', '580.90', '583.85', '586.80', '589.75', '592.70', '595.65', '598.60', '601.55', '604.51', '607.46', '610.42', '613.38', '616.34', '619.30', '622.26', '625.22', '628.18', '631.15', '634.11', '637.08', '640.04', '643.01', '645.98', '648.95', '651.92', '654.89', '657.87', '660.84', '663.81', '666.79', '669.77', '672.75', '675.73', '678.71', '681.69', '684.67', '687.65', '690.64', '693.62', '696.61', '699.60', '702.58', '705.57', '708.57', '711.56', '714.55', '717.54', '720.54', '723.53', '726.53', '729.53', '732.53', '735.53', '738.53', '741.53', '744.53', '747.54', '750.54', '753.55', '756.56', '759.56', '762.57', '765.58', '768.60', '771.61', '774.62', '777.64', '780.65', '783.67', '786.68', '789.70', '792.72', '795.74', '798.77', '801.79', '804.81', '807.84', '810.86', '813.89', '816.92', '819.95', '822.98', '826.01', '829.04', '832.07', '835.11', '838.14', '841.18', '844.22', '847.25', '850.29', '853.33', '856.37', '859.42', '862.46', '865.50', '868.55', '871.60', '874.64', '877.69', '880.74', '883.79', '886.84', '889.90', '892.95', '896.01', '899.06', '902.12', '905.18', '908.24', '911.30', '914.36', '917.42', '920.48', '923.55', '926.61', '929.68', '932.74', '935.81', '938.88', '941.95', '945.02', '948.10', '951.17', '954.24', '957.32', '960.40', '963.47', '966.55', '969.63', '972.71', '975.79', '978.88', '981.96', '985.05', '988.13', '991.22', '994.31', '997.40', '1000.49', '1003.58']}

        self.wavelengths = self.metadata['wavelength']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reduce(self, image, n_neighbors, n_components):
        # print("reducing dimensionality of image...")
        reducer = umap.UMAP(n_neighbors = n_neighbors, n_components = n_components)

        image = image.reshape((np.prod(image.shape[:2]), image.shape[2]))
        scaled_image = StandardScaler().fit_transform(image)

        embedding = reducer.fit_transform(scaled_image)

        reduced_image = embedding.reshape((512,512,n_components))

        return reduced_image
        
    # def standardize(self, images):
    #     print("standardizing data...")
    #     images_mean = np.sum(images) / np.prod(images.shape)
    #     images_std = np.std(images)
    #     images_standardized = (images - images_mean) / images_std

    #     return images_standardized
    
    def standardize(self, training_set, validation_set, test_set):
        print("standardizing data...")
        set = np.append(training_set, validation_set, axis = 0)
        set = np.append(set, test_set, axis = 0)
        set_mean = np.sum(set) / np.prod(set.shape)
        set_std = np.std(set)
        images_standardized = (set - set_mean) / set_std

        i = len(training_set)
        j = len(validation_set)
        train_standardized = images_standardized[:i]
        val_standardized = images_standardized[i:i+j]
        test_standardized = images_standardized[i+j:]
        return train_standardized, val_standardized, test_standardized
    
    def get_hsi(self, hdr_file, data_file):
        image = envi.open(hdr_file, data_file)
        channels = image.shape[-1]
        c = np.arange(channels)
        hsi = image.read_bands(c)
        return hsi
    
    def get_image_labels(self, csv_file):
        if csv_file:
            # print("getting labels...")
            labels_DF = pd.read_csv(csv_file)
#                             print(labels_DF.to_numpy())
            return labels_DF.values.tolist()

        else: 
            return [0]
        
    def get_empty_sets(self, image_shape):
        desired_shape = tuple(np.insert(image_shape, 0, 1))
        images = np.empty(desired_shape)
        labels = [] 
        info = []
        return images, labels, info

    def find_files(self, plant_folder):
        hdr_file = False
        data_file = False
        csv_file = False

        for file in sorted(plant_folder.iterdir()):
            if file.name[0] == ".":
                continue

            if file.suffix == '.hdr':
                hdr_file = file

            elif file.suffix == '.dat':
                data_file = file

            elif file.suffix == '.csv':
                csv_file = file
        return hdr_file, data_file, csv_file

    def open_gathered_data(self, month_folder):
        with open('./datasets/reduced_data/' + str(month_folder.name) + '/images.pkl', 'rb') as f:
                images = pickle.load(f)

        with open('./datasets/reduced_data/' + str(month_folder.name) + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)

        with open('./datasets/reduced_data/' + str(month_folder.name) + '/info.pkl', 'rb') as f:
            info = pickle.load(f)

        return images, labels, info
        
    def save_gathered_data(self, images, labels, info, month_folder):
        print("saving images...")
        with open('./datasets/reduced_data/' + str(month_folder.name) + '/images.pkl', 'wb') as f:
            images = images[1:]
            print("images list size: " + str(len(images)))
            print("images memory size: " + str(sys.getsizeof(images) / 1000000000) + " GB")
            pickle.dump(images, f)
    
        print("saving labels...")
        print("labels list size: " + str(len(labels)))
        
        print("labels memory size: " + str(sys.getsizeof(labels) / 1000000000) + " GB")
        with open('./datasets/reduced_data/' + str(month_folder.name) + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)

        print("info list size: " + str(len(info)))
        with open('./datasets/reduced_data/' + str(month_folder.name) + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)
    
    def gather_data(self, n_components = int):
        print("gathering data...")
 
        path = Path(r'./datasets/raw_data')
        image_shape = np.array([512, 512, n_components])
        
        

        for month_folder in sorted(path.iterdir()):
            if month_folder.name[0] == ".":
                    continue
            
            print("month:", month_folder.name)

            images, labels, info = self.get_empty_sets(image_shape)

            for day_folder in sorted(month_folder.iterdir()):
                if day_folder.name[0] == ".":
                    continue

                for plant_folder in sorted(day_folder.iterdir()):
                    if plant_folder.name[0] == ".":
                        continue
                    
                    hdr_file, data_file, csv_file = self.find_files(plant_folder)
                    try:
                        if hdr_file and data_file:
                            pca_algorithm = PCAnalysis(n_components=n_components) #must be reinitialized each time there is new image
                            hsi = self.get_hsi(hdr_file, data_file)
                            hsi_reduced = pca_algorithm.pca_transform(hsi)

                            images = np.append(images, np.expand_dims(hsi_reduced, axis = 0), axis = 0)
                            info.append([month_folder.name, day_folder.name, plant_folder.name])
                            labels.append(self.get_image_labels(csv_file))
                    except:
                        continue

            self.save_gathered_data(images, labels, info, month_folder)

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


    def shuffle_save_sets(self, images:np.ndarray, labels:list, info:list, set_split_type: str):
        images, labels, info = self.shuffle_sets(images[1:], labels, info) #image[1:] removes dummy element at index 0

        with open('./datasets/split_data/' + set_split_type + '/images.pkl', 'wb') as f:
            pickle.dump(images, f)
                           
        with open('./datasets/split_data/' + set_split_type + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
            
        with open('./datasets/split_data/' + set_split_type + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)

    def standardize_set(self, dataset_folder: str):
        path = Path(r'./datasets/' + dataset_folder)

        for month_folder in sorted(path.iterdir()):
            if month_folder.name[0] == ".":
                    continue
            if month_folder.name[0] == "sets":
                    continue
            
            with open('./datasets/reduced_data/' + str(month_folder.name) + '/images.pkl', 'rb') as f:
                images = pickle.load(f)
            
            with open('./datasets/reduced_data/' + str(month_folder.name) + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)

            with open('./datasets/reduced_data/' + str(month_folder.name) + '/info.pkl', 'rb') as f:
                info = pickle.load(f)

    def split_set(self, images:np.ndarray, labels:list, info:list, threshold_ratio: float, shuffle: bool):
        if shuffle:
            images, labels, info = self.shuffle_sets(images, labels, info)

        test_thresh =int(np.floor(threshold_ratio*len(images)))
        images_split = [images[:test_thresh], images[test_thresh:]]
        labels_split = [labels[:test_thresh], labels[test_thresh:]]
        info_split = [info[:test_thresh], info[test_thresh:]]

        return images_split[0], labels_split[0], info_split[0], images_split[1], labels_split[1], info_split[1]

    def append_set(image_set:np.ndarray, labels_set:list, info_set:list, images:np.ndarray, labels:list, info:list):
        image_set = np.append(image_set, images, axis = 0)
        label_set += labels
        info_set += info
        
        return image_set, label_set, info_set

    def split_data(self, image_shape: np.ndarray):
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

            print("training subset size:",  len(self.train_set_images))
            print("validation subset:", len(self.val_set_images))
            print("test subset:", len(self.test_set_images))
        
        #insert standardization
        self.shuffle_save_sets(self.train_set_images, self.train_set_labels, self.train_set_info, "train")
        self.shuffle_save_sets(self.val_set_images, self.val_set_labels, self.val_set_info, "validation")
        self.shuffle_save_sets(self.test_set_images, self.test_set_labels, self.test_set_info, "test")
        
class Loader():
    def __init__(self, n_components):
        self.n_components = n_components

    def open_datasets(self, set_split_type: str):
        images, labels, info = None
        with open('./datasets/split_data/' + set_split_type + '/images.pkl', 'rb') as f:
            images = pickle.load(f)
                           
        with open('./datasets/split_data/' + set_split_type + '/labels.pkl', 'rb') as f:
            labels = pickle.dump(labels, f)
            
        with open('./datasets/split_data/' + set_split_type + '/info.pkl', 'rb') as f:
            info = pickle.dump(info, f)

        return images, labels, info
    
    def form_dataset(self, set_split_type:str):
        images, labels, info = self.open_data_sets(set_split_type)
        data = dataset(images, labels, info)
        return data


    def load_data(self):
        print("loading datasets...")
        
        data_train = self.form_dataset("train")
        data_train = self.form_dataset("validation")
        data_test = self.form_dataset("test")
                
        return data_train, data_val, data_test, self.metadata, [n_neighbors, n_components]

##to-do
#generate script with metadata dictionary
#standardization

def main():
    loader = Loader()
    loader.load_data()
    
if __name__ == "__main__":
    main()
