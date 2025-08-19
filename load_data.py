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
# from sklearn.model_selection import train_test_split

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

class Loader():
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
        images = np.empty((1, image_shape[0], image_shape[1], image_shape[2]))
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

    def image_set_mean(self, image_count, image_shape, voxel_value_sum):
        total_voxels = image_count*np.prod(image_shape)
        voxel_value_mean = voxel_value_sum / total_voxels
        return voxel_value_mean, total_voxels
    
    def open_gathered_images(self, dataset_folder, month_folder):
        images = None
        labels = None
        info = None
        
        for set_folder in sorted(month_folder.iterdir()):
            with open('./datasets/' + dataset_folder + '/' + str(month_folder.name) + '/' + str(set_folder.name) + 'images.pkl', 'rb') as f:
                images = pickle.load(f)
                
                
        

    def open_gathered_labels_info(self, dataset_folder, month_folder):
        with open('./datasets/' + dataset_folder + '/' + str(month_folder.name) + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)

        with open('./datasets' + dataset_folder + '/' + str(month_folder.name) + '/info.pkl', 'rb') as f:
            info = pickle.load(f)

        return images, labels, info
        
    def save_image_subset(self, image_array, set_number, month_folder):
        print("saving images...")
        with open('./datasets/raw_data/' + str(month_folder.name) + '/image_set' + str(set_number) + '.pkl', 'wb') as f:
            images = image_array[1:]
            print("images list size: " + str(len(image_array)))
            print("images memory size: " + str(sys.getsizeof(image_array) / 1000000000) + " GB")
            pickle.dump(images, f)
    
    def save_label_set(self, labels, month_folder):
        print("saving labels...")
        print("labels list size: " + str(len(labels)))
        
        print("labels memory size: " + str(sys.getsizeof(labels) / 1000000000) + " GB")
        with open('./datasets/raw_data/' + str(month_folder.name) + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)

    def save_info_set(self, info, month_folder):
        print("info list size: " + str(len(info)))
        with open('./datasets/raw_data/' + str(month_folder.name) + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)
    
    def gather_data(self, n_neighbors: int, n_components = int, save_files = bool):
        print("gathering data...")
 
        path = Path(r'./datasets/apple_fireblight')
        image_shape = np.array([512, 512, 204])
        image_count = 0
        image_set_count = 0
        voxel_value_sum = 0

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

                            hsi = self.get_hsi(hdr_file, data_file)
                            image_count += 1
                            voxel_value_sum += np.sum(hsi)

                            if save_files:
                                images = np.append(images, np.array([hsi]), axis = 0)
                                
                                if sys.getsizeof(images) / 1000000000 >= 24:
                                    
                                    image_set_count += 1
                                    print("check1")
                                    self.save_image_subset(images, image_set_count, month_folder)
                                    print("check2")
                                    images, _, _ = self.get_empty_sets(image_shape)
                                    print("check3")

                                info.append([month_folder.name, day_folder.name, plant_folder.name])

                                labels.append(self.get_image_labels(csv_file))
                            
                    except:
                        continue
            if save_files:
                image_set_count += 1

                self.save_image_subset(images, image_set_count, month_folder)
                self.save_label_set(labels, month_folder)
                self.save_info_set(info, month_folder)
            
            voxel_value_mean, total_voxels = self.image_set_mean(image_count, image_shape, voxel_value_sum)
            
            return [voxel_value_mean, total_voxels]


    def shuffle_save_sets(self, images, labels, info, set: str, dataset_folder):
        i = np.arange(len(images))
        random.shuffle(i)
        images = images[i]
        labels = [labels[k] for k in i]
        info = [info[k] for k in i]

        with open('./datasets/' + dataset_folder + '/' + set + '/images.pkl', 'wb') as f:
            pickle.dump(images, f)
                           
        with open('./datasets/' + dataset_folder + '/' + set + '/labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
            
        with open('./datasets/' + dataset_folder + '/' + set + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)

    def standardize_set(self, dataset_folder: str):
        path = Path(r'./datasets/' + dataset_folder)

        for month_folder in sorted(path.iterdir()):
            if month_folder.name[0] == ".":
                    continue
            if month_folder.name[0] == "sets":
                    continue
            
            with open('./datasets/' + dataset_folder + '/' + str(month_folder.name) + '/images.pkl', 'rb') as f:
            images = pickle.load(f)
            
            with open('./datasets/' + dataset_folder + '/' + str(month_folder.name) + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)

            with open('./datasets' + dataset_folder + '/' + str(month_folder.name) + '/info.pkl', 'rb') as f:
                info = pickle.load(f)

            
    def split_data(self, n_components: int, dataset_folder: str, image_shape: np.ndarray, image_count):
        print("splitting data...")
        
        train_set_images, train_set_labels, train_set_info = self.get_empty_sets(image_shape)
        val_set_images, val_set_labels, val_set_info = self.get_empty_sets(image_shape)
        test_set_images, test_set_labels, test_set_info = self.get_empty_sets(image_shape)
        
        path = Path(r'./datasets/' + dataset_folder + '/')
        for month_folder in sorted(path.iterdir()):
            if month_folder.name[0] == ".":
                    continue
            if month_folder.name[0] == "sets":
                    continue
            print("month:", month_folder.name)
            
            images, labels, info = self.open_gathered_data(dataset_folder, month_folder)
        
            #shuffle and divide into 5 sets
            indices = np.arange(image_count)
            random.shuffle(indices)
            #divide into training, validation, and test sets along with subsets
            image_set = np.zeros((image_count, image_shape[0], image_shape[1], image_shape[2]))
            # print(len(labels))
            
            images = images[indices]
            labels = [labels[i] for i in indices]
            info = [info[i] for i in indices]

            # images = self.standardize(images)
            
            test_thresh =int(np.floor(.8*len(images)))
            images_train_val = images[:test_thresh]
            labels_train_val = labels[:test_thresh]
            info_train_val = info[:test_thresh]
            
            images_test = images[test_thresh:]
            labels_test = labels[test_thresh:]
            info_test = info[test_thresh:]
            
            val_thresh = int(np.floor(.8*len(images_train_val)))
            images_train = images_train_val[:val_thresh]
            labels_train = labels_train_val[:val_thresh]
            info_train = info_train_val[:val_thresh]
            
            images_val = images_train_val[val_thresh:]
            labels_val = labels_train_val[val_thresh:]
            info_val = info_train_val[val_thresh:]
            
            train_set_images = np.append(train_set_images, images_train, axis = 0)
            train_set_labels += labels_train
            train_set_info += info_train
            
            val_set_images = np.append(val_set_images, images_val, axis = 0)
            val_set_labels += labels_val
            val_set_info += info_val
            
            test_set_images = np.append(test_set_images, images_test, axis = 0)
            test_set_labels += labels_test
            test_set_info += info_test
            
            print("training subset size:",  len(train_set_images))
            
            print("validation subset:", len(val_set_images))
            
            print("test subset:", len(test_set_images))
        
        train_set_images = train_set_images[1:] #removes dummy image at index 0
        val_set_images = val_set_images[1:]
        test_set_images = test_set_images[1:]
        
        
        self.shuffle_save_sets(train_set_images, train_set_labels, train_set_info, "training", dataset_folder)
        self.shuffle_save_sets(val_set_images, val_set_labels, val_set_info, "validation", dataset_folder)
        self.shuffle_save_sets(test_set_images, test_set_labels, test_set_info, "test", dataset_folder)

        train_set_images, val_set_images, test_set_images = self.standardize(train_set_images, val_set_images, test_set_images)
        
        ###insert avg function
        
                
    def load_data(self, save_files = 'False'):
        n_neighbors = 20
        n_components = 10
        statistical_data =  self.gather_data(n_neighbors, n_components, save_files) #returns [voxel_value mean, total number of voxels]
        # dataset_folder = "raw_data"
        # self.split_data(n_components)
        # 
        # print("loading datasets...")
        # data_train = None
        # data_val = None
        # data_test = None
        
        # with open('./datasets/' + dataset_folder + '/sets/train_images.pkl', 'rb') as f:

        #     with open('./datasets/' + dataset_folder + '/sets/train_labels.pkl', 'rb') as g:

        #         with open('./datasets/' + dataset_folder + '/sets/train_info.pkl', 'rb') as h:
            
        #             data_train = dataset(pickle.load(f), pickle.load(g), pickle.load(h))
        #             print("training images, labels, & info loaded")
        
        # with open('./datasets/' + dataset_folder + '/sets/val_images.pkl', 'rb') as f:
            
        #     with open('./datasets/' + dataset_folder + '/sets/val_labels.pkl', 'rb') as g:

        #         with open('./datasets/' + dataset_folder + '/sets/val_info.pkl', 'rb') as h:
            
        #             data_val = dataset(pickle.load(f), pickle.load(g), pickle.load(h))
        #             print("validation images, labels, & info loaded")
       
        # with open('./datasets/' + dataset_folder + '/sets/test_images.pkl', 'rb') as f:
            
        #     with open('./datasets/' + dataset_folder + '/sets/test_labels.pkl', 'rb') as g:

        #         with open('./datasets/' + dataset_folder + '/sets/test_info.pkl', 'rb') as h:
            
        #             data_test = dataset(pickle.load(f), pickle.load(g), pickle.load(h))
        #             print("test images, labels, & info loaded")
                
        # return data_train, data_val, data_test, self.metadata, [n_neighbors, n_components]


def main():
    loader = Loader()
    loader.load_data()
    
if __name__ == "__main__":
    main()
