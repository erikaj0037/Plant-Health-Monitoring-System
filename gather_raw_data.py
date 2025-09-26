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

class GatherData():
    def __init__(self):
        
        self.metadata = {'description': 'Data recorded with Specim IQ', 'samples': '256', 'lines': '256', 'bands': '204', 'header offset': '0', 'file type': 'ENVI', 'data type': '4', 'interleave': 'BIL', 'sensor type': 'SPECIM IQ', 'byte order': '0', 'default bands': ['70', '53', '19'], 'latitude': '0.00000000', 'longitude': '0.00000000', 'acquisition date': '07-04-2021', 'errors': 'none', 'binning': ['1', '1'], 'tint': '17', 'fps': '58.8235', 'wavelength': ['397.32', '400.20', '403.09', '405.97', '408.85', '411.74', '414.63', '417.52', '420.40', '423.29', '426.19', '429.08', '431.97', '434.87', '437.76', '440.66', '443.56', '446.45', '449.35', '452.25', '455.16', '458.06', '460.96', '463.87', '466.77', '469.68', '472.59', '475.50', '478.41', '481.32', '484.23', '487.14', '490.06', '492.97', '495.89', '498.80', '501.72', '504.64', '507.56', '510.48', '513.40', '516.33', '519.25', '522.18', '525.10', '528.03', '530.96', '533.89', '536.82', '539.75', '542.68', '545.62', '548.55', '551.49', '554.43', '557.36', '560.30', '563.24', '566.18', '569.12', '572.07', '575.01', '577.96', '580.90', '583.85', '586.80', '589.75', '592.70', '595.65', '598.60', '601.55', '604.51', '607.46', '610.42', '613.38', '616.34', '619.30', '622.26', '625.22', '628.18', '631.15', '634.11', '637.08', '640.04', '643.01', '645.98', '648.95', '651.92', '654.89', '657.87', '660.84', '663.81', '666.79', '669.77', '672.75', '675.73', '678.71', '681.69', '684.67', '687.65', '690.64', '693.62', '696.61', '699.60', '702.58', '705.57', '708.57', '711.56', '714.55', '717.54', '720.54', '723.53', '726.53', '729.53', '732.53', '735.53', '738.53', '741.53', '744.53', '747.54', '750.54', '753.55', '756.56', '759.56', '762.57', '765.58', '768.60', '771.61', '774.62', '777.64', '780.65', '783.67', '786.68', '789.70', '792.72', '795.74', '798.77', '801.79', '804.81', '807.84', '810.86', '813.89', '816.92', '819.95', '822.98', '826.01', '829.04', '832.07', '835.11', '838.14', '841.18', '844.22', '847.25', '850.29', '853.33', '856.37', '859.42', '862.46', '865.50', '868.55', '871.60', '874.64', '877.69', '880.74', '883.79', '886.84', '889.90', '892.95', '896.01', '899.06', '902.12', '905.18', '908.24', '911.30', '914.36', '917.42', '920.48', '923.55', '926.61', '929.68', '932.74', '935.81', '938.88', '941.95', '945.02', '948.10', '951.17', '954.24', '957.32', '960.40', '963.47', '966.55', '969.63', '972.71', '975.79', '978.88', '981.96', '985.05', '988.13', '991.22', '994.31', '997.40', '1000.49', '1003.58']}

        self.wavelengths = self.metadata['wavelength']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def save_to_metadata(self, metadata_element):
    #     file_path = './datasets/raw_data/metadata.pkl'
    #     if os.path.exists(file_path):
    #         with open(file_path, 'r') as f:
    #             pickle.load()
    #             pickle.dump()

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
            if len(labels_DF.values) > 0:
                labels = [labels_DF.columns.to_numpy().astype(int).tolist()]
                labels += labels_DF.values.tolist()
                return labels
            else: 
                return [[0,0]]

        else: 
            return [[0,0]]
        
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

        #saving info list with folder information associated with each image (will not be passed to dataloader)
        print("saving info list...")
        print("info list size: " + str(len(info)))
        with open('./datasets/reduced_data/' + str(month_folder.name) + '/info.pkl', 'wb') as f:
            pickle.dump(info, f)
    
    def gather_data(self, n_components: int):
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
                    # try:
                    if hdr_file and data_file:
                        print("day:", day_folder.name)
                        print("plant:", plant_folder.name)
                        pca_algorithm = PCAnalysis(n_components=n_components) #must be reinitialized each time there is new image
                        hsi = self.get_hsi(hdr_file, data_file)
                        hsi_reduced = pca_algorithm.pca_transform(hsi)

                        images = np.append(images, np.expand_dims(hsi_reduced, axis = 0), axis = 0)
                        labels.append(self.get_image_labels(csv_file))
                        info.append([month_folder.name, day_folder.name, plant_folder.name])
                    # except:
                    #     continue

            self.save_gathered_data(images, labels, info, month_folder)

##to-do
#generate script with metadata dictionary
#standardization

def main():
    gather_raw_data = GatherData()
    gather_raw_data.gather_data(n_components=3)
    
if __name__ == "__main__":
    main()
