import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
# import torcharrow as ta
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# import pandas as pd

# import matplotlib.pyplot as plt
# from decimal import Decimal
# from spectral import *
# import spectral.io.envi as envi
# from pathlib import Path
import sys
from networks.wnet_model import WNet3D
from networks.wnet_model_2D import WNet2D
from networks.unet_model_2D import UNet2D
from load_data import Loader
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from train_reduction import *

writer = SummaryWriter()

class TestModel():
    def __init__(self):
        self.n_components = 3
        self.k = 7
        self.batch_size = 16
        self.dr_method = "pca"
        self.model_path = "model_weights/"+ self.dr_method + "/n_components-" + str(self.n_components) + "/k-" + str(self.k) + "/batch_size-" + str(self.batch_size) + "/"

    def histogram_bins(self, image_losses):
        _, loss_bins = torch.histogram(image_losses, density = True)
        return loss_bins

    def anomaly_loss_threshold(self, bin_threshold, image_losses): #choose bin threshold from [0, max)
        loss_bins = self.histogram_bins(image_losses)
        return  loss_bins[bin_threshold]

    def get_png(self, path):
        for file in sorted(path.iterdir()):
            if file.name[0] == ".":
                continue

            if file.suffix == '.png':
                png_file = file

        return png_file
    
    def save_anomalous_image(self, image_info_index, image_losses, labels, bin_threshold):
        anomaly_loss_threshold = self.anomaly_loss_threshold(bin_threshold, image_losses)
        #read in rgb png
        with open('./datasets/split_data/test/info.pkl', 'rb') as f:
            info = pickle.load(f)
        print(image_info_index)

        for i in range(len(image_info_index)):

            image_info = info[image_info_index[i]]
            image_path = Path(r"./datasets/raw_data/" + image_info[0] + "/" + image_info[1] + "/" + image_info[2] + "/")
            png_file = self.get_png(image_path)
            image_rgb = None

            if png_file:

                image_rgb = cv2.imread(filename = png_file, flags = cv2.IMREAD_COLOR_RGB)
                image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                locs = torch.where(image_losses[i] > anomaly_loss_threshold)
                image_rgb[locs[1], locs[2]] = np.array([0,0,255])
                image_rgb[labels[i,:,0], labels[i,:,1]] = np.array([255, 165, 0])
                anom_image_path = self.model_path + "test_images/"
                os.makedirs(anom_image_path, exist_ok=True)
                plt.figure()
                plt.imshow(image_rgb)
                plt.savefig(anom_image_path + str(image_info) + ".png")
    
    def sample(self, image_losses):

        num_samples = 1000000
        losses = torch.flatten(image_losses)
        probabilities = torch.ones_like(losses) * 1/num_samples
        # Sample 3 elements with replacement
        sampled_indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=False)
        sampled_values = losses[sampled_indices]
        return sampled_values

    def plot_loss_distribution(self, image_losses, batch):
        sample_values = self.sample(image_losses)
        plt.figure()
        plt.ylim(0, 200)
        hist = sns.histplot(data=pd.DataFrame({'Loss': sample_values.numpy()}), x="Loss", kde = True, stat="density")
        plot_path = self.model_path + "plots/"
        os.makedirs(plot_path, exist_ok=True)
        hist_fig = hist.get_figure()
        hist_fig.savefig(plot_path + "batch-" + str(batch) + ".png") 

    def test(self, data_loader_test: DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss, global_step = int):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
    #     loss_reduced = None
    #     loss_mean = None
    #     loss_stddev = None
    #     loss_list = torch.tensor([]).to(device)
        
    #     label_list = torch.tensor([]).to(device)
        loss = None
        for idx, (data, labels, info_index) in enumerate(data_loader_test): 
            start = time.time()
            if torch.cuda.is_available():
                data = data.cuda()
                # labels = labels.to(device)
    # #             target = target.cuda()
            # data = torch.unsqueeze(data, dim = 1)
            with torch.no_grad():
                out = model(data)
                # seg, out = model(data)
                loss = criterion(out, data)
                self.save_anomalous_image(info_index, loss, labels, 50)
                loss_mean = torch.mean(loss)
                writer.add_scalar("Test Loss/Index", loss_mean, global_step = global_step) 
                global_step += 1
    #             loss_list = torch.cat((loss_list, loss), dim = 0)
    #             loss_reduced = torch.mean(loss)
    #             label_list = torch.cat((label_list, labels))
    # #         batch_acc = accuracy(out, target)
            losses.update(loss_mean, out.shape[0])
            self.plot_loss_distribution(loss, idx)

    # #         acc.update(batch_acc, out.shape[0])

    #         last_idx = 100    
    #         if idx % last_idx == 0:
    #             image_target = data[0][0] * data_max
    #             image_target = image_target.transpose(0,2)
    #             image_target = image_target.transpose(0,1)
    #             image_model = out[0][0]*data_max
    #             image_model = image_model.transpose(0,2)
    #             image_model = image_model.transpose(0,1)

    #         iter_time.update(time.time() - start)
    #         if idx % 16 == 0:
    #             print(('Iteration: [{0}/{1}]\t'
    #                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
    #                    'Loss {loss.val:.4E} ({loss.avg:.4E})\t'
    #                   )
    #                   .format(idx, len(data_loader_test), iter_time=iter_time, loss=losses))
    #             ##get first image of batch   
    #             image_target = data[0][0] * data_max
    #             image_target = image_target.transpose(0,2)
    #             image_target = image_target.transpose(0,1)
    #             image_model = out[0][0]*data_max
    #             image_model = image_model.transpose(0,2)
    #             image_model = image_model.transpose(0,1)
                
    #             save_images(image_target, metadata, k, "testing", "original", "original", idx)
    #             save_images(image_model, metadata, k, "testing", "reconstruction", "model", idx)
                
        
            
    #     threshold, overall_loss_df = histogram(loss_list, "Test", k)
    #     data_all_df = anomaly_histogram(loss_list, overall_loss_df, label_list, "Test", k)
    #     percent_matched, percent_falsematches, anomaly_masks = find_anomalies(loss_list, label_list, data_all_df, threshold, metadata, k, "Test")     
    #     return losses.avg, anomaly_masks

    def test_model(self):
        train_dataloader, val_dataloader, test_dataloader = load(self.batch_size, self.n_components)

        model = UNet2D(self.k, self.n_components).float()
        model.load_state_dict(torch.load(self.model_path + "model.pth", weights_only=True))
        criterion = nn.MSELoss(reduction = 'none')
        global_step_test = 0
        self.test(test_dataloader, model, criterion, global_step_test)
        writer.close()
        
        print("model weights loaded")
    #     loss, anomaly_masks = test(k, test_dataloader, model, criterion, max_factor, metadata)
def main():
    model_tester = TestModel()
    model_tester.test_model()
    
    
if __name__ == '__main__':
    main()
