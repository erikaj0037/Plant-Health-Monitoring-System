import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
# import torcharrow as ta
from torch.utils.tensorboard import SummaryWriter
# import seaborn as sns
# import pandas as pd

# import matplotlib.pyplot as plt
# from decimal import Decimal
# from spectral import *
# import spectral.io.envi as envi
# from pathlib import Path
import sys
from networks.wnet_model import WNet3D
from load_data import Loader
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary

writer = SummaryWriter()
# from format_segmentations import histogram, anomaly_histogram, find_anomalies, color_code_segmentations, color_code_anomalies

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# # def accuracy(output, target):
# #     """Computes the precision@k for the specified values of k"""
# #     batch_size = target.shape[0]

# #     _, pred = torch.max(output, dim=-1)

# #     correct = pred.eq(target).sum() * 1.0

# #     acc = correct.item() / batch_size

# #     return acc
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu"

def train(epoch: int, data_loader_train: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, criterion: nn.modules.loss._Loss, metadata = dict):
    
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    print("Training...")
    loss = None
    iters = len(data_loader_train)
    for idx, (data, labels, info) in enumerate(data_loader_train): 
        start = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
#             target = target.cuda()
#         data = torch.transpose(data, 2, 4)
#         data = torch.transpose(data, 3, 4)#.float().cuda()
        data = torch.unsqueeze(data, dim = 1)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data)
        loss = torch.mean(loss)
        writer.add_scalar("Loss/Index", loss, idx) 
#         writer.add_scalar("Loss/train", loss)
        loss.backward()
        optimizer.step()

#         batch_acc = accuracy(out, target)
        losses.update(loss, out.shape[0])
#         acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
   
        if idx % 50 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4E} ({loss.avg:.4E})\t'
                   'LR ({lr:.4E})\t')
#                    'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader_train), iter_time=iter_time, loss=losses, lr=scheduler.get_last_lr()[0]))#, top1=acc))
        # if idx % 100 == 0:
        #     image_target = data[0][0] * data_max
        #     image_target = image_target.transpose(0,2)
        #     image_target = image_target.transpose(0,1)
        #     image_model = out[0][0]*data_max
        #     image_model = image_model.transpose(0,2)
        #     image_model = image_model.transpose(0,1)
            
            
#             envi.save_image("images/training/header_files/epoch_" + str(epoch) + "target.hdr", image_target.detach().cpu().numpy(), metadata = metadata, force = True)
#             envi.save_image("images/training/header_files/epoch_" + str(epoch) + "model.hdr", image_model.detach().cpu().numpy(), metadata = metadata, force = True)
            
#             header_file_target = Path(r"images/training/header_files/epoch_" + str(epoch) + "target.hdr")
#             header_file_model = Path(r"images/training/header_files/epoch_" + str(epoch) + "model.hdr")
#             image_target = envi.open(header_file_target)
#             image_model = envi.open(header_file_model)
            
#             save_rgb("images/training/model_in_out/epoch" + str(epoch) + "_index" + str(idx) + "_target.png", image_target)
#             save_rgb("images/training/model_in_out/epoch" + str(epoch) + "_index" + str(idx) + "_model.png", image_model)
            # save_images(image_target, metadata, k, "training", "original", "original", idx, epoch)
            # save_images(image_model, metadata, k, "training", "reconstruction", "model", idx, epoch)
        
        scheduler.step()
    return loss
                
# def validate(epoch: int, k: int, data_loader_val: DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss, data_max: float, metadata = dict):
#     print("Validating...")
#     iter_time = AverageMeter()
#     losses = AverageMeter()
#     acc = AverageMeter()
#     loss_reduced = None
#     loss_mean = None
#     loss_stddev = None
#     loss_list = torch.tensor([]).to(device)
    
#     label_list = torch.tensor([]).to(device)
    
#     for idx, (data, labels, info) in enumerate(data_loader_val): 
#         start = time.time()
#         if torch.cuda.is_available():
#             data = data.to(device)
#             labels = labels.to(device)
# #             target = target.cuda()
#         data = torch.unsqueeze(data, dim = 1)
#         with torch.no_grad():
#             seg, out = model(data)
#             loss = criterion(out, data)
#             loss_list = torch.cat((loss_list, loss), dim = 0)
#             loss_reduced = torch.mean(loss)
#             label_list = torch.cat((label_list, labels))
# #         batch_acc = accuracy(out, target)
#         losses.update(loss_reduced, out.shape[0])
# #         acc.update(batch_acc, out.shape[0])

#         iter_time.update(time.time() - start)
#         if idx % 50 == 0:
#             print(('Epoch: [{0}][{1}/{2}]\t'
#                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
#                    'Loss {loss.val:.4E} ({loss.avg:.4E})\t'
#                       )
#                    .format(epoch, idx, len(data_loader_val), iter_time=iter_time, loss=losses))#, top1=acc))
#         last_idx = 100    
#         if idx % last_idx == 0:
#             image_target = data[0][0] * data_max
#             image_target = image_target.transpose(0,2)
#             image_target = image_target.transpose(0,1)
#             image_model = out[0][0]*data_max
#             image_model = image_model.transpose(0,2)
#             image_model = image_model.transpose(0,1)
            
            
# #             envi.save_image("images/validation/reconstruction/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_target.hdr", image_target.detach().cpu().numpy(), metadata = metadata, force = True)
# #             envi.save_image("images/validation/reconstruction/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_model.hdr", image_model.detach().cpu().numpy(), metadata = metadata, force = True)
            
# #             header_file_target = Path(r"images/validation/reconstruction/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_target.hdr")
# #             header_file_model = Path(r"images/validation/reconstruction/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_ model.hdr")
# #             image_target = envi.open(header_file_target)
# #             image_model = envi.open(header_file_model)
            
# #             save_rgb("images/validation/reconstruction/model_images/epoch" + str(epoch) + "_k" + str(k) + "_target.png", image_target)
# #             save_rgb("images/validation/reconstruction/model_images/epoch" + str(epoch) + "_k" + str(k) + "_model.png", image_model)
#             save_images(image_target, metadata, k, "validation", "original", "original", idx, epoch)
#             save_images(image_model, metadata, k, "validation", "reconstruction", "model", idx, epoch)
        
#     threshold = histogram(loss_list, k, epoch)
#     percent_matched, percent_falsematches, anomaly_masks = find_anomalies(loss_list, label_list, threshold, metadata, k, epoch)
#     return loss_reduced, percent_matched, percent_falsematches

# def test(k: int, data_loader_test: DataLoader, model: nn.Module, criterion: nn.modules.loss._Loss, data_max: float, metadata = dict):
    
#     iter_time = AverageMeter()
#     losses = AverageMeter()
#     acc = AverageMeter()
#     loss = None
#     loss_list = torch.tensor([]).to(device)
#     label_list = torch.tensor([]).to(device)
#     segmentation_list = torch.tensor([]).to(device)
#     for idx, (data, labels, info) in enumerate(data_loader_test): 
#         start = time.time()
#         if torch.cuda.is_available():
#             data = data.to(device)
#             labels = labels.to(device)
# #             target = target.cuda()
#         data = torch.unsqueeze(data, dim = 1)
#         with torch.no_grad():
#             seg, out = model(data)
#             loss = criterion(out, data)
#             loss_list = torch.cat((loss_list, loss), dim = 0)
#             loss_reduced = torch.mean(loss)
#             label_list = torch.cat((label_list, labels))
#             segmentation_list = torch.cat((segmentation_list, seg), dim = 0)
    
#         losses.update(loss_reduced, out.shape[0])

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

# def save_images(image, metadata, k, data_set: str, task: str, model_output_type: str, idx, epoch = -1):
#     envi.save_image("images/" + data_set + "/" + task + "/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_" + model_output_type + ".hdr", image.detach().cpu().numpy(), metadata = metadata, force = True)

#     header_file = Path(r"images/" + data_set + "/" + task + "/header_files/epoch_" + str(epoch) + "_k" + str(k) + "_" + model_output_type + ".hdr")

#     new_image = envi.open(header_file)
#     save_rgb("images/" + data_set + "/" + task + "/model_images/" + model_output_type + "_image" + str(idx) + "_epoch" + str(epoch) + "_k" + str(k) + ".png", new_image)
    
# def plot_results(dataset: str, epochs: int, loss_list: list, k: int):
#     plt.figure()
#     sns.set(style="white")
    
#     epoch_list = np.arange(epochs)
#     plt.figure()
#     sns.set(style="white")

#     for losses in loss_list:
#         data = pd.DataFrame({"Epoch": epoch_list, "Loss": losses})
#         sns.lineplot(data = data, x="Epoch", y="Loss", marker='o')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss per Epoch - " + str(dataset) + " (k = " + str(k) + ")")
#     plt.savefig("images/plots/" + str(dataset) + "-loss_per_epoch_k" + str(k) + ".png")

# def scale_data(data_train, data_val, data_test):
#     # get maxes of each and use .images
#     # compare maxes
#     print("normalizing...")    
#     max_factor = None
#     with open("max_intensity.txt", "r") as f:
#         max_factor = float(f.readline())
        
#     data_train.images = torch.divide(torch.from_numpy(np.array(data_train.images)), max_factor)
#     data_val.images = torch.divide(torch.from_numpy(np.array(data_val.images)), max_factor)
#     data_test.images = torch.divide(torch.from_numpy(np.array(data_test.images)), max_factor)
        
#     return data_train, data_val, data_test, max_factor
def custom_collate_fn(batch):
        # batch is a list of (image, label, info) tuples
        images, labels, info = zip(*batch)

        # Pad labels and ascii form of info to the maximum length in the current batch
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0) #padded twice to pad both dimensions
        temp = torch.transpose(labels_padded, 1, 2)
        temp_padded = pad_sequence(temp, batch_first=True, padding_value=0) 
        labels_padded = torch.transpose(temp_padded, 1, 2)
        print(labels_padded.shape)

        info_padded = pad_sequence(info, batch_first=True, padding_value=0)

        # Stack images
        images_stacked = torch.stack(images)
        
        return images_stacked, labels_padded, info_padded

def load(batch_size):
    loader = Loader()
    data_train, data_val, data_test, metadata, umap_parameters = loader.load_data()
    train_dataloader = DataLoader(data_train, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True, shuffle=False)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True, shuffle=False) 
    test_dataloader = DataLoader(data_test, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True, shuffle=False) 
    return train_dataloader, val_dataloader, test_dataloader, metadata, umap_parameters

def run_model():
    batch_size = 8
    train_dataloader, val_dataloader, test_dataloader, metadata, umap_parameters = load(batch_size)

    k = 7 # number of labels of data for semantic segmentation
    model = WNet3D(k).float()
    print("summary")
    if torch.cuda.is_available():
        model = model.cuda()
    print(summary(model, (1, 10, 512, 512)))

    criterion = nn.MSELoss(reduction = 'none')
    
#     optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4,
#                                 momentum=0.99,
#                                 weight_decay=1e-4)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 768, T_mult=1, eta_min=1e-8)
    epochs = 10
    for epoch in range(epochs):
        train_loss = train(epoch, train_dataloader, model, optimizer, scheduler, criterion, metadata)
    writer.close()
    torch.save(model.state_dict(), "model_weights/model_batchsize" + str(batch_size) + "_k" + str(k) + "_umap[" + str(umap_parameters[0]) + "," + str(umap_parameters[1]) + "].pth")

        
#         ##training
#         print("training model...")
#         loss_list_train = []
#         loss_list_val = []
#         match_list = []
#         for epoch in range(epochs):

            # train_loss = train(epoch, train_dataloader, model, optimizer, scheduler, criterion, max_factor, metadata)
#             loss_list_train.append(float(train_loss.cpu().detach().numpy().astype(float)))
            
# #             print(loss_list_train)
#             val_loss, percent_matched, percent_falsematches = validate(epoch, k, val_dataloader, model, criterion, max_factor, metadata)
#             match_list.append([percent_matched, percent_falsematches])
        
# #             writer.add_scalar("Loss/validate", final_loss)
#             loss_list_val.append(float(val_loss.cpu().detach().numpy().astype(float)))
            
#         torch.save(model.state_dict(), "checkpoints/train_model_k" + str(k) + ".pth")
#         with open("/home/hice1/ejones338/scratch/data_files/model_runs/training/losses_train_k" + str(k) + ".npy", 'wb') as f:
#             np.save(f, np.array(loss_list_train))
#         with open("/home/hice1/ejones338/scratch/data_files/model_runs/training/losses_val_k" + str(k) + ".npy", 'wb') as f:
#             np.save(f, np.array(loss_list_val))
#         with open("/home/hice1/ejones338/scratch/data_files/model_runs/training/matches_k" + str(k) + ".npy", 'wb') as f:
#             np.save(f, np.array(match_list.cpu()))
            
#         plot_results('Training', epochs, loss_list_train, k)
#         plot_results('Validation', epochs, loss_list_val, k)
        
    
#         loss_list_train_val = np.vstack((np.array(loss_list_train), np.array(loss_list_val)))
#         plot_results('training_validation', epochs, list(loss_list_train_val), k)
#     print("testing model...")
    # model.load_state_dict(torch.load("checkpoints/train_model_k" + str(k) + ".pth", weights_only=True))
    
#     print("model weights loaded")
#     loss, anomaly_masks = test(k, test_dataloader, model, criterion, max_factor, metadata)
#     with open("/home/hice1/ejones338/scratch/data_files/model_runs/testing/anomaly_masks_k" + str(k) + ".npy", 'wb') as f:
#         np.save(f, np.array(anomaly_masks.cpu()))
    
#     return loss
    

def main():
    run_model()
    
    
if __name__ == '__main__':
    main()
