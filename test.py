import pickle
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import spectral.io.envi as envi
from pathlib import Path
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch

# images = None
# info = None
# labels = None

# with open('./datasets/reduced_data/july21/images.pkl', 'rb') as f:
#     images = pickle.load(f)

# with open('./datasets/reduced_data/july21/info.pkl', 'rb') as f:
#     info = pickle.load(f)

# with open('./datasets/reduced_data/july21/labels.pkl', 'rb') as f:
#     labels = pickle.load(f)

# print(images.shape)
# print(info[0])

# gray_img = images[0,:,:,0]
# plt.imshow(gray_img, cmap='gray')
# plt.savefig("test0.png")

# hdr_file = Path(r'./datasets/apple_fireblight/july21/day_10/plant_1/REFLECTANCE_2090.hdr')
# data_file = Path(r'./datasets/apple_fireblight/july21/day_10/plant_1/REFLECTANCE_2090.dat')
# image = envi.open(hdr_file, data_file)
# save_rgb('rgb0.jpg', image, [70, 53, 19])

# rgb_img = cv2.imread("rgb0.jpg")
# label_list = np.array(labels[0])
# rgb_img[label_list[:,0],label_list[:,1]] = np.array([255,0,0])
# save_rgb('rgb0_labeled.jpg', rgb_img)

x = torch.randn((2, 7, 5, 4, 4))
x = torch.flatten(x, 1, 2)
softmax = nn.Softmax(dim = 1)
out = softmax(x)
x_labeled = torch.argmax(out, dim = 1)
print(x_labeled.shape)
print(x_labeled)

