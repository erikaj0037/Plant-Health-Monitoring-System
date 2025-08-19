import pickle
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
import spectral.io.envi as envi
from pathlib import Path
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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

# x = torch.randn((2, 7, 5, 4, 4))
# x = torch.flatten(x, 1, 2)
# softmax = nn.Softmax(dim = 1)
# out = softmax(x)
# x_labeled = torch.argmax(out, dim = 1)
# print(x_labeled.shape)
# print(x_labeled)

# labels = tuple([torch.tensor([[0,0]]), torch.tensor([[4,3],[7,2]]), torch.tensor([0]), torch.tensor([[4,3],[7,2], [0,0]])])
# labels_padded = pad_sequence(labels, batch_first=True, padding_value=0) #padded twice to pad both dimensions
# print(labels_padded)
# temp = torch.transpose(labels_padded, 1, 2)
# temp_padded = pad_sequence(temp, batch_first=True, padding_value=0) 
# labels_padded = torch.transpose(temp_padded, 1, 2)
# print(labels_padded)

# x = np.array([[2, 3], [5,2], [3,4]])
# y = np.array([[9, 8], [7,2], [9,0]])

# with open('test.npy', 'wb') as f:
#     np.save(f, x)
#     np.save(f, y)

# with open('test.npy', 'rb') as f:
#     x = np.load('test.npy', mmap_mode='r')
#     print(x)
#     print(str(sys.getsizeof(x) / 1000000000) + " GB")
#     mean = np.mean(x)
#     print("x mean: " + str(mean))

#     y = np.load('test.npy', mmap_mode='r')
#     print(y)
#     print(str(sys.getsizeof(y) / 1000000000) + " GB")
#     print("y mean: " + str(mean))

#     z = np.array([x, y])
#     print(str(sys.getsizeof(z) / 1000000000) + " GB")
#     print("z mean: " + str(mean))

# with open('test.npy', 'rb') as f:
#     x = np.load('test.npy')
#     print(str(sys.getsizeof(x) / 1000000000) + " GB")
#     mean = np.mean(x)
#     print("x mean: " + str(mean))

#     y = np.load('test.npy')
#     print(str(sys.getsizeof(y) / 1000000000) + " GB")
#     print("y mean: " + str(mean))

#     z = np.array([x, y])
#     print(str(sys.getsizeof(z) / 1000000000) + " GB")
#     print("z mean: " + str(mean))

# a=np.array([[1, 2, 3], [4, 5, 6]])
# b=np.array([1, 2])
# with open('tmp/test.npz', 'wb') as f:
#     np.savez('tmp/123.npz', a=a)
#     np.savez('tmp/123.npz', b=b)
#     data = np.load('tmp/123.npz')
#     # data['a']
#     print(data['b'])
    
#     data.close()