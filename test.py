import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# # CNTK - BGR, CHW
# sample_cntk = np.zeros((32, 32, 3), dtype=float)
#
# with open('convInput_cntk.txt', 'r') as f:
#     data = f.readline().split(' ')
#     # print(len(data))
#     cnt = 0
#     for i in range(2, -1, -1):
#         for j in range(32):
#             for k in range(32):
#                 sample_cntk[j, k, i] = data[cnt]
#                 cnt += 1
# print('Read CNTK input.')
# plt.imshow(sample_cntk)
# plt.show()
#
# # PyTorch - RGB, CHW
# sample_pytorch = np.zeros((32, 32, 3), dtype=float)
#
# with open('convInput_pytorch.txt', 'r') as f:
#     data = f.readline().split(' ')
#     cnt = 0
#     for i in range(2, -1, -1):
#     # for i in range(3):
#         for j in range(32):
#             for k in range(32):
#                 sample_pytorch[j, k, i] = data[cnt]
#                 cnt += 1
#     # print(cnt)
# print('Read PyTorch input.')
# plt.imshow(sample_pytorch)
# plt.show()
#
# print('Comparing...')
# cnt = 0
# for i in range(32):
#     for j in range(32):
#         for k in range(3):
#             if math.fabs(sample_cntk[i, j, k] - sample_pytorch[i, j, k]) > 1e-4:
#                 print('(%d, %d, %d)' % (i, j, k))
#             else:
#                 cnt += 1
# print('Same: %d' % cnt)

# image = Image.open('raw.png')
# # print(type(image))
# arr = np.array(image)
# arr = arr[4:36,4:36,:]
# print(arr.shape)
# with open('raw.txt', 'w') as f:
#     for i in range(3):
#         for j in range(32):
#             for k in range(32):
#                 f.write(str(arr[j, k, i]))
#                 f.write(' ')
#     # f.write('\n')

# x = np.array([[1, 2, 3],
#               [4, 5, 6]], dtype=float)
# x = torch.from_numpy(x)
# print(x)
# print(x.shape)
# x_norm = torch.norm(x, dim=0, keepdim=True)
# print(x_norm)
# print(x_norm.shape)
# x = x / x_norm
# print(x)
# print(x.shape)

a = float(0.1333333333333542131231231231231245464364333)
print(str(a))
