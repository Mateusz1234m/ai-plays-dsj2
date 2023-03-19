import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.resnet
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from collections import namedtuple, deque
import math
import random
import acquisition_module
import sys
import neural_network_module
import pandas as pd

df = pd.read_csv("pretrain_dataset/labels.csv")

print(df.head().to_markdown())
print(df.tail().to_markdown())

sys.exit()

# don't remove
data_grabber = acquisition_module.DataGrabber()
points_detector = neural_network_module.PointsClassifier()
img = data_grabber.get_points_img()
img.show()
points = points_detector.get_points(img)
print(points)
#####
# MAX_GAMES = 100
# for n_games in range(MAX_GAMES):
#     epsilon = []
#     for iteration in range(40):
#         epsilon.append((2 * (((MAX_GAMES - n_games) / MAX_GAMES) - 0.5) + (iteration / 20)) * MAX_GAMES)
#     print(f"{n_games}: {list(np.around(np.array(epsilon),2))}")

# print(torch.cuda.is_available())

# import usb.core
# import usb.util
#
# print(list(usb.core.find(find_all=True)))
#
# # Find the USB device that represents the mouse
# dev = usb.core.find(idVendor=0x046d, idProduct=0xc077)
# if dev is None:
#     sys.exit("Could not find Logitech USB mouse.")
# # Set the configuration of the device
# dev.set_configuration()
#
# # Get the endpoint for the mouse
# endpoint = dev[0][(0,0)][0]
#
# # Read data from the endpoint
# while True:
#     try:
#         data = dev.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize)
#         x = data[1]
#         y = data[2]
#         print(f"Mouse movement: X={x}, Y={y}")
#     except usb.core.USBError as e:
#         if e.args == ('Operation timed out',):
#             continue

# import numpy as np
#
# arr = np.array([[1, 2]])
# print(arr.shape)
# print(arr.ndim)
#
# arr = np.squeeze(arr)
# print(arr.shape)
# print(arr.ndim)