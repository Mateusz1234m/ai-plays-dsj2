import os
import time

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

# don't remove
# data_grabber = acquisition_module.DataGrabber()
# points_detector = neural_network_module.PointsClassifier()
# img = data_grabber.get_points_img()
# points = points_detector.get_points(img)
# print(points)
#####


print(torch.cuda.is_available())