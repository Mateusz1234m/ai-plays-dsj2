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
import numpy as np
import sys


class DQN(nn.Module):
    """
    Class for neural network
    """
    def __init__(self, n_actions, device):
        super(DQN, self).__init__()
        self.device = device

        self.resnet18 = torchvision.models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = nn.Linear(512, n_actions)
        self.resnet18.to(device=device)

    def forward(self, x):
        # don't use softmax, because it will make Q values < 1
        # x = F.softmax(self.resnet18(x), dim=1)
        x = self.resnet18(x)
        return x




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DigitsDataset(Dataset):
    def __init__(self, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
                 dataset_folder='digits_dataset/'):
        self.transform = transform
        self.dataset_folder = dataset_folder

    def __len__(self):
        return len(os.listdir(self.dataset_folder))

    def __getitem__(self, idx):
        img = Image.open(f'{self.dataset_folder}/{idx}.png')
        label = idx
        img = self.transform(img)

        return img, label


class DigitsClassifier(nn.Module):
    def __init__(self, path='models/', filename='digits_classifier.pt', source='train'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding='same')
        self.fc1 = nn.Linear(924, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)

        return x

    def save(self):
        pass


def train():
    dataset = DigitsDataset()

    dataloader = DataLoader(dataset, batch_size=dataset.__len__())
    model = DigitsClassifier()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):  # loop over the dataset multiple times

        for i, data in enumerate(dataloader, 0):
            print(
                i)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            print(f'Pred: {outputs.argmax(dim=1)}, True: {labels}')
            # for output in outputs:
            #     print(output)
            # print()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # print(f'Loss: {loss.item()}')

    torch.save(model.state_dict(), f'models/digits_classifier.pt')
    print('Finished Training')

class PointsClassifier():
    def __init__(self,
                 points_digit_1_bbox = (0, 0, 7, 11),
        points_digit_2_bbox = (6, 0, 13, 11),
        points_digit_3_bbox = (12, 0, 19, 11),
        points_digit_4_bbox = (18, 0, 25, 11),
        points_digit_5_bbox = (30, 0, 37, 11)):

        self.points_digit_1_bbox = points_digit_1_bbox
        self.points_digit_2_bbox = points_digit_2_bbox
        self.points_digit_3_bbox = points_digit_3_bbox
        self.points_digit_4_bbox = points_digit_4_bbox
        self.points_digit_5_bbox = points_digit_5_bbox
        self.neural_network = DigitsClassifier()
        self.neural_network.load_state_dict(torch.load('models/digits_classifier.pt'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_network.to(device=self.device)
        self.neural_network.eval()
        self.labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ' ']

    def get_points(self, img):
        digits = []
        points = 0


        # img.crop(self.points_digit_1_bbox).show()


        img_digit_1 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(img.crop(self.points_digit_1_bbox)).unsqueeze(0).to(device=self.device)
        # print(img_digit_1.shape)
        digits.append(int(torch.argmax(self.neural_network(img_digit_1))))
        # print(self.neural_network(img_digit_1))
        img_digit_2 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(img.crop(self.points_digit_2_bbox)).unsqueeze(0).to(device=self.device)
        digits.append(int(torch.argmax(self.neural_network(img_digit_2))))

        img_digit_3 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(img.crop(self.points_digit_3_bbox)).unsqueeze(0).to(device=self.device)
        digits.append(int(torch.argmax(self.neural_network(img_digit_3))))

        img_digit_4 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(img.crop(self.points_digit_4_bbox)).unsqueeze(0).to(device=self.device)
        digits.append(int(torch.argmax(self.neural_network(img_digit_4))))

        img_digit_5 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(img.crop(self.points_digit_5_bbox)).unsqueeze(0).to(device=self.device)
        digits.append(int(torch.argmax(self.neural_network(img_digit_5))))
        # print(digits)
        for i in range(4, -1, -1):
            if digits[i] < 10:
                points += digits[i] * 10**(4-i)
            elif digits[i] == 10:
                points *= -1
            else:
                break

        if digits.count(11) == 5:
            points = None
        else:
            points /= 10

        return points


if __name__ == '__main__':
    train()

