import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models.resnet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from collections import namedtuple, deque
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


class Identity(nn.Module):
    """
    Class for nn.Module that returns values from previous layer. Used to replace last fc in resnet.
    """
    def __init__(self):
        super(Identity, self).__init__()

    # return unchanged input during forward pass
    def forward(self, x):
        return x


class DQN(nn.Module):
    """
    Class for neural network. The network consists of a resnet backbone and a fully connected layer containing Q-values
    for each action.
    """

    def __init__(self, n_actions, device, weights=None):
        super(DQN, self).__init__()
        self.device = device

        # resnet backbone, fc replaced by layer that returns previous activations
        self.resnet18 = torchvision.models.resnet.resnet18()
        self.resnet18.fc = Identity()

        # layer with Q-values
        self.fc = nn.Linear(512, n_actions)

        # load pretrained weghts
        if weights == "pretrained":
            #     for param in self.resnet18.parameters():
            #         param.required_grad = False
            self.resnet18.load_state_dict(torch.load('models/resnet18_pretrained.pt'))

        # pass network to device
        self.resnet18.to(device=device)

    # forward pass
    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x


class DQN_pretrain(nn.Module):
    """
    Neural network class used to pretrain backbone. Consists of resnet backbone an fully connected layers that
    predict total time, flight time, poistion and inclination.
    """

    def __init__(self):
        super(DQN_pretrain, self).__init__()

        # resnet backbone, fc replaced by layer that returns previous activations
        self.resnet18 = torchvision.models.resnet.resnet18()
        self.resnet18.fc = Identity()

        # fully connected output layers
        self.fc_total_time = nn.Linear(512, 1)
        self.fc_flight_time = nn.Linear(512, 1)
        self.fc_position = nn.Linear(512, 5)
        self.fc_inclination = nn.Linear(512, 1)

    # forward pass
    def forward(self, x):

        # pass through backbone
        x = self.resnet18(x)

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        # get output values
        total_time = self.fc_total_time(x)
        flight_time = self.fc_flight_time(x)
        position = F.softmax(self.fc_position(x), dim=1)
        inclination = self.fc_inclination(x)

        return total_time, flight_time, position, inclination


class DigitsDataset(Dataset):
    """
    Class for the dataset used to train points classifier.
    """
    def __init__(self, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
                 dataset_folder='digits_dataset/'):
        self.transform = transform
        self.dataset_folder = dataset_folder

    # get length of the dataset
    def __len__(self):
        return len(os.listdir(self.dataset_folder))

    # get image and corresponding label
    def __getitem__(self, idx):
        img = Image.open(f'{self.dataset_folder}/{idx}.png')
        label = idx
        img = self.transform(img)

        return img, label


class PretrainDataset(Dataset):
    """
    Class for the dataset used to pretrain backbone.
    """
    def __init__(self, transform=transforms.Compose([transforms.ToTensor()]),
                 dataset_folder='pretrain_dataset/data/', labels_path='pretrain_dataset/labels.csv',
                 input_shape=(200, 116)):

        # transforms for image preprocessing
        self.transform = transform

        self.dataset_folder = dataset_folder

        # image list
        self.img_list = [int(x[:-4]) for x in os.listdir(self.dataset_folder)]

        # load labels from csv file
        self.labels = pd.read_csv(labels_path)

        self.input_shape = input_shape

    # get length of the dataset
    def __len__(self):
        return len(self.img_list)

    # get image and corresponding labels
    def __getitem__(self, idx):

        # open and preprocess image
        img_name = str(self.img_list[idx]).zfill(6) + '.png'
        img = Image.open(f'{self.dataset_folder}/{img_name}')
        img = img.resize(self.input_shape)
        img = np.asarray(img) / 255.0
        img = img.astype(np.float32)
        img = self.transform(img)

        # read labels
        total_time = self.labels.loc[self.labels["img"] == img_name, "labels_total_time"].values[0].astype(
            np.float32) / 10
        flight_time = self.labels.loc[self.labels["img"] == img_name, "labels_flight_time"].values[0].astype(
            np.float32) / 10
        position = self.labels.loc[self.labels["img"] == img_name, "labels_position"].values[0].astype(np.float32)
        inclination = self.labels.loc[self.labels["img"] == img_name, "labels_inclination"].values[0].astype(
            np.float32) / 100

        return img, [total_time.astype(np.float32), flight_time.astype(np.float32), position.astype(np.float32),
                     inclination.astype(np.float32)]


class DigitsClassifier(nn.Module):
    """
    A class for a neural network that recognizes digits in scores.
    """
    def __init__(self, path='models/', filename='digits_classifier.pt', source='train'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding='same')
        self.fc1 = nn.Linear(462, 12)

    # forward pass
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        # outputs
        x = F.softmax(self.fc1(x), dim=1)

        return x


def train_digits_classifier():
    """
    Function in which digits classifier training is performed. A given digit always looks the same, and is in the same
    place, so there is no need to create a validation set.
    :return:
    """

    # initialize digits dataset
    dataset = DigitsDataset()

    # initialize digits dataloader
    dataloader = DataLoader(dataset, batch_size=dataset.__len__())

    # initialize neural network
    model = DigitsClassifier()

    # pass neural network to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # specify loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # training loop
    for epoch in range(1500):

        # get batch from dataloader
        for i, data in enumerate(dataloader, 0):

            # split the data into input and output
            inputs, labels = data

            # pass inputs and outputs to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate loss value
            loss = criterion(outputs, labels)
            print(f"Loss: {loss.item()}")

            # perform backpropagation
            loss.backward()

            # perform optimizer step
            optimizer.step()

    # save trained model
    torch.save(model.state_dict(), f'models/digits_classifier.pt')
    print('Finished Training')


def pretrain_resnet18(epochs=10, batch_size=32, train_valid_ratio=0.85):
    """
    Function in which pretraining of the resnet18 backbone is performed. Neural network predicts total time elapsed.
    The neural network predicts the time that has elapsed since the start of the jump, the time since the jump,
    the position, i.e. the run-up, jump, flight and landing, and the inclination of the jumper in flight
    :return:
    """

    # initialize dataset
    dataset = PretrainDataset()

    # calculate length of training ang validation datasets
    train_length = int(dataset.__len__() * train_valid_ratio)
    val_length = dataset.__len__() - train_length

    # split dataset into training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_length, val_length])

    # initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # initialize neural network
    model = DQN_pretrain()

    # pass neural network to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # specify loss functions for every output layer
    criterion_total_time = nn.MSELoss()
    criterion_flight_time = nn.MSELoss()
    criterion_position = nn.CrossEntropyLoss()
    criterion_inclination = nn.MSELoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # create lists where loss values will be stored
    train_loss_list = []
    val_loss_list = []

    # training loop
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        # train
        for i, data in enumerate(train_dataloader, 0):

            # split data into inputs and outputs
            inputs, labels = data

            # pass inputs device
            inputs = inputs.to(device)

            # pass all outputs to device, change position label type to int
            labels = [x.to(device) for x in labels]
            labels[2] = labels[2].type(torch.LongTensor)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate loss values for all output layers
            loss_1 = criterion_total_time(outputs[0].squeeze(), labels[0])
            loss_2 = criterion_flight_time(outputs[1].squeeze(), labels[1])
            loss_3 = criterion_position(outputs[2], labels[2])
            loss_4 = criterion_inclination(outputs[3].squeeze(), labels[3])

            # sum up loss values
            loss = loss_1 + loss_2 + loss_3 + loss_4

            # append loss to loss list
            train_loss_list.append(loss.item())

            # print loss value
            # print(f"epoch: {epoch}, iteration: {i}, loss: {loss.item()}")

            # perform backpropagation
            loss.backward()

            # perform optimizer step
            optimizer.step()

        # validate
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):

                # split the data into inputs and outputs
                inputs, labels = data

                # pass inputs device
                inputs = inputs.to(device)

                # pass all outputs to device, change position label type to int
                labels = [x.to(device) for x in labels]
                labels[2] = labels[2].type(torch.LongTensor)

                # forward pass
                outputs = model(inputs)

                # calculate loss values for all output layers
                loss_1 = criterion_total_time(outputs[0].squeeze(), labels[0])
                loss_2 = criterion_flight_time(outputs[1].squeeze(), labels[1])
                loss_3 = criterion_position(outputs[2], labels[2])
                loss_4 = criterion_inclination(outputs[3].squeeze(), labels[3])

                # sum up loss values
                loss = loss_1 + loss_2 + loss_3 + loss_4

                # print loss value
                # print(f"Validation, epoch: {epoch}, iteration: {i}, loss: {loss.item()}")

                # sum up loss value with previous values to calculate average later
                total_loss += loss.item()

        # calculate average loss value for the entire validation dataset
        val_loss = total_loss / math.ceil(val_length / batch_size)

        # print average loss and append to loss list
        # print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
        val_loss_list.append(val_loss)

    # save the model
    torch.save(model.state_dict(), f'models/resnet18_pretrained_all.pt')

    # plot training and validation loss
    visualize_training(train_loss_list, val_loss_list, no_epochs=epochs, dataset_length=train_length,
                       batch_size=batch_size)

    print('Finished Training')


def visualize_training(train_loss, val_loss, no_epochs, dataset_length, batch_size):
    """
    A function used to visualize training progress. Creates a graph of the loss.
    :param train_loss: list of training loss values
    :param val_loss: list of validation loss values
    :param no_epochs: number of epochs that the neural network was trained for
    :param dataset_length: training dataset length
    :param batch_size: batch size
    :return:
    """

    # create array with x values for the training loss plot
    x1 = np.arange(0, len(train_loss), 1)

    # create array with x values for the validation loss plot
    x2_start = math.ceil(dataset_length / batch_size)
    x2_stop = (no_epochs + 1) * math.ceil(dataset_length / batch_size)
    x2_step = math.ceil(dataset_length / batch_size)
    x2 = np.arange(x2_start, x2_stop, x2_step)

    # specify plot size
    plt.figure(figsize=(8, 4))

    # plot training loss
    plt.plot(x1, train_loss, label='Training loss')

    # plot validation loss
    plt.plot(x2, val_loss, label='Validation loss')

    # add axes labels
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # add title
    plt.title('Loss function')

    # add grid
    plt.grid()

    # add legend
    plt.legend()

    # limit y axis
    plt.ylim([0, 4])

    # save plot
    plt.savefig("results/training_and_valid_loss.png")

    # show plot
    # plt.show()


class PointsClassifier:
    """
    Class that classifies points based on a neural network that classifies single digits. Neural network returns if there
    is an empty space, sign or digit and this class merges its outputs to points value.
    """
    def __init__(self,
                 points_digit_1_bbox=(0, 0, 7, 11),
                 points_digit_2_bbox=(6, 0, 13, 11),
                 points_digit_3_bbox=(12, 0, 19, 11),
                 points_digit_4_bbox=(18, 0, 25, 11),
                 points_digit_5_bbox=(30, 0, 37, 11)):
        """
        Initialize points classifier
        :param points_digit_1_bbox: coordinates of the bounding box with the first digit
        :param points_digit_2_bbox: coordinates of the bounding box with the second digit
        :param points_digit_3_bbox: coordinates of the bounding box with the third digit
        :param points_digit_4_bbox: coordinates of the bounding box with the fourth digit
        :param points_digit_5_bbox: coordinates of the bounding box with the fifth digit
        """

        self.points_digit_1_bbox = points_digit_1_bbox
        self.points_digit_2_bbox = points_digit_2_bbox
        self.points_digit_3_bbox = points_digit_3_bbox
        self.points_digit_4_bbox = points_digit_4_bbox
        self.points_digit_5_bbox = points_digit_5_bbox

        # initialize neural network that classifies digits
        self.neural_network = DigitsClassifier()

        # load weights
        self.neural_network.load_state_dict(torch.load('models/digits_classifier.pt'))

        # pass neural network to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_network.to(device=self.device)

        # change mode to evaluation
        self.neural_network.eval()

        # specify labels
        self.labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ' ']

    def get_points(self, img):
        """
        Returns points value based on a given image.
        :param img: image of points
        :return: points value
        """

        # create a list where single digits will be stored
        digits = []
        points = 0

        # get the image of the first digit from the image of the points and transform
        img_digit_1 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(
            img.crop(self.points_digit_1_bbox)).unsqueeze(0).to(device=self.device)

        # predict first digit and append to list
        digits.append(int(torch.argmax(self.neural_network(img_digit_1))))

        # get the image of the second digit from the image of the points and transform
        img_digit_2 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(
            img.crop(self.points_digit_2_bbox)).unsqueeze(0).to(device=self.device)

        # predict second digit and append to list
        digits.append(int(torch.argmax(self.neural_network(img_digit_2))))

        # get the image of the third digit from the image of the points and transform
        img_digit_3 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(
            img.crop(self.points_digit_3_bbox)).unsqueeze(0).to(device=self.device)

        # predict third digit and append to list
        digits.append(int(torch.argmax(self.neural_network(img_digit_3))))

        # get the image of the fourth digit from the image of the points and transform
        img_digit_4 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(
            img.crop(self.points_digit_4_bbox)).unsqueeze(0).to(device=self.device)

        # predict fourth digit and append to list
        digits.append(int(torch.argmax(self.neural_network(img_digit_4))))

        # get the image of the fifth digit from the image of the points and transform
        img_digit_5 = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])(
            img.crop(self.points_digit_5_bbox)).unsqueeze(0).to(device=self.device)

        # predict fifth digit and append to list
        digits.append(int(torch.argmax(self.neural_network(img_digit_5))))

        # merge digits
        for i in range(4, -1, -1):
            if digits[i] < 10:
                points += digits[i] * 10 ** (4 - i)
            elif digits[i] == 10:
                points *= -1
            else:
                break

        # divide points by 10
        if digits.count(11) == 5:
            points = None
        else:
            points /= 10

        return points


# main function
if __name__ == '__main__':
    # train_digits_classifier()
    dqn = DQN_pretrain()
    pretrain_resnet18(epochs=5)
