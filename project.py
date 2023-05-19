import argparse
import json
import os.path
import pickle
import sys
import time

import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import LambdaLR, WarmLR
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from timm.data.auto_augment import rand_augment_transform

import torchvision
import torchvision.transforms as transforms
import numpy as np
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random

MODEL_FILENAME = "baseline_model.pth"
PICKLE_FILENAME = "data.pickle"
PLOT_FILENAME = "baseline.pdf"

torch.manual_seed(30)
random.seed(30)
np.random.seed(30)


class BaselineModel(nn.Module):
    def __init__(self, dropout=0, increased_dropout=0, batch_norm=False, num_classes=10):
        super(BaselineModel, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.block1.apply(init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout + increased_dropout)
        )
        self.block2.apply(init_weights)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout + increased_dropout * 2)
        )
        self.block3.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Dropout(dropout + increased_dropout * 3),
            nn.Linear(128, num_classes)
        )
        self.fc.apply(init_weights)

        self.to(device)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # x4 = x3.view(-1, 64 * 8 * 8)
        current_batch_size = x3.size(dim=0)
        x4 = x3.reshape((current_batch_size, -1))

        x5 = self.fc(x4)

        return x5


class BaselineModelModifiedBNDropoutOrder(nn.Module):
    def __init__(self, dropout=0, increased_dropout=0, batch_norm=False, num_classes=10):
        super(BaselineModelModifiedBNDropoutOrder, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1.apply(init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout((dropout + increased_dropout) / 2),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout((dropout + increased_dropout) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2.apply(init_weights)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout((dropout + increased_dropout * 2) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout((dropout + increased_dropout * 2) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(dropout + increased_dropout * 3),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Linear(128, num_classes)
        )
        self.fc.apply(init_weights)

        self.to(device)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        current_batch_size = x3.size(dim=0)
        x4 = x3.reshape((current_batch_size, -1))
        x5 = self.fc(x4)

        return x5


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=initial_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x2 += self.k_l * self.skip_connection(x)
        return self.relu(x2)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()

        out_channels = in_channels // reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class SqueezeExcitationBlockFullBasic(nn.Module):
    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(SqueezeExcitationBlockFullBasic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=initial_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se = SqueezeExcitationBlock(out_channels, None, reduction=16)

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y += self.k_l * self.skip_connection(x)
        return self.relu(y)
    
    
class SqueezeExcitationBlockFullBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(SqueezeExcitationBlockFullBottleneck, self).__init__()

        reduced_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=initial_stride, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se = SqueezeExcitationBlock(out_channels, None, reduction=16)

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
    
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.se(y)

        y += self.k_l * self.skip_connection(x)
        return self.relu(y)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(BottleneckBlock, self).__init__()

        reduced_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=initial_stride, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x3 += self.skip_connection(x)
        return self.relu(x3)


class ResNetModel(nn.Module):

    def __init__(self, block=ResNetBlock, num_classes=10, k_list=[1.]*16, dropout=0):
        super(ResNetModel, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = self._create_block(block, 64, 64, 3, initial_stride=1, k_l=k_list[:3])
        self.block2 = self._create_block(block, 64, 128, 4, initial_stride=2, k_l=k_list[3:7])
        self.block3 = self._create_block(block, 128, 256, 6, initial_stride=2, k_l=k_list[7:13])
        self.block4 = self._create_block(block, 256, 512, 3, initial_stride=2, k_l=k_list[13:])

        self.average_pool = nn.AvgPool2d(kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.fc = nn.Linear(512, num_classes)

        for entry in [self.initial_block, self.block1, self.block2, self.block3, self.block4, self.fc]:
            entry.apply(init_weights)

        self.to(device)

    def _create_block(self, block, in_channels, out_channels, num_blocks, initial_stride, k_l):
        layers = [block(in_channels, out_channels, initial_stride, k_l[0])]

        for index in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1, k_l[index]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.initial_block(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        x6 = self.average_pool(x5)
        x7 = self.dropout(x6)

        current_batch_size = x7.size(dim=0)
        x7 = x7.reshape((current_batch_size, -1))
        x8 = self.fc(x7)

        return x8


def plot_curves(ax, train, val, name):
    ax.set_title(name)
    ax.plot(train, color='blue', label='Training ' + name)
    ax.plot(val, color='orange', label='Validation ' + name)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(name)
    ax.legend()


def summarize_diagnostics(plotpath, history):
    fig, axs = plt.subplots(2, 1)

    plot_curves(axs[0], history['train_loss'], history['val_loss'], 'Loss')
    plot_curves(axs[1], history['train_acc'], history['val_acc'], 'Accuracy')

    plt.tight_layout()
    plt.savefig(plotpath)
    plt.close()


def compute_metrics(model, dataloader, loss_function=nn.CrossEntropyLoss()):
    running_loss = 0

    correct = 0
    total = 0

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    num_batches = len(dataloader.dataset) // batch_size

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # images, labels = trainloader.dataset.dataset.data, trainloader.dataset.dataset.targets
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)
            current_loss = loss_function(outputs, labels)
            running_loss += current_loss.detach().item()

            _, predicted = torch.max(outputs.data, 1)

            # the class with the highest energy is what we choose as prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    network_loss = running_loss / num_batches
    network_accuracy = (float(correct) / total) * 100

    # print(f'Accuracy of the network on the test images: {accuracy} %')

    accuracy_per_class = {}

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        else:
            accuracy = 0

        accuracy_per_class[classname] = accuracy
        # print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return network_loss, network_accuracy, accuracy_per_class


class PickleHelper:
    def __init__(self, filename):
        self.pickled = {}
        self.filename = filename
        self.has_loaded = False

        if os.path.exists(filename):
            self.load()
            self.has_loaded = True

    def update(self, key, value):
        self.pickled[key] = value

    def register(self, key, value):
        if self.has_loaded:
            if key in self.pickled:
                return self.pickled[key]
            else:
                sys.stderr.write(f"{key} not in dict, returning value instead\n")
                return value
        self.update(key, value)
        return value

    def save(self, path=None):
        if path is None:
            path = self.filename
        with open(path, "wb") as file:
            pickle.dump(self.pickled, file, 2)

    def load(self):
        with open(self.filename, "rb") as file:
            self.pickled = pickle.load(file)


def folder_helper(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    nested_path = os.path.join(folderpath, time_str)
    if not os.path.exists(nested_path):
        os.makedirs(nested_path)

    return nested_path


def get_path(folderpath, filename, epoch):
    epoch_path = os.path.join(folderpath, str(epoch))
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    return os.path.join(epoch_path, filename)


def load_data(apply_augmentation=False, apply_random_aug=False, norm_m0_sd1=False, dataset=torchvision.datasets.CIFAR10):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trainset = dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    classes = trainset.classes

    
    if apply_augmentation:
        if apply_random_aug:
            print("Using random augmentation")
            train_mean = trainset.data.mean(axis=(0, 1, 2))

            transform_norm_aug = transforms.Compose([
                rand_augment_transform(
                config_str='rand-m9-mstd0.5', 
                hparams={'img_mean': (int(train_mean[0]), int(train_mean[1]), int(train_mean[2]))}),
                transforms.ToTensor()
            ])
        elif norm_m0_sd1: # mean0 std1
            train_mean = trainset.data.mean(axis=(0, 1, 2)) / 255
            train_std = trainset.data.std(axis=(0, 1, 2)) / 255
            transform_norm_aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_mean, train_std),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip()
            ])
        
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])

        trainset = dataset(root='./data', train=True,
                                                transform=transform_norm_aug if apply_augmentation else transform_norm,
                                                download=True)

        testset = dataset(root='./data', train=False,
                                               transform=transform_norm,
                                               download=True)

    else:

        trainset = dataset(root='./data', train=True,
                                                transform=transform if apply_augmentation else transforms.ToTensor(),
                                                download=True)

        testset = dataset(root='./data', train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - validation_size, validation_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainset, valset, testset, trainloader, valloader, testloader, classes


if __name__ == "__main__":
    validation_size = 5000

    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--baseline_model', action='store_true', default=True, help='Use baseline model')
    model_group.add_argument('--other_model', action='store_true', help='Use other model')
    model_group.add_argument('--baseline_model_bn_dropout_reversed', action='store_true', help='using a different order for bn and dropout. Dropout is now applied twice within each layer!')
    model_group.add_argument('--resnet_model', action='store_true', help='Use the ResNet model architecture')
    model_group.add_argument('--resnet_model_bottleneck', action='store_true', help='Use the ResNet model with bottleneck blocks')
    model_group.add_argument('--resnet_model_squeeze_excitation', action='store_true', help='Use the ResNet model with SqueezeExciation blocks')
    model_group.add_argument('--resnet_model_squeeze_excitation_bottleneck', action='store_true', help='Use the ResNet model with SqueezeExciation blocks with bottleneck')
    model_group.add_argument('--resnet_model_squeeze_excitation_adjustable', action='store_true', help='Use the ResNet model with SqueezeExciation blocks with bottleneck')
    model_group.add_argument('--resnet_model_adjustable', action='store_true', help='Use the ResNet model architecture with adjustable residual connection')
    model_group.add_argument('--resnet_pytorch', action='store_true', help="train a pre-defined resnet34 model architecture")

    parser.add_argument("-s", "--scheduler",
                        choices=["default", "step", "warm-up+cosine_annealing", "cosine_annealing+re-starts", "val_loss_plateau"],
                        default="default", help="Select a mode")

    parser.add_argument("-ds", "--dataset", choices=['CIFAR10', 'CIFAR100'], default='CIFAR10', help='choose a training/test data set')
    parser.add_argument("-bs", "--batch_size", type=int, default=100, help="Set the batch size")

    parser.add_argument("-l", "--load", nargs=2, metavar=("FOLDERPATH", "EPOCH"), help="Load data from folder")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("-id", "--increased_dropout", type=float, default=0.0, help="Increased dropout probability")
    parser.add_argument("-w", "--l2_decay", type=float, default=0.0, help="Apply L2 weight regularization")
    parser.add_argument("-e", "--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-a", "--apply_augmentation", action="store_true",
                        help="apply augmentation on the training data")
    parser.add_argument("-ra", "--apply_random_augmentation", action="store_true",
                        help="apply random augmentation on the training data")
    parser.add_argument("--save_every", type=int, default=5, help="How often to save (in epochs)")
    parser.add_argument("-bn", "--batch_norm", action="store_true", help="Use batch normalization")
    parser.add_argument("-n", "--norm_m0_sd1", action="store_true",
                        help="Normalize to have 0 mean and standard deviation 1")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Intial learning rate")

    parser.add_argument("-ad", "--adam", action="store_true", help="Use Adam optimizer")
    parser.add_argument("-aw", "--adamw", action="store_true", help="Use AdamW optimizer")
    parser.add_argument("-rms", "--rmsprop", action="store_true", help="Use RMSprop optimizer")

    args = parser.parse_args()

    batch_size = args.batch_size

    # -l trained_models/20230517-095003 50 -e 50 --save_every 2 -d 0.2

    # Already done
    # -e 100 --save_every 2 # baseline, Viktor | Baseline: 73.xxxxx%
    # -e 100 --save_every 2 -d 0.2 # baseline + dropout, Viktor | Baseline + Dropout: 83.450%
    # -e 50 --save_every 2 -w 0.001 # baseline + weight decay, Santi  | Baseline + Weight Decay: 72.550% Model = "trained_models/20230517-081045_w0_001"
    # Baseline + Increasing Dropout: 84.690%

    # -e 50 --save_every 2 -a, # baseline + augmentation, Theo | Baseline + Data Augmentation: 84.470%
    # --e 200 --save_every 2 -d 0.2 -a  Baseline + Dropout + Data Augmentation: 85.880%, Viktor

    # Increased dropout + data augmentation + batch normalization: 88% (TAKES 3:30 HOURS TO TRAIN. Sight...)
    # -e 400 --save_every 2 -d 0.2 -id 0.1 -a -bn, Viktor (THIEF)
 
    # -e 200 --save_every 2 -d 0.2 -id 0.1 # baseline + dropout + increased dropout, Santi - Colab 

    # -e 200 --save_every 2 -d 0.2 -id 0.1 # baseline + dropout + increased dropout, Santi - Colab

    # Normalize input data to have 0 mean and standard deviation 1 +
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -n , Santi - GCP

    # Using AdamW Optimizer
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -aw, Theo

    # Using Adam Optimizer
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -ad, Viktor

    # Using cosine annealing with warm restarts
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -s cosine_annealing+re-starts, Theo

    # Using step decay
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -s step, Viktor

    # Using Cosine Annealing with Warm-up
    # -e 100 --save_every 2 -d 0.2 -id 0.1 -a -bn -s warm-up+cosine_annealing, Santi

    # python project.py --resnet_model -e 100 --save_every 2 -w 0.00001 -bs 256 -lr .1 --scheduler val_loss_plateau, Theo -> 71.16% loss: 1.95674826854
    # python project.py --resnet_model -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Theo -> 71.51%
    # python project.py --resnet_model -e 100 --save_every 2 -w 0.001 -bs 256 -lr .1 --scheduler val_loss_plateau, Theo -> Test loss: 0.9685037365326514, Test accuracy: 78.22 

    # python project.py --resnet_pytorch -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Theo -> Test loss: 1.7761883766223223, Test accuracy: 72.33000000000001  

    # Using ResNet bottleneck
    # python project.py --resnet_model_bottleneck -e 100 --save_every 2 -w 0.00001 -bs 256 -lr .1 --scheduler val_loss_plateau, Santi ->Test loss: 1.545174586467254, Test accuracy: 54.949999999999996
    # {"airplane": 70.4, "automobile": 60.5, "bird": 50.7, "cat": 35.4, "deer": 44.5, "dog": 40.7,  "frog": 70.6, "horse": 55.2, "ship": 67.5, "truck": 54.0}
    
    # Using ResNet bottleneck with dropout
    # python project.py --resnet_model -e 80 --save_every 2 -d 0.2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Santi -> Test loss: 1.3713150796217797, Test accuracy: 72.65
    # {"airplane": 76.1, "automobile": 83.6, "bird": 62.8, "cat": 57.8, "deer": 65.9, "dog": 59.3, "frog": 82.3, "horse": 76.9, "ship": 84.0, "truck": 77.8}  

    # Random data augmentationf from timm
    # python project.py --resnet_model -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau -n -a -ra, Santi -> Test loss: 0.7131062425099887, Test accuracy: 81.01
    # {"airplane": 84.3, "automobile": 87.4, "bird": 75.8, "cat": 67.8, "deer": 80.9, "dog": 66.8, "frog": 85.7, "horse": 84.4, "ship": 90.2, "truck": 86.8}

    # In progress: 
    # different optimizer:
    # project.py --resnet_model -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .001 --scheduler val_loss_plateau --rmsprop

    # Squeeze Excitation
    # --resnet_model_squeeze_excitation -e 100 --save_every 2 -d 0.2 -w 0.001 -bs 256 -lr .1 --scheduler val_loss_plateau, Viktor -> Test loss: 0.8458700485718555, Test accuracy: 78.06 @ 30 epochs
    # --resnet_model_squeeze_excitation -e 100 --save_every 2 -d 0.2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Viktor -> 
    # Squeeze Excitation with bottleneck
    # --resnet_model_squeeze_excitation_bottleneck -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Viktor
    #python project.py --resnet_model -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau -aw, Viktor
    #python project.py --resnet_model -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau -ad, Eduardo

    # # Adjustable Residual connections
    # python project.py --resnet_model_adjustable -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau, Eduardo -> Highest 74.37%
    # python project.py --resnet_model_adjustable -e 100 --save_every 2 -w 0.0001 -bs 256 -lr 0.001, Eduardo -> Highest 53.15%
    # python project.py --resnet_model_adjustable -e 100 --save_every 2 -w 0.001 -bs 256 -lr .1 --scheduler val_loss_plateau, Eduardo -> 79.18\%

    
    # Combine augmentation with adjustable;
    # python project.py --resnet_model_adjustable -e 100 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau -a -ra -n, Santi
    
    # TODO:

    # Combine augmentation with dropout;
    # python project.py --resnet_model -e 100 -d 0.2 --save_every 2 -w 0.0001 -bs 256 -lr .1 --scheduler val_loss_plateau -a -ra -n , ?

    # Using ResNet bottleneck
    # python project.py --resnet_model_bottleneck -e 100 --save_every 2 -w 0.00001 -bs 256 -lr .1 --scheduler val_loss_plateau, ?

    if args.load:
        folder_path = args.load[0]
        epoch = args.load[1]
        pickle_path = get_path(folder_path, PICKLE_FILENAME, epoch)
        if not os.path.exists(pickle_path):
            raise Exception(f"File {pickle_path} does not exist")
        ph = PickleHelper(pickle_path)
    else:
        folder_path = folder_helper("trained_models")
        ph = PickleHelper(get_path(folder_path, PICKLE_FILENAME, min(args.n_epochs, args.save_every)))

    trainset, valset, testset, trainloader, valloader, testloader, classes = load_data(args.apply_augmentation, args.apply_random_augmentation, args.norm_m0_sd1, torchvision.datasets.CIFAR100 if args.dataset == 'CIFAR100' else torchvision.datasets.CIFAR10)

    showImages = False
    if showImages:
        for i in range(3):
            def imshow(img):
                npimg = img.numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.show()


            # get some random training images
            dataiter = iter(trainloader)
            images, labels = next(dataiter)

            # show images
            imshow(torchvision.utils.make_grid(images))
            print(' '.join(f'{classes[labels[j]]:5s} {chr(10) if (j % 8) == 7 else ""}' for j in range(batch_size)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}...")

    model = BaselineModel(args.dropout, args.increased_dropout, args.batch_norm, num_classes=len(classes))
    if args.baseline_model_bn_dropout_reversed:
        model = BaselineModelModifiedBNDropoutOrder(args.dropout, args.increased_dropout, args.batch_norm, num_classes=len(classes))
        
    elif args.resnet_model:
        model = ResNetModel(num_classes=len(classes), dropout=args.dropout)
    elif args.resnet_model_squeeze_excitation:
        model = ResNetModel(block=SqueezeExcitationBlockFullBasic, num_classes=len(classes), dropout=args.dropout)
    elif args.resnet_model_squeeze_excitation_bottleneck:
        model = ResNetModel(block=SqueezeExcitationBlockFullBottleneck, num_classes=len(classes), dropout=args.dropout)
    elif args.resnet_model_squeeze_excitation_adjustable:
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = ResNetModel(block=SqueezeExcitationBlockFullBasic, num_classes=len(classes), k_list=k_list, dropout=args.dropout)
    elif args.resnet_model_bottleneck:
        model = ResNetModel(block=BottleneckBlock, num_classes=len(classes), dropout=args.dropout)
    elif args.resnet_model_adjustable:
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = ResNetModel(num_classes=len(classes), k_list=k_list, dropout=args.dropout)
    elif args.resnet_pytorch:
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.to(device)

    history = ph.register(
        "history",
        {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    )
    test_loss, test_acc, _test_acc_per_class = ph.register("test_performance", (None, None, None))
    np_random_state = ph.register("np_random_state", np.random.get_state())
    torch_random_state = ph.register("torch_random_state", torch.get_rng_state())
    random_state = ph.register("random_state", random.getstate())

    np.random.set_state(np_random_state)
    torch.set_rng_state(torch_random_state)
    random.setstate(random_state)

    num_epochs = args.n_epochs
    offset_epochs = 0
    num_batches = int(len(trainset) / batch_size)

    if args.load:
        print("Loading model from file...")
        epochs = int(args.load[1])
        model_path = get_path(folder_path, MODEL_FILENAME, epochs)
        model.load_state_dict(torch.load(model_path))

        plot_path = get_path(folder_path, PLOT_FILENAME, epochs)
        summarize_diagnostics(plot_path, history)

        print(f"Test loss: {test_loss}")
        offset_epochs = epochs
        num_epochs += epochs

    if num_epochs - offset_epochs > 0:
        print(f"Training the network for {num_epochs - offset_epochs} {'more epochs' if offset_epochs > 0 else ''}...")
        loss_function = nn.CrossEntropyLoss()
        lr = args.learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=args.l2_decay)
        if args.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_decay)
        elif args.adamw:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.l2_decay)
        elif args.rmsprop:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.l2_decay, momentum=.9)

        # lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.001)  # constant default learning rate
        if args.scheduler == 'step':
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.scheduler == 'cosine_annealing+re-starts':
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        elif args.scheduler == 'warm-up+cosine_annealing':
            n_warm_up_epochs = 10
            n_annealing_epochs = num_epochs - n_warm_up_epochs
            eta_min = 0.0001
            eta_max = 0.1
            warm_up_learning_rate = lambda epoch: eta_min + epoch / n_warm_up_epochs * (eta_max - eta_min)
            cosine_annealing = lambda epoch: eta_min + 0.5 * (eta_max - eta_min) * (
                        1 + np.cos(epoch / n_annealing_epochs * np.pi))

            warm_up_and_cosine_annealing = lambda epoch: warm_up_learning_rate(epoch) if epoch < n_warm_up_epochs else cosine_annealing(epoch - n_warm_up_epochs)
        elif args.scheduler == 'val_loss_plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        for epoch in tqdm(range(offset_epochs, num_epochs), total=num_epochs - offset_epochs, ascii=True):
            model.train()

            for index, (images, labels) in tqdm(enumerate(trainloader), total=num_batches, leave=False, ascii=True):
                
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                prediction = model(images)
                current_loss = loss_function(prediction, labels)
                current_loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                model.eval()

                val_loss, val_acc, _val_acc_per_class = compute_metrics(model, valloader)
                train_loss, train_acc, _train_acc_per_class = compute_metrics(model, trainloader)

                history['train_acc'].append(train_acc)
                history['train_loss'].append(train_loss)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)

                if epoch % 2 == 0:
                    tqdm.write(f"Training loss: {train_loss}, Validation loss: {val_loss}, current learning rate: {optimizer.param_groups[-1]['lr']}")

            if args.scheduler == 'warm-up+cosine_annealing':
                for g in optimizer.param_groups:
                    g['lr'] = warm_up_and_cosine_annealing(epoch)
            elif args.scheduler == 'val_loss_plateau':
                lr_scheduler.step(val_loss)
            elif args.scheduler != 'default':
                lr_scheduler.step()
            

            if (epoch + 1) % args.save_every == 0:
                tqdm.write(f'Saving the model after epoch: {epoch + 1}...')
                model_path = get_path(folder_path, MODEL_FILENAME, epoch + 1)
                torch.save(model.state_dict(), model_path)

                tqdm.write(f'Generating plots after epoch: {epoch + 1}...')
                plotpath = get_path(folder_path, PLOT_FILENAME, epoch + 1)
                summarize_diagnostics(plotpath, history)

                # Testing the network
                tqdm.write(f'Testing the network after epoch: {epoch + 1}...')
                model.eval()
                test_loss, test_acc, _test_acc_per_class = compute_metrics(model, testloader)
                tqdm.write(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
                tqdm.write(json.dumps(_test_acc_per_class))

                ph.update("history", history)
                ph.update("test_performance", (test_loss, test_acc, _test_acc_per_class))
                ph.update("np_random_state", np.random.get_state())
                ph.update("torch_random_state", torch.get_rng_state())
                ph.update("random_state", random.getstate())
                pickle_path = get_path(folder_path, PICKLE_FILENAME, epoch + 1)
                ph.save(path=pickle_path)
