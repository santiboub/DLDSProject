import argparse
import json
import os.path
import pickle
import sys
import time

import torch
import torch.nn as nn
from fastai.data.external import untar_data, URLs
from fastai.vision.data import ImageDataLoaders
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import torchvision
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform


import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random

from senet import SENet34, SENetBottleneck34
from resnet import ResNet34, ResNetBottleneck34
from baseline import BaselineModel, BaselineModelModifiedBNDropoutOrder
from loss import SymmetricCrossEntropyLoss, LabelSmoothingCrossEntropyLoss

MODEL_FILENAME = "baseline_model.pth"
PICKLE_FILENAME = "data.pickle"
PLOT_FILENAME = "baseline.pdf"

torch.manual_seed(30)
random.seed(30)
np.random.seed(30)


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

    accuracy_per_class = {}

    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        else:
            accuracy = 0

        accuracy_per_class[classname] = accuracy

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


def load_data(base_augmentation=False, random_augmentation=False, norm_m0_sd1=False, dataset=torchvision.datasets.CIFAR10, label_noise_probability=0.0):
    trainset = dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    classes = trainset.classes

    train_mean = trainset.data.mean(axis=(0, 1, 2))
    train_std = trainset.data.std(axis=(0, 1, 2))

    transform_train = []
    transform_test = []

    if base_augmentation:
        transform_train.append(transforms.RandomCrop(32, padding=4))
        transform_train.append(transforms.RandomHorizontalFlip())
    elif random_augmentation:
        transform_train.append(
            rand_augment_transform(
                config_str='rand-m9-mstd0.5', 
                hparams={
                    'img_mean': (int(train_mean[0]), int(train_mean[1]), int(train_mean[2]))
                }
            )
        )

    transform_train.append(transforms.ToTensor())
    transform_test.append(transforms.ToTensor())

    if norm_m0_sd1:
        train_mean /= 255
        train_std /= 255
        transform_train.append(transforms.Normalize(train_mean, train_std))
        transform_test.append(transforms.Normalize(train_mean, train_std))

    trainset = dataset(
        root='./data', 
        train=True,
        transform=transforms.Compose(transform_train),
        download=True
    )

    def contaminate_label(label, probability):
        if np.random.rand() < probability:
            noisy_label = np.random.randint(len(classes), size=1)[0]
            while noisy_label == label:
                noisy_label = np.random.randint(len(classes), size=1)[0]
            return noisy_label
        return label

    for i in range(len(trainset)):
        label = trainset[i][1]
        noisy_label = contaminate_label(label, label_noise_probability)
        trainset.targets[i] = noisy_label

    testset = dataset(
        root='./data', 
        train=False,
        transform=transforms.Compose(transform_test),
        download=True
    )

    validation_size = int(len(trainset) * .1)
    print(validation_size)
    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - validation_size, validation_size])

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )

    return trainset, valset, testset, trainloader, valloader, testloader, classes


def parse_arguments():
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--baseline_model', action='store_true', help='Use baseline model')
    model_group.add_argument('--other_model', action='store_true', help='Use other model')
    model_group.add_argument('--baseline_model_bn_dropout_reversed', action='store_true',
                             help='using a different order for bn and dropout. Dropout is now applied twice within each layer!')
    model_group.add_argument('--resnet_model', action='store_true', help='Use the ResNet model architecture')
    model_group.add_argument('--resnet_model_bottleneck', action='store_true',
                             help='Use the ResNet model with bottleneck blocks')
    model_group.add_argument('--resnet_model_squeeze_excitation', action='store_true',
                             help='Use the ResNet model with SqueezeExciation blocks')
    model_group.add_argument('--resnet_model_squeeze_excitation_bottleneck', action='store_true',
                             help='Use the ResNet model with SqueezeExciation blocks with bottleneck')
    model_group.add_argument('--resnet_model_squeeze_excitation_adjustable', action='store_true',
                             help='Use the ResNet model with SqueezeExciation blocks with bottleneck, adjustable residual connection')
    model_group.add_argument('--resnet_model_adjustable', action='store_true',
                             help='Use the ResNet model architecture with adjustable residual connection')
    model_group.add_argument('--resnet_pytorch', action='store_true',
                             help="train a pre-defined resnet34 model architecture")

    parser.add_argument("-s", "--scheduler",
                        choices=["default", "step", "warm-up+cosine_annealing", "cosine_annealing+re-starts",
                                 "val_loss_plateau", "cosine_annealing"],
                        default="default", help="Select a mode")

    parser.add_argument("-ds", "--dataset", choices=['CIFAR10', 'CIFAR100', 'Imagenette'], default='CIFAR10',
                        help='choose a training/test data set')
    parser.add_argument("-bs", "--batch_size", type=int, default=100, metavar=("SIZE"), help="Set the batch size")

    parser.add_argument("-l", "--load", nargs=2, metavar=("FOLDERPATH", "EPOCH"), help="Load data from folder")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, metavar=("PROBABILITY"), help="Dropout probability")
    parser.add_argument("-id", "--increased_dropout", type=float, default=0.0, metavar=("PROBABILITY"), help="Increased dropout probability")
    parser.add_argument("-w", "--l2_decay", type=float, default=0.0, metavar=("WEIGHT_DECAY"), help="Apply L2 weight regularization")
    parser.add_argument("-e", "--n_epochs", type=int, default=10, metavar=("EPOCH"), help="Number of epochs")

    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument("-a", "--apply_augmentation", action="store_true",
                                    help="apply augmentation on the training data")
    augmentation_group.add_argument("-ra", "--apply_random_augmentation", action="store_true",
                                    help="apply random augmentation on the training data")

    parser.add_argument("-n", "--norm_m0_sd1", action="store_true",
                        help="Normalize to have 0 mean and standard deviation 1")
    parser.add_argument("--save_every", type=int, default=5, metavar=("EPOCHS"), help="How often to save (in epochs)")
    parser.add_argument("-bn", "--batch_norm", action="store_true", help="Use batch normalization")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, metavar=("LEARNING RATE"), help="Intial learning rate")

    parser.add_argument("-ad", "--adam", action="store_true", help="Use Adam optimizer")
    parser.add_argument("-aw", "--adamw", action="store_true", help="Use AdamW optimizer")
    parser.add_argument("-rms", "--rmsprop", action="store_true", help="Use RMSprop optimizer")

    parser.add_argument("-ln", "--label_noise_probability", type=float, default=0, metavar=("PROBABILITY"), help="contaminate the labels")
    
    loss_group = parser.add_mutually_exclusive_group()
    loss_group.add_argument("-sce", "--symmetric_cross_entropy_loss", action="store_true", help="Use the symmetric cross entropy loss function")
    loss_group.add_argument("-lsce", "--label_smoothing_cross_entropy_loss", type=float, metavar=("SMOOTHING"), help="Use the smoothing loss regularization cross entropy loss")

    return parser.parse_args()

def get_model(device, args, classes):
    if args.baseline_model:
        model = BaselineModel(
            args.dropout,
            args.increased_dropout,
            args.batch_norm,
            num_classes=len(classes)
        )
    elif args.baseline_model_bn_dropout_reversed:
        model = BaselineModelModifiedBNDropoutOrder(
            args.dropout,
            args.increased_dropout,
            args.batch_norm,
            num_classes=len(classes)
        )
    elif args.resnet_model:
        print("Using ResNet model...")
        model = ResNet34(num_classes=len(classes))
    elif args.resnet_model_bottleneck:
        print("Using ResNet model with bottleneck blocks...")
        model = ResNetBottleneck34(num_classes=len(classes), dropout=args.dropout)
    elif args.resnet_model_adjustable:
        print("Using ResNet model with adjustable residual connection...")
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = ResNet34(num_classes=len(classes), dropout=args.dropout, k_list=k_list)
    elif args.resnet_model_squeeze_excitation:
        print("Using ResNet model with SqueezeExcitation blocks...")
        model = SENet34(num_classes=len(classes))
    elif args.resnet_model_squeeze_excitation_bottleneck:
        print("Using ResNet model with SqueezeExcitation blocks with bottleneck...")
        model = SENetBottleneck34(num_classes=len(classes))
    elif args.resnet_model_squeeze_excitation_adjustable:
        print("Using ResNet model with SqueezeExcitation blocks with bottleneck, adjustable residual connection...")
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = SENet34(num_classes=len(classes), dropout=args.dropout, k_list=k_list)
    elif args.resnet_pytorch:
        print("Using pre-defined ResNet model from PyTorch...")
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_parameters}")
    return model


def get_optimizer(lr, args, model, momentum=.9):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=args.l2_decay)
    if args.adam:
        print("Using Adam optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_decay)
    elif args.adamw:
        print("Using AdamW optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.l2_decay)
    elif args.rmsprop:
        print("Using RMSprop optimizer...")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.l2_decay, momentum=momentum)

    return optimizer


def get_dataset(args):
    if args.dataset == 'CIFAR10':
        return torchvision.datasets.CIFAR10
    elif args.dataset == 'CIFAR100':
        return torchvision.datasets.CIFAR100
    elif args.dataset == 'Imagenette':
        path = untar_data(URLs.IMAGENETTE_160)
        dataloader = ImageDataLoaders.from_folder(path, train='train', valid='val')

        train_loader = dataloader.train
        classes = dataloader.vocab

        # data set does not have a test set -> we use the validation set as our test set and split the training data
        # into training and validation data
        val_loader = dataloader.valid

        dataset_lambda = lambda root, train, transform, download: ImagenetteDataset(classes, train_loader, val_loader, train=train, transform=transform)

        return dataset_lambda

    return None


class ImagenetteDataset(Dataset):

    def __init__(self, classes, train_loader, test_loader, train=True, transform=None):
        self.classes = classes
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.train = train
        self.transform = transform

        self.data = [None] * self.__len__()
        self.targets = [None] * self.__len__()
        self._load_data()

    def _load_data(self):
        loader = self.train_loader if self.train else self.test_loader

        for idx in range(self.__len__()):
            sample, label = loader.dataset[idx]

            sample = sample.resize((32, 32))

            self.data[idx] = np.array(sample)
            self.targets[idx] = label.item()

        self.data = np.stack(self.data, axis=0)

    def __len__(self):
        if self.train:
            return len(self.train_loader.dataset)
        else:
            return len(self.test_loader.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample, label = self.data[idx], self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


if __name__ == "__main__":
    args = parse_arguments()
    batch_size = args.batch_size

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

    trainset, valset, testset, trainloader, valloader, testloader, classes = load_data(
        args.apply_augmentation, 
        args.apply_random_augmentation, 
        args.norm_m0_sd1, 
        get_dataset(args),
        args.label_noise_probability
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}...")
    model = get_model(device, args, classes)

    if args.symmetric_cross_entropy_loss:
        print("Using Symmetric Cross Entropy Loss...")
        loss_function = SymmetricCrossEntropyLoss(len(classes), alpha=0.1, beta=1.0, A=-4)
    elif args.label_smoothing_cross_entropy_loss:
        print("Using Label Smoothing Cross Entropy Loss...")
        loss_function = LabelSmoothingCrossEntropyLoss(len(classes), smoothing=args.label_smoothing_cross_entropy_loss)
    else:
        print("Using Cross Entropy Loss...")
        loss_function = nn.CrossEntropyLoss()

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
        lr = args.learning_rate
        optimizer = get_optimizer(lr, args, model)

        if args.scheduler == 'step':
            print("Using StepLR scheduler...")
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.scheduler == 'cosine_annealing+re-starts':
            print("Using CosineAnnealingWarmRestarts scheduler...")
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        elif args.scheduler == 'warm-up+cosine_annealing':
            print("Using warm-up+cosine_annealing scheduler...")
            n_warm_up_epochs = 10
            n_annealing_epochs = num_epochs - n_warm_up_epochs
            eta_min = 0.0001
            eta_max = 0.1
            warm_up_learning_rate = lambda epoch: eta_min + epoch / n_warm_up_epochs * (eta_max - eta_min)
            cosine_annealing = lambda epoch: eta_min + 0.5 * (eta_max - eta_min) * (
                        1 + np.cos(epoch / n_annealing_epochs * np.pi))

            warm_up_and_cosine_annealing = lambda epoch: warm_up_learning_rate(epoch) if epoch < n_warm_up_epochs else cosine_annealing(epoch - n_warm_up_epochs)
        elif args.scheduler == 'val_loss_plateau':
            print("Using ReduceLROnPlateau scheduler...")
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif args.scheduler == 'cosine_annealing':
            print("Using CosineAnnealingLR scheduler...")
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            print('Using the default fixed initial learning rate scheduler')
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

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
            else:
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
