import argparse
import json
import os.path
import pickle
import sys
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random

from senet import SENet34, SENetBottleneck34
from resnet import ResNet34, ResNetBottleneck34

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


def load_data(apply_augmentation=False, apply_random_aug=False, norm_m0_sd1=False, dataset=torchvision.datasets.CIFAR10):
    trainset = dataset(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    classes = trainset.classes

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = dataset(
        root='./data', 
        train=True,
        transform=transform_train,
        download=True
        )

    testset = dataset(
        root='./data', 
        train=False,
        transform=transform_test,
        download=True
    )

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
    model_group.add_argument('--baseline_model', action='store_true', default=True, help='Use baseline model')
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
                             help='Use the ResNet model with SqueezeExciation blocks with bottleneck')
    model_group.add_argument('--resnet_model_adjustable', action='store_true',
                             help='Use the ResNet model architecture with adjustable residual connection')
    model_group.add_argument('--resnet_pytorch', action='store_true',
                             help="train a pre-defined resnet34 model architecture")

    parser.add_argument("-s", "--scheduler",
                        choices=["default", "step", "warm-up+cosine_annealing", "cosine_annealing+re-starts",
                                 "val_loss_plateau"],
                        default="default", help="Select a mode")

    parser.add_argument("-ds", "--dataset", choices=['CIFAR10', 'CIFAR100'], default='CIFAR10',
                        help='choose a training/test data set')
    parser.add_argument("-bs", "--batch_size", type=int, default=100, help="Set the batch size")

    parser.add_argument("-l", "--load", nargs=2, metavar=("FOLDERPATH", "EPOCH"), help="Load data from folder")
    parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("-id", "--increased_dropout", type=float, default=0.0, help="Increased dropout probability")
    parser.add_argument("-w", "--l2_decay", type=float, default=0.0, help="Apply L2 weight regularization")
    parser.add_argument("-e", "--n_epochs", type=int, default=10, help="Number of epochs")

    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument("-a", "--apply_augmentation", action="store_true",
                                    help="apply augmentation on the training data")
    augmentation_group.add_argument("-ra", "--apply_random_augmentation", action="store_true",
                                    help="apply random augmentation on the training data")

    parser.add_argument("-n", "--norm_m0_sd1", action="store_true",
                        help="Normalize to have 0 mean and standard deviation 1")
    parser.add_argument("--save_every", type=int, default=5, help="How often to save (in epochs)")
    parser.add_argument("-bn", "--batch_norm", action="store_true", help="Use batch normalization")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Intial learning rate")

    parser.add_argument("-ad", "--adam", action="store_true", help="Use Adam optimizer")
    parser.add_argument("-aw", "--adamw", action="store_true", help="Use AdamW optimizer")
    parser.add_argument("-rms", "--rmsprop", action="store_true", help="Use RMSprop optimizer")

    return parser.parse_args()

def get_model(device, args, classes):
    if args.baseline_model_bn_dropout_reversed:
        model = BaselineModelModifiedBNDropoutOrder(
            args.dropout,
            args.increased_dropout,
            args.batch_norm,
            num_classes=len(classes)
        )
    elif args.resnet_model:
        model = ResNet34()
    elif args.resnet_model_bottleneck:
        model = ResNetBottleneck34(args.dropout)
    elif args.resnet_model_adjustable:
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = ResNet34(args.dropout, k_list)
    elif args.resnet_model_squeeze_excitation:
        model = SENet34()
    elif args.resnet_model_squeeze_excitation_bottleneck:
        model = SENetBottleneck34()
    elif args.resnet_model_squeeze_excitation_adjustable:
        k_list = [1.0, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
        model = SENet34(args.dropout, k_list)
    elif args.resnet_pytorch:
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_parameters}")
    return model


def get_optimizer(lr, args, model, momentum=.9):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=args.l2_decay)
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_decay)
    elif args.adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.l2_decay)
    elif args.rmsprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.l2_decay, momentum=momentum)

    return optimizer


if __name__ == "__main__":
    validation_size = 5000
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
        torchvision.datasets.CIFAR100 if args.dataset == 'CIFAR100' else torchvision.datasets.CIFAR10
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}...")
    model = get_model(device, args, classes)

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
        optimizer = get_optimizer(lr, args, model)

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
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
