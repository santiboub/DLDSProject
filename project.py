import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm
import pandas as pd 
import matplotlib.pyplot as plt
import random


torch.manual_seed(30)
random.seed(30)
np.random.seed(30)

class BaselineModel(nn.Module):

    
    def __init__(self, dropout=0):
        super(BaselineModel, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)  

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.block1.apply(init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.block2.apply(init_weights)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.block3.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.fc.apply(init_weights)

        self.to(device)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # x4 = x3.view(-1, 64 * 8 * 8)
        x4 = x3.reshape((batch_size,-1))

        x5 = self.fc(x4)

        return x5

def plot_curves(ax, train, val, name):
    ax.set_title(name)
    ax.plot(train, color='blue', label='Training ' + name)
    ax.plot(val, color='orange', label='Validation ' + name)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(name)
    ax.legend()

def summarize_diagnostics2(history):
    fig, axs = plt.subplots(2, 1)

    plot_curves(axs[0], history['train_loss'], history['val_loss'], 'Loss')
    plot_curves(axs[1], history['train_acc'], history['val_acc'], 'Accuracy')

    plt.tight_layout()
    plt.savefig('baseline.eps')

def summarize_diagnostics(history):
    fig, axs = plt.subplots(2, 1)

    axs[0].set_title('Cross Entropy Loss')
    axs[0].plot(history['train_loss'], color='blue', label='train')
    axs[0].plot(history['val_loss'], color='orange', label='test')

    axs[1].set_title('Classification Accuracy')
    axs[1].plot(history['train_acc'], color='blue', label='train')
    axs[1].plot(history['val_acc'], color='orange', label='test')

    plt.savefig("wat.pdf")
    
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

    #print(f'Accuracy of the network on the test images: {accuracy} %')

    accuracy_per_class = {}

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        accuracy_per_class[classname] = accuracy
        #print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    return network_loss, network_accuracy, accuracy_per_class
    
if __name__ == "__main__":
    batch_size = 100
    validation_size = 5000

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - validation_size, validation_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    showImages = False
    if showImages:
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # get some random training images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # show images
        imshow(torchvision.utils.make_grid(images))
        print(' '.join(f'{classes[labels[j]]:5s} {chr(10) if (j % 8)==7 else ""}' for j in range(batch_size)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training...")
    

    baseline_model = BaselineModel()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(baseline_model.parameters(), lr=.001, momentum=.9)

    # num_epochs = 50
    num_epochs = 10

    num_batches = int(len(trainset) / batch_size)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"Training the network for {num_epochs}...")
    for epoch in tqdm(range(num_epochs), total=num_epochs):              
        baseline_model.train()

        for index, (images, labels) in tqdm(enumerate(trainloader), total=num_batches, leave=False):
            # print(f"Training on batch {index} of {int(len(trainset) / batch_size)}", end="\r")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = baseline_model(images)
            current_loss = loss_function(prediction, labels)
            current_loss.backward()
            optimizer.step()

            # train_loss_sum += current_loss.detach().item()
        
        with torch.no_grad():
            baseline_model.eval()
            
            val_loss, val_acc, _val_acc_per_class = compute_metrics(baseline_model, valLoader)
            train_loss, train_acc, _train_acc_per_class = compute_metrics(baseline_model, trainloader)

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
         
            if epoch % 2 == 0:
                tqdm.write(f"Training loss: {train_loss}, Validation loss: {val_loss}")
        

    print('Saving the model...')
    torch.save(baseline_model.state_dict(), "./trained_models/baseline_model.pth")

    print('Generating plots...')
    summarize_diagnostics2(history)


    # Testing the network
    print('Testing the network...')
    baseline_model.eval()
    test_loss, test_acc, _test_acc_per_class = compute_metrics(baseline_model, testloader)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
    print(_test_acc_per_class)
