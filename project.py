import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random


torch.manual_seed(30)
random.seed(30)
np.random.seed(30)

class BaselineModel(nn.Module):

    def __init__(self):
        super(BaselineModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block1[0].weight),   
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block1[3].weight),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block2[0].weight),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block2[3].weight),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block3[0].weight),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.init.kaiming_uniform_(self.block3[0].weight),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
            

        """ self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same')
        # nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        nn.init.kaiming_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        nn.init.kaiming_uniform_(self.conv4.weight)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        nn.init.kaiming_uniform_(self.conv5.weight)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        nn.init.kaiming_uniform_(self.conv6.weight)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2) """

        """ 
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 10) """

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.init.kaiming_uniform_(self.fc[0].weight),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(-1, 64 * 8 * 8)

        x = self.fc(x)

        return x
    
def summarize_diagnostics(history):
    fig, axs = plt.subplots(2, 1)

    axs[0].title('Cross Entropy Loss')
    axs[0].plot(history.history['loss'], color='blue', label='train')
    axs[0].plot(history.history['val_loss'], color='orange', label='test')

    axs[1].title('Classification Accuracy')
    axs[1].plot(history.history['accuracy'], color='blue', label='train')
    axs[1].plot(history.history['val_accuracy'], color='orange', label='test')

    plt.show()

if __name__ == "__main__":

    batch_size = 32
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

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
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
    

    baseline_model = BaselineModel()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(baseline_model.parameters(), lr=.001, momentum=.9)

    num_epochs = 10

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print("Training the network...")
    for epoch in range(num_epochs):              
        baseline_model.train()

        train_loss_sum = 0
        train_acc_sum = 0

        val_loss_sum = 0
        val_acc_sum = 0

        for index, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = baseline_model(images)
            current_loss = loss_function(prediction, labels)
            current_loss.backward()
            optimizer.step()

            train_loss_sum += current_loss.detach().item()
        
        with nn.no_grad():
            baseline_model.eval()

            for index, (images, labels) in enumerate(valLoader):
                images = images.to(device)
                labels = labels.to(device)

                prediction = baseline_model(images)
                current_loss = loss_function(prediction, labels)

                val_loss_sum += current_loss.detach().item()

    print('Saving the model...')
    torch.save(baseline_model.state_dict(), "./trained_models/baseline_model.pth")