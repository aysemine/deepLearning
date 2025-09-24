"""
CIFAR-10 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(((0.5, 0.5, 0.5)), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train= True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data", train= False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    return train_loader, test_loader

def imshow(img):
    # denormalize
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))

def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loader()

    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f"label: {labels[i].item()}")
        plt.axis("off")
    plt.show()

# visualize(3)

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, padding=1) # in_channel = rgb 3 out_channel = number of filter
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2) # randomly drops 0.2 neurons connections
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*8*8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
model = CNN().to(device)

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
)

def train_model(model, train_loader, criterion, optimizer, epochs = 5):

    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epochs: {epoch+1}/{epochs}, Loss: {avg_loss:.3f}")
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label= "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

# train_loader, test_loader = get_data_loader()
# model = CNN().to(device)
# criterion, optimizer = define_loss_and_optimizer(model)
# train_model(model, train_loader, criterion, optimizer)

def test_model(model, test_loader, dataset_type):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted =torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"{dataset_type} accuracy: {100*correct/ total}%")

# test_model(model, test_loader, dataset_type = "Test")
# test_model(model, train_loader, dataset_type = "Train")

# %% main

if __name__ == "__main__":
    
    train_loader, test_loader = get_data_loader()

    visualize(10)

    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    test_model(model, test_loader, dataset_type = "Test")
    test_model(model, train_loader, dataset_type = "Train")