"""
problem : number classification with mnist dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision # for image processing and predefined models
import torchvision.transforms as transforms # image transformation
import matplotlib.pyplot as plt

# optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading

def get_data_loaders(batch_size = 64): # data size for each iterations
    transform = transforms.Compose([
        transforms.ToTensor(), # transforms image to tensor 0-255 -> 0-1
        transforms.Normalize((0.5,), (0.5,)) # scales values between -1 to 1
    ])

    train_set = torchvision.datasets.MNIST(root="./data", train= True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root="./data", train= False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

# data visualization

def visualize_samples(loader, n):
    images, labels = next(iter(loader)) # get image and labels from first batch
    print(images[0].shape)
    fig, axes = plt.subplots(1, n, figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {labels[i].item()}")
    plt.show()

visualize_samples(train_loader, 4)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # vectorization of images 1D
        self.flatten = nn.Flatten()

        # first fully connected layer
        self.fcl = nn.Linear(28*28, 128)

        # activation function
        self.relu = nn.ReLU()

        # second fully connected layer
        self.fc2 = nn.Linear(128, 64)

        # output layer
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):

        # initial x is 28*28 image
        x = self.flatten(x)
        x = self.fcl(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
    
model = NeuralNetwork().to(device)

# loss function

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(),
    optim.Adam(model.parameters(), lr=0.001)
)

criterion, optimizer = define_loss_and_optimizer(model)

def train_model(model, train_loader, criterion, optimizer, epochs = 10):

    model.train()
    # list of loss for each epoch
    train_losses = []

    # train
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # zero gradients
            optimizer.zero_grad()
            
            # forward propogation
            predictions = model(images)
            
            # find losses
            loss = criterion(predictions, labels)
            
            # update weights (backpropogation)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch + 1} / {epochs}, Loss: {avg_loss:.3f}")
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker="o", linestyle="-", label= "Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

train_model(model, train_loader, criterion, optimizer, epochs=5)

def test_model(model, test_loader):
    model.eval()

    # counters
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy: {100*correct/total:.3f}")

test_model(model, test_loader)

# %% main

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    visualize_samples(train_loader, 5)
    model = NeuralNetwork().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    test_model(model, test_loader)



