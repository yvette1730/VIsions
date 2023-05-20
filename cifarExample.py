import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from alexNet import AlexNet
from resNet import ResNet
from tqdm import tqdm 
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def imshow(img):
    """functions to show an image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(2)  # show for 2 seconds # plt.show()


def show_images(loader):
    """show some images from the loader"""

    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(images.shape[0])))


def build_loaders():
    """returns both dataloaders"""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 4

    loaders = dict()
    for key, train in zip(["train", "test"], [True, False]):
        dataset = torchvision.datasets.CIFAR10(
            train=train, root="./data", download=True, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        loaders[key] = loader

    return loaders


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, loader, optimizer, criterion):
    """trains a model"""

 
    epochs = 10
    prog = tqdm(total=epochs)
    for epoch in range(epochs):  # loop over the dataset multiple times

        losses = []
        for data in tqdm(loader, leave=False):
            X, Y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            Yh = model(X)
            loss = criterion(Yh, Y)
            loss.backward()
            optimizer.step()

            # print statistics
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        prog.set_description(f"loss: {avg_loss:.3f}")
        prog.update()

    print("Finished Training")


def inference(model, loader):
    """inference on a model"""

    correct, total = 0, 0
    with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
        for (X, Y) in loader:
            Yh = model(X)  # calculate outputs by running images through the network
            _, Yh = torch.max(
                Yh, dim=1
            )  # the class with the highest energy is what we choose as prediction

            total += Y.size(0)
            correct += (Yh == Y).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def print_shape(model,input, output):
    print(model.__class__.__name__)
   # print(len(input)) 
   # print(input)
    print(f'input:{input[0].shape}|output:{output[0].shape}')
    print()

h = torch.nn.modules.module.register_module_forward_hook(print_shape)
#    """docstring"""

# Initialize model
 
def main():

    loaders = build_loaders()
    show_images(loaders["train"])
    model = AlexNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, loaders["train"], optimizer, criterion)

    # save and/or load model
    PATH = "cifar_model.pt"
    torch.save(model.state_dict(),PATH)
    model.load_state_dict(torch.load(PATH))

    inference(model, loaders["test"])


if __name__ == "__main__":
    main()
