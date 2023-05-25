import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from alexNet import AlexNet
from resNet import ResNet18
from western_blot import WBLOT   
from tqdm import tqdm 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch import distributed as dist
from datetime import timedelta
import sklearn
from torch.distributed import init_process_group, destroy_process_group
import os 

world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0 
world_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0 
master = world_rank == 0 

dist.init_process_group(backend="gloo",
    rank = world_rank,
    world_size = world_size,
    timeout=timedelta(seconds=30),)
torch.cuda.device(rank)


classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imshow(img):
    """functions to show an image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('figure.png')  # show for 2 seconds # plt.show()


def show_images(loader):
    """show some images from the loader"""

    # get some random training images
   # dataiter = iter(loader)
    #images, labels = next(dataiter)

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j].argmax(dim=1)]:5s}" for j in range(images.shape[0])))


def build_loaders():
    """returns both dataloaders"""

    dataset = WBLOT()
    idxs = list(range(len(dataset)))
    test_size = 0.3
    idxa, idxb = sklearn.model_selection.train_test_split(idxs, test_size=test_size, random_state=0)

    dataseta = torch.utils.data.Subset(dataset, idxa)
    datasetb = torch.utils.data.Subset(dataset, idxb)
    datasets = [dataseta, datasetb]
    
    sampler = DistributedSampler(dataset) 
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 4

    loaders = dict()
    for key, dataset in zip(["train", "test"], datasets):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2,sampler = DistributedSampler(dataset) 
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
    prog = tqdm(total=epochs, leave=False)
    for epoch in range(epochs):  # loop over the dataset multiple times

        losses = []
        for data in tqdm(loader, leave=False):
            X, Y = data
            X = X.to(device) 
            Y = Y.to(device)
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
    print(type(input))
    print(input)
    print(f'input:{input[0].shape}|output:{output[0].shape}')
    print()

#    """docstring"""

# Initialize model
 
def main():
    

    loaders = build_loaders()
    #show_images(loaders["train"])
    model = ResNet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = DDP(model)
    h = model.module.register_forward_hook(print_shape)
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
