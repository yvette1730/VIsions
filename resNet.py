import torch
from torch import nn
from torch.nn import functional as F

def print_shape(model,input, output):
    print(model.__class__.__name__)
    print(f'input:{input[0].shape}|output:{output[0].shape}')
    print()


class Res_Block(nn.Module):  # @save
    """The Residual block of ResNet models."""

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        #print(f'input shape: {X.shape}')
        #print(Y.shape)
        #print(Y.shape)
        #print(Y.shape)
        Y = self.bn2(self.conv2(Y))
       # print(X.get_device())
        #print(Y.get_device())

        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=5):
        super(ResNet, self).__init__()
        self.arch = arch
        self.prev_channels = self.arch[0][1]
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f"b{i+2}", self.block(*b, first_block=(i == 0)))
        self.net.add_module(
            "last",
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes)
            ),
        )

    def b1(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Res_Block(self.prev_channels, num_channels, use_1x1conv=True, strides=2)
                )
            else:
                blk.append(Res_Block(self.prev_channels, num_channels))
            self.prev_channels = num_channels
        return nn.Sequential(*blk)

    def forward(self, X):
        return self.net(X)


class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=5):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes)

    def forward(self, X):
        return self.net(X)


def test():
    """docstring"""

    r18 = ResNet18()

    imgs = torch.ones(size=(32,3,256,256))

    print(imgs)
    print(imgs.shape)

    print(r18(imgs))
    #print(X.get_device())
    #print(Y.get_device())

def main():
    """docstring"""

    test()

if __name__ == '__main__': 
    main()
