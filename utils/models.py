from torch import nn
import torchvision
import torchvision.models as models
from utils.train import *


class LeNet(nn.Module):
    def __init__(self, name = 'LeNet', distilation = False):
        super().__init__()

        self.name = name
        self.distilation = distilation

        self.conv1      = nn.Conv2d(1, 6, kernel_size=5, padding=2) # nn.Sigmoid()
        self.avgPool1   = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2      = nn.Conv2d(6, 16, kernel_size=5)           # nn.Sigmoid()
        self.avgPool2   = nn.AvgPool2d(kernel_size=2, stride=2)
        # nn.Flatten()
        self.linear1    = nn.Linear(16 * 5 * 5, 120)                # nn.Sigmoid()
        self.linear2    = nn.Linear(120, 84)                        # nn.Sigmoid()

        self.head = nn.Linear(84, 10)
        self.head_dist = nn.Linear(84, 10)

        init_model(self)

    def forward(self, x):
        Y = self.conv1(x)
        Y = nn.Sigmoid()(Y)
        Y = self.avgPool1(Y)
        Y = self.conv2(Y)
        Y = nn.Sigmoid()(Y)
        Y = self.avgPool2(Y)
        Y = nn.Flatten()(Y)
        Y = self.linear1(Y)
        Y = nn.Sigmoid()(Y)
        Y = self.linear2(Y)
        Y = nn.Sigmoid()(Y)

        if self.distilation:
            x = self.head(Y), self.head_dist(Y)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(Y)
        return x


class CifarResNet(nn.Module):
    def __init__(self, name = "CifarResNet") -> None:
        super().__init__()

        self.name = name
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        init_layer(self.model.fc)

    def forward(self, X):
        if not X.size(dim=-1) == 224:
            X = torchvision.transforms.Resize(224)(X)

        return self.model(X)


class FashionMnistResNet(nn.Module):
    def __init__(self, name = "FashionMnistResNet", pretrained = True) -> None:
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.name = name
        self.model = models.resnet18(weights=weights)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        if pretrained:
            init_layer(self.model.conv1)
            init_layer(self.model.fc)
        else:
            init_model(self)

    def forward(self, X):
        if not X.size(dim=-1) == 224:
            X = torchvision.transforms.Resize(224)(X)

        return self.model(X)

