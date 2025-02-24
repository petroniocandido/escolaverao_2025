import torch
from torch import nn, optim
from torch.nn import functional as F


class Lenet5(nn.Module):
  def __init__(self):
    super(Lenet5,self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(5,5),stride=(1,1)),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=(1,1)),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=(1,1)),
        nn.Tanh()
    )

    self.flat = nn.Flatten()

    self.linear1 = nn.Sequential(
        nn.Linear(120,84),
        nn.Tanh()
    )

    self.linear2 = nn.Linear(84,10)

  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flat(x)
    x = self.linear1(x)
    x = self.linear2(x)

    return x


class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 2),
        nn.ReLU(),
        nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75, k = 2)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
        nn.ReLU()
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(384, 384, kernel_size = 3, padding = 1),
        nn.ReLU()
    )

    self.conv5 = nn.Sequential(
        nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
        nn.ReLU()
    )

    self.flat = nn.Flatten(1)

    self.linear1 = nn.Sequential(
        nn.Dropout(p = 0.5),
        nn.Linear(2304, 4096),
        nn.ReLU()
    )

    self.linear2 = nn.Sequential(
        nn.Dropout(p = 0.5),
        nn.Linear(4096, 4096),
        nn.ReLU()
    )

    self.linear3 = nn.Linear(4096, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.flat(x)
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    return x


class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    )

    self.conv5 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    self.conv6 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    self.conv7 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    )

    self.conv8 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.conv9 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.conv10 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    )

    self.conv11 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.conv12 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.conv13 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    )

    self.flat = nn.Flatten()

    self.linear1 = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    self.linear2 = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    self.linear3 = nn.Linear(4096, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)
    x = self.conv8(x)
    x = self.conv9(x)
    x = self.conv10(x)
    x = self.conv11(x)
    x = self.conv12(x)
    x = self.conv13(x)
    x = self.flat(x)
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    return x