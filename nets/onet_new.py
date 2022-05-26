import torch
import torch.nn as nn
from torchsummary import summary

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(2,2),bias=False)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(1,1),bias=True)

        self.avepool = nn.AvgPool2d(9)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        x = self.avepool(x)



        return x


if __name__ == '__main__':
    net = MyNet()
    summary(net,(3,48,48),batch_size=1,device="cpu")
