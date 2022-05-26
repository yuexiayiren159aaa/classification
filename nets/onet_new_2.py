import torch.nn as nn
from torchsummary import summary
import torch

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))  # 48*48*3 -> 46*46*32
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)          # 48*48*32 ->
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)) #
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.prelu4 = nn.PReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)


        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=2)

        # self.flatten = nn.Flatten()

        # self.fc = nn.Linear(in_features=576, out_features=256)
        # self.dropout = nn.Dropout(0.2)
        # self.class_fc = nn.Linear(in_features=256, out_features=2)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))


        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.fc(x)

        # x = self.flatten(x)
        # x = self.fc(x)
        # x = self.dropout(x)
        # # 分类是否人脸的卷积输出层
        # class_out = self.class_fc(x)




        return x



def newNet():
    model = NewNet()

    return model



# if __name__ == '__main__':
    # net = ONet()
    # summary(net, (3, 48, 48), batch_size=1, device="cpu")
    # input = torch.randn(1,3,48,48)
    # output = net(input)
    # print(output.shape)
