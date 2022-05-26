import torch.nn as nn
from torchsummary import summary
import torch

class ONet_Imagenet(nn.Module):
    def __init__(self):
        super(ONet_Imagenet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))  # 48*48*3 -> 46*46*32
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)          # 48*48*32 -> 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)) # 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)          
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2))
        self.prelu4 = nn.PReLU()
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(in_features=1152, out_features=256)
        self.fc = nn.Linear(in_features=40000, out_features=256)
        self.dropout = nn.Dropout(0.2)
        self.class_fc = nn.Linear(in_features=256, out_features=1000)
        # self.bbox_fc = nn.Linear(in_features=256, out_features=4)
        # self.landmark_fc = nn.Linear(in_features=256, out_features=10)

        # self.fc = nn.Conv2d(in_channels=64, out_channels=256,kernel_size=(1,1))
        # self.class_fc = nn.Conv2d(256,2,1)

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
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)

        # class_out = class_out.view(class_out.size(0),-1)
        # 人脸box的回归卷积输出层
        # bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        # landmark_out = self.landmark_fc(x)
        # return class_out, bbox_out, landmark_out
        return class_out

    # def freeze_backbone(self):
    #     for param in self.features.parameters():
    #         param.requires_grad = False
    #
    # def Unfreeze_backbone(self):
    #     for param in self.features.parameters():
    #         param.requires_grad = True

def onet_Imagenet():
    model = ONet_Imagenet()

    return model



if __name__ == '__main__':
    net = ONet_Imagenet()
    summary(net, (3, 224, 224), batch_size=1, device="cpu")
    # input = torch.randn(1,3,48,48)
    # output = net(input)
    # print(output.shape)
