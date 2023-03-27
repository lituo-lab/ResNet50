import torch
import torch.nn as nn
import cv2 as cv

class Bottleneck(nn.Module):

    def __init__(self, in_planes, mid_planes):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(mid_planes, in_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Bottle_top(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride):
        super(Bottle_top, self).__init__()

        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn0 = nn.BatchNorm2d(out_planes)

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.bn0(self.conv0(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        # 3,224,224 -> 64,112,112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 64,112,112 -> 64,56,56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 64,56,56 -> 256,56,56
        self.layer1 = nn.Sequential(
            Bottle_top(in_planes=64, mid_planes=64, out_planes=256, stride=1),
            Bottleneck(in_planes=256, mid_planes=64),
            Bottleneck(in_planes=256, mid_planes=64))

        # 256,56,56 -> 512,28,28
        self.layer2 = nn.Sequential(
            Bottle_top(in_planes=256, mid_planes=128, out_planes=512, stride=2),
            Bottleneck(in_planes=512, mid_planes=128),
            Bottleneck(in_planes=512, mid_planes=128),
            Bottleneck(in_planes=512, mid_planes=128))

        # 512,28,28 -> 1024,14,14  到这里可以获得一个1024,14,14的共享特征层
        self.layer3 = nn.Sequential(
            Bottle_top(in_planes=512, mid_planes=256, out_planes=1024, stride=2),
            Bottleneck(in_planes=1024, mid_planes=256),
            Bottleneck(in_planes=1024, mid_planes=256),
            Bottleneck(in_planes=1024, mid_planes=256),
            Bottleneck(in_planes=1024, mid_planes=256),
            Bottleneck(in_planes=1024, mid_planes=256))

        # self.layer4被用在classifier模型中 1024,14,14 -> 2048,7,7
        self.layer4 = nn.Sequential(
            Bottle_top(in_planes=1024, mid_planes=512, out_planes=2048, stride=2),
            Bottleneck(in_planes=2048, mid_planes=512),
            Bottleneck(in_planes=2048, mid_planes=512))

        # 2048,7,7 -> 2048,1,1
        self.avgpool = nn.AvgPool2d(7)

        # 2048 -> 1000
        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    model = ResNet50()
    
    img = cv.imread('horses.jpg')
    img = cv.resize(img, (224, 224))
    img = torch.tensor(img).unsqueeze(dim=0)
    img = img.permute(0, 3, 1, 2).to(torch.float32)

    model.eval()
    out = model(img)

