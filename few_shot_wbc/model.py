import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.fc1 = nn.Linear(256, 50)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x


class TorchVisionModel(nn.Module):
    def __ini__(self, name, pretrain):
        super(TorchVisionModel, self).__init__(name)
        name_list = ["alexnet", "vgg", "resnet", "densenet", "mobilenet", "resnext"]
        assert name in name_list
        assert pretrain in [True, False]
        if name == "alexnet":
            self._model = models.alexnet(pretrained=pretrain)
        elif name == "vgg":
            self._model = models.vgg19(pretrained=pretrain)
        elif name == "resnet":
            self._model = models.resnet152(pretrained=pretrain)
        elif name == "densenet":
            self._model = models.densenet161(pretrained=pretrain)
        elif name == "mobilenet":
            self._model = models.mobilenet_v2(pretrained=pretrain)
        elif name == "resnext":
            self._model = models.resnext101_32x8d(pretrained=pretrain)

    def forward(self, x):
        return self._model(x)
