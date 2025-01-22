import torch
import torch.nn as nn
from torchvision.models import resnet18,resnet34

class ResNet18Modified(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Modified, self).__init__()
        self.model = resnet18(pretrained=True)


        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


        self.conv1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )  # Conv1 + MaxPool
        self.layer1 = self.model.layer1  # Residual Block 1
        self.layer2 = self.model.layer2  # Residual Block 2
        self.layer3 = self.model.layer3  # Residual Block 3
        self.layer4 = self.model.layer4  # Residual Block 4

    def forward(self, x):
        return self.model(x)

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.layer1(x) #[1, 64, 56, 56]
        return x
    def fea_part2_0(self, x):
        x= self.layer2(x)  # [1, 128, 28, 28]

        return x
    def fea_part2_1(self, x):
        x=self.layer3(x)  # [1, 256, 14, 14]
        return x
    def fea_part3(self, x):
        x=self.layer4(x)  # [1, 512, 7, 7]
        return x
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )  # Conv1 + MaxPool
        self.layer1 = self.model.layer1  # Residual Block 1
        self.layer2 = self.model.layer2  # Residual Block 2
        self.layer3 = self.model.layer3  # Residual Block 3
        self.layer4 = self.model.layer4  # Residual Block 4

    def forward(self, x):
        return self.model(x)

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.layer1(x) #[1, 64, 56, 56]
        return x
    def fea_part2_0(self, x):
        x= self.layer2(x)  # [1, 128, 28, 28]

        return x
    def fea_part2_1(self, x):
        x=self.layer3(x)  # [1, 256, 14, 14]
        return x
    def fea_part3(self, x):
        x=self.layer4(x)  # [1, 512, 7, 7]
        return x
if __name__ == '__main__':
    # model = ResNet18Modified(5)
    # # model = model.cuda()
    # # from torchsummary import summary
    # # input_s = (3, image_size, image_size)
    # # print(summary(model, input_s))
    # dummy = torch.rand(10, 4, 64, 64)
    # out = model.fea_part1(dummy)
    # print(out.size())
    # out = model.fea_part2_0(out)
    # print(out.size())
    # out = model.fea_part2_1(out)
    # print(out.size())
    # out = model.fea_part3(out)
    # print(out.size())

    model = ResNet18(5)
    # model = model.cuda()
    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s))
    dummy = torch.rand(10,3, 224, 224)
    out = model.fea_part1(dummy)
    print(out.size())
    out = model.fea_part2_0(out)
    print(out.size())
    out = model.fea_part2_1(out)
    print(out.size())
    out = model.fea_part3(out)
    print(out.size())