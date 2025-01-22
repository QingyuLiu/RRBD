import torch
import torch.nn as nn
import torch.nn.functional as F
from two_stream_model.resnet_modified import ResNet18Modified,ResNet18
from two_stream_model.linear_fusion import HdmProdBilinearFusion
from two_stream_model.attention_new import NewDualCrossModalAttention

class Two_Stream_Net(nn.Module):
    def __init__(self,num_classes = 2):
        super(Two_Stream_Net, self).__init__()
        
        self.resnet_bias = ResNet18Modified(num_classes=2)
        self.resnet_rgb = ResNet18(num_classes=2)

        # self.resnet_align_up4 = nn.Conv2d(512, 2048, kernel_size=1)



        self.HBFusion = HdmProdBilinearFusion(512, dim2=512,
                                              hidden_dim=1024,
                                              output_dim=512)
        self.dcma1 = NewDualCrossModalAttention(64,64, xsize=56, ysize=16)
        self.dcma2 = NewDualCrossModalAttention(128,128, xsize=28, ysize=8)
        self.dcma3 = NewDualCrossModalAttention(256,256, xsize=14, ysize=4)

        self.spatial_adjust = lambda x: F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)

        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def features(self, diff, rgb):
        # srm = self.srm_conv0(x)


        x0 = self.resnet_rgb.fea_part1(rgb)
        y0 = self.resnet_bias.fea_part1(diff)
        x0, y0 = self.dcma1(x0, y0)

        x1 = self.resnet_rgb.fea_part2_0(x0)
        y1 = self.resnet_bias.fea_part2_0(y0)
        x1, y1 = self.dcma2(x1, y1)

        x2 = self.resnet_rgb.fea_part2_1(x1)
        y2 = self.resnet_bias.fea_part2_1(y1)
        x2, y2 = self.dcma3(x2,y2)

        x3 = self.resnet_rgb.fea_part3(x2)
        y3 = self.resnet_bias.fea_part3(y2)

        return self.HBFusion(x3, self.spatial_adjust(y3))

    def forward(self, noise, frequency):
        feas = self.features(noise, frequency)
        preds = self.cls_header(feas)

        return feas, preds
if __name__ == '__main__':
    model = Two_Stream_Net()
    # model = model.cuda()
    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s))
    dummy1 = torch.rand(10, 4, 64, 64)
    dummy2 = torch.rand(10, 3, 224, 224)
    out = model.forward(dummy1,dummy2)