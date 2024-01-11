import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from utils import initialize_weights


class FCNDenseNet121(nn.Module):
    def __init__(self, num_classes, input_channels=None, pretrained=True, skip_layers='1_2_3_4', classif=False):
        super(FCNDenseNet121, self).__init__()
        self.skip_layers = skip_layers
        self.classif = classif

        # DenseNet with Skip Connections (adapted from FCN-8s).
        densenet = models.densenet121(pretrained=pretrained, progress=False)

        if pretrained:
            self.init = densenet.features[:4]
        else:
            self.init = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.dense1 = densenet.features[4:6]  # output feat = 128
        self.dense2 = densenet.features[6:8]  # output feat = 256
        self.dense3 = densenet.features[8:10]  # output feat = 512
        self.dense4 = densenet.features[10:12]  # output feat = 1024

        if self.classif:
            if self.skip_layers == '1_4':
                feat_in = 128 + 1024
                # self.classifier1 = nn.Sequential(
                #     nn.Conv2d(128 + 1024, 128, kernel_size=3, padding=1),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(),
                #     nn.Dropout2d(0.5),
                # )
            elif self.skip_layers == '2_4':
                feat_in = 256 + 1024
                # self.classifier1 = nn.Sequential(
                #     nn.Conv2d(256 + 1024, 128, kernel_size=3, padding=1),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(),
                #     nn.Dropout2d(0.5),
                # )
            elif self.skip_layers == '1_2_3_4':
                feat_in = 128 + 256 + 512 + 1024
                # self.classifier1 = nn.Sequential(
                #     nn.Conv2d(128 + 256 + 512 + 1024, 128, kernel_size=3, padding=1),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(),
                #     nn.Dropout2d(0.5),
                # )
            else:
                feat_in = 1024
                # self.classifier1 = nn.Sequential(
                #     nn.Conv2d(1024, 128, kernel_size=3, padding=1),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(),
                #     nn.Dropout2d(0.5),
                # )
            self.final = nn.Conv2d(feat_in, num_classes, kernel_size=3, padding=1)

        if not pretrained:
            initialize_weights(self)
        elif self.classif:
            # initialize_weights(self.classifier1)
            initialize_weights(self.final)

    def forward(self, x, feat=False):
        fv_init = self.init(x)
        fv1 = self.dense1(fv_init)
        fv2 = self.dense2(fv1)
        fv3 = self.dense3(fv2)
        fv4 = self.dense4(fv3)

        if self.skip_layers == '1_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(fv1, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        elif self.skip_layers == '2_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(fv2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        elif self.skip_layers == '1_2_3_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(fv1, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv3, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        else:
            # Forward on FCN without Skip Connections.
            fv_final = F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)

        output = None
        if self.classif:
            # classif1 = self.classifier1(fv_final)
            output = self.final(fv_final)

        return output, fv_final
