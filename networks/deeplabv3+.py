import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from utils import initialize_weights


# https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master/network
class FCNEfficientNetB0(nn.Module):
    def __init__(self, num_classes, pretrained=True, skip=True, classif=True):
        super(FCNEfficientNetB0, self).__init__()
        self.skip = skip
        self.classif = classif

        # load pre-trained model
        efficientnet_b0 = models.efficientnet_b0(pretrained=pretrained, progress=False)

        self.layer0 = efficientnet_b0.features[0]  # output feat = 32
        self.layer1 = efficientnet_b0.features[1]  # output feat = 16
        self.layer2 = efficientnet_b0.features[2]  # output feat = 24
        self.layer3 = efficientnet_b0.features[3]  # output feat = 40
        self.layer4 = efficientnet_b0.features[4]  # output feat = 80
        self.layer5 = efficientnet_b0.features[5]  # output feat = 112
        self.layer6 = efficientnet_b0.features[6]  # output feat = 192
        self.layer7 = efficientnet_b0.features[7]  # output feat = 320
        self.layer8 = efficientnet_b0.features[8]  # output feat = 1280

        if self.classif:
            if self.skip:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(2096, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
            else:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(1280, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
            self.final = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)

        if not pretrained:
            initialize_weights(self)
        elif self.classif:
            initialize_weights(self.classifier1)
            initialize_weights(self.final)

    def forward(self, x):
        fv0 = self.layer0(x)
        fv1 = self.layer1(fv0)
        fv2 = self.layer2(fv1)
        fv3 = self.layer3(fv2)
        fv4 = self.layer4(fv3)
        fv5 = self.layer5(fv4)
        fv6 = self.layer6(fv5)
        fv7 = self.layer7(fv6)
        fv8 = self.layer8(fv7)

        if self.skip:
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(fv0, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv1, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv3, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv5, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv6, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv7, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv8, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        else:
            # Forward on FCN without Skip Connections.
            fv_final = F.interpolate(fv8, x.size()[2:], mode='bilinear', align_corners=False)

        output = None
        if self.classif:
            classif1 = self.classifier1(fv_final)
            output = self.final(classif1)

        return output, torch.cat([F.interpolate(fv0, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv1, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv3, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv5, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv6, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv7, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv8, x.size()[2:], mode='bilinear', align_corners=False)], 1)
