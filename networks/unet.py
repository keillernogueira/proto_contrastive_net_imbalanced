import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels, skip_layers='1_2_3_4', classif=None):
        super(UNet, self).__init__()
        self.skip_layers = skip_layers
        self.classif = classif

        self.enc1 = _EncoderBlock(input_channels, 64)  # 64
        self.enc2 = _EncoderBlock(64, 128)  # 128
        self.enc3 = _EncoderBlock(128, 256)  # 256
        self.enc4 = _EncoderBlock(256, 512, dropout=True)  # 512

        self.center = _DecoderBlock(512, 1024, 512)  # 512

        self.dec4 = _DecoderBlock(1024, 512, 256)  # 256
        self.dec3 = _DecoderBlock(512, 256, 128)  # 128
        self.dec2 = _DecoderBlock(256, 128, 64)  # 64

        self.dec1 = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        if classif:
            if self.skip_layers == '1_4' or self.skip_layers == '2_4':
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(64 + 256, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
            elif self.skip_layers == '1_2_3_4':
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(64 + 64 + 128 + 256, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
            self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        initialize_weights(self)

    def forward(self, x, feat=False):
        enc1 = self.enc1(x)
        # print('enc1', enc1.size())
        enc2 = self.enc2(enc1)
        # print('enc2', enc2.size())
        enc3 = self.enc3(enc2)
        # print('enc3', enc3.size())
        enc4 = self.enc4(enc3)
        # print('enc4', enc4.size())

        center = self.center(enc4)
        # print('center', center.size())

        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        # print('dec4', dec4.size())
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        # print('dec3', dec3.size())
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        # print('dec2', dec2.size())
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        # print('dec1', dec1.size())

        if self.skip_layers == '1_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([dec1,
                                  F.interpolate(dec4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        elif self.skip_layers == '2_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(dec2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(dec4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        elif self.skip_layers == '1_2_3_4':
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([dec1,
                                  F.interpolate(dec2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(dec3, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(dec4, x.size()[2:], mode='bilinear', align_corners=False)], 1)

        output = None
        if self.classif:
            if self.skip_layers is not None:
                classif1 = self.classifier1(fv_final)
                output = self.final(classif1)
            else:
                final = self.final(dec1)

        return output, fv_final