###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
from torch import nn

import ai8x


class AI85LanguageNetLarge(nn.Module):
    def __init__(
            self,
            num_classes=3,
            num_channels=3,
            dimensions=(64, 108),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(32, 64, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 32, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedMaxPoolConv2dReLU(32, 8, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.linear1 = ai8x.FusedLinearReLU(8*8*13, 128)
        self.linear2 = ai8x.Linear(128, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x) # 16x64x108
        x = self.conv2(x) # 32x64x108
        x = self.conv3(x) # 64x64x108
        x = self.conv4(x) # 64x32x54
        x = self.conv5(x) # 32x16x27
        x = self.conv6(x) # 8x8x13
        x = x.view(x.size(0), -1)
        x = self.linear1(x) # 128
        x = self.linear2(x) # 3
        return x

class AI85LanguageNetSmall(nn.Module):
    def __init__(
            self,
            num_classes=3,
            num_channels=3,
            dimensions=(16, 251),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(32, 16, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(16, 4, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.linear1 = ai8x.FusedLinearReLU(4*4*62, 128, bias=True)
        self.linear2 = ai8x.Linear(128, num_classes, wide=True, bias=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x) # 16x16x251
        x = self.conv2(x) # 32x16x251
        x = self.conv4(x) # 16x8x125
        x = self.conv5(x) # 8x4x62
        x = x.view(x.size(0), -1)
        x = self.linear1(x) # 128
        x = self.linear2(x) # 3
        return x

def ai85languagenet(pretrained=False, **kwargs):
    assert not pretrained
    return AI85LanguageNetSmall(**kwargs)


models = [
    {
        'name': 'ai85languagenet',
        'min_input': 1,
        'dim': 2,
    },
]
