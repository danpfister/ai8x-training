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


class AI85LanguageNet(nn.Module):
    def __init__(
            self,
            num_classes=3,
            num_channels=3,
            dimensions=(64, 79),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(num_channels, 16, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 16, 3, pool_size=2, pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(16, 8, 3, stride=1, padding=1, bias=bias,
                                          **kwargs)
        self.linear = ai8x.Linear(8*8*13, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def ai85languagenet(pretrained=False, **kwargs):
    assert not pretrained
    return AI85LanguageNet(**kwargs)


models = [
    {
        'name': 'ai85languagenet',
        'min_input': 1,
        'dim': 2,
    },
]
