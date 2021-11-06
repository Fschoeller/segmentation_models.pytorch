import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules

from sfnet_resnet import UperNetAlignHead
from nn.mynn import Norm2d, Upsample


class SFnetDecoder(nn.Module):

    def __init__(self, 
                num_classes,
                encoder_channels,
                fpn_dim=64,
                upsampling=8,):
        super().__init__()

        fpn_dsn = False
        self.fpn_dsn = fpn_dsn

        self.head = UperNetAlignHead(inplane=encoder_channels[-1], num_class=num_classes, norm_layer=Norm2d,
                                     fpn_inplanes=encoder_channels, fpn_dim=fpn_dim, conv3x3_type=flow_conv_type, fpn_dsn=fpn_dsn)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

    def forward(self, *features):
        x = self.head([features])
        x = self.upsample(x[0])
        return x