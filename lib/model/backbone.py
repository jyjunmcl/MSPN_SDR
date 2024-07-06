import torch
import torch.nn as nn
import torch.nn.functional as F
from .NAFNet_arch import NAFBlock
from .pvt import PVT

from lib.model.NAF_utils.arch_util import LayerNorm2d


def conv_ln_relu(ch_in, ch_out, kernel, stride=1, padding=0, ln=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not ln))
    if ln:
        layers.append(LayerNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_ln_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  ln=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not ln))
    if ln:
        layers.append(LayerNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.args = args

        # Encoder
        self.conv1_rgb = conv_ln_relu(3, 48, kernel=3, stride=1, padding=1,
                                      ln=False)
        self.conv1_dep = conv_ln_relu(1, 16, kernel=3, stride=1, padding=1,
                                      ln=False)

        if self.args.mode == 'conventional':
            self.conv1 = conv_ln_relu(64, 64, kernel=3, stride=1, padding=1,
                                      ln=False)
        else:
            self.conv1_rgb_dep = conv_ln_relu(64, 64, kernel=3, stride=1, padding=1,
                                              ln=False)
            self.conv1_dep_MD = conv_ln_relu(1, 16, kernel=3, stride=1, padding=1,
                                             ln=False)
            self.conv1 = conv_ln_relu(64 + 16, 64, kernel=3, stride=1, padding=1,
                                      ln=False)

        self.spatial_conv3x3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.spatial_conv1x1 = nn.Conv2d(48, 48, kernel_size=1)

        self.former = PVT(in_chans=64, patch_size=2, pretrained='lib/pretrained/pvt.pth', )

        channels = [64, 128, 64, 128, 320, 512]
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_ln_relu(channels[5], 256, kernel=3, stride=2,
                          padding=1, output_padding=1),
            NAFBlock(256),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_ln_relu(256 + channels[4], 128, kernel=3, stride=2,
                          padding=1, output_padding=1),
            NAFBlock(128),

        )
        # 1/4
        self.dec4 = nn.Sequential(
            convt_ln_relu(128 + channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            NAFBlock(64),
        )

        # 1/2
        self.dec3 = nn.Sequential(
            convt_ln_relu(64 + channels[2], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            NAFBlock(64),
        )

        # 1/1
        self.dec2 = nn.Sequential(
            convt_ln_relu(64 + channels[1], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            NAFBlock(64),
        )

        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_ln_relu(64 + channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_ln_relu(64 + 64, args.embed_dim, kernel=3, stride=1,
                                    padding=1, ln=False, relu=False)

        if self.args.mode == 'conventional':
            # Init Depth Branch
            # 1/1
            self.dep_dec1 = conv_ln_relu(64 + 64, 64, kernel=3, stride=1,
                                         padding=1)
            self.dep_dec0 = conv_ln_relu(64 + 64, 1, kernel=3, stride=1,
                                         padding=1, ln=False, relu=True)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb=None, depth=None, depth_MD=None):
        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_rgb = self.spatial_conv3x3(fe1_rgb) - self.spatial_conv1x1(fe1_rgb)

        fe1_dep = self.conv1_dep(depth)

        if self.args.mode == 'conventional':
            fe1 = self.conv1(torch.cat((fe1_rgb, fe1_dep), dim=1))
        elif self.args.mode == 'SDR':
            fe1_rgb_dep = self.conv1_rgb_dep(torch.cat((fe1_rgb, fe1_dep), dim=1))
            fe1_dep_MD = self.conv1_dep_MD(depth_MD)
            fe1 = self.conv1(torch.cat((fe1_rgb_dep, fe1_dep_MD), dim=1))

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        if self.args.mode == 'conventional':
            guide = self.gd_dec0(self._concat(gd_fd1, fe1))
            # Init Depth Decoding
            dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
            init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        elif self.args.mode == 'SDR':
            guide = self.gd_dec0(self._concat(gd_fd1, fe1_rgb_dep))
            init_depth = None

        return init_depth, guide

