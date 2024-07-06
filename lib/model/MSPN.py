import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.model.NAF_utils.arch_util import LayerNorm2d
from natten.natten2d import natten2dqkrpb, natten2dav


class MSPNLayer(nn.Module):
    def __init__(self, args, embed_dim, window_size=7, bias=True):
        super().__init__()
        self.args = args

        self.window_size = window_size
        self.pad = self.window_size // 2

        self.norm1 = LayerNorm2d(embed_dim)
        self.conv = nn.Conv2d(in_channels=embed_dim + 1, out_channels=embed_dim * 2, kernel_size=1, padding=0, bias=bias)

        # define a parameter table of relative position bias
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(1, (2 * self.window_size - 1), (2 * self.window_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, cd, sd, mask):
        g = self.norm1(g)
        z = torch.cat((g, cd), dim=1)
        qk = self.conv(z)
        q, k = qk.chunk(2, dim=1)
        v = cd

        B, C, H, W = q.shape

        query = q.view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        key = (k * mask).view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        attn = natten2dqkrpb(query, key, self.rpb, kernel_size=self.window_size, dilation=1)
        attn = self.softmax(attn)

        v_out = natten2dav(attn, v.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        cd_out = v_out.squeeze().view(cd.shape)
        if self.args.data_name == 'NYU':
            if self.args.mode == 'SDR':
                cd_out = cd_out * mask + cd * (1 - mask)
            cd_out[sd > 0] = sd[sd > 0]

        mask_out = natten2dav(attn, mask.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        mask_out = mask_out.squeeze().view(mask.shape)
        mask_out[sd > 0] = mask[sd > 0]

        return cd_out, mask_out


class MSPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embed_dim = args.embed_dim
        self.window_size = 13
        self.min_prop_time = args.prop_time
        self.kappa = 2

        self.mspn = MSPNLayer(self.args, embed_dim=self.embed_dim, window_size=self.window_size, bias=True)
        if args.mode == 'SDR':
            self.mspn_2nd = MSPNLayer(self.args, embed_dim=self.embed_dim, window_size=self.window_size, bias=True)

    def set_prop_times(self, sd, num_samples):
        B, C, H, W = sd.shape

        if self.args.data_name == 'NYU':
            avg_dist = (H * W / num_samples) ** 0.5 - 1
        else:
            # avg_dist = (H / num_samples) ** 0.5 - 1
            avg_dist = (H * W / num_samples) ** 0.5 - 1
        min_iter = torch.floor(avg_dist * self.kappa / (self.window_size // 2)) + 1

        prop_time = []
        for b in range(B):
            prop_time.append(int(max(self.min_prop_time, min_iter[b])))

        return prop_time

    def forward(self, pred_init, list_feat, list_mask, guide, dep, mask_init, num_samples=None, mask=None):
        B, _, Wh, Ww = pred_init.shape

        cd = pred_init
        if mask is None:
            mask = mask_init

        prop_times = self.set_prop_times(dep, num_samples)
        prop_times_1st = max(prop_times)

        for pt in range(prop_times_1st):
            cd, mask = self.mspn(guide, cd, dep, mask)

            cd_out = cd.contiguous()
            list_feat.append(cd_out)
            mask_out = mask.contiguous()
            list_mask.append(mask_out)

        if self.args.mode == 'SDR':
            mask = mask_init
            cd[dep > 0] = dep[dep > 0]
            for pt in range(6):
                cd, mask = self.mspn_2nd(guide, cd, dep, mask)

                cd_out = cd.contiguous()
                list_feat.append(cd_out)
                mask_out = mask.contiguous()
                list_mask.append(mask_out)

        return cd_out, list_feat, list_mask

