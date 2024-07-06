"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

import torch
import torch.nn as nn


class SDR(nn.Module):
    def __init__(self, args, backbone, spn):
        super(SDR, self).__init__()

        self.args = args
        self.backbone = backbone(args)

        if self.args.prop_time > 0:
            self.prop_layer = spn(args)

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        dep_MD = sample['dep_MD']
        mask_init = sample['mask_init']
        num_sample = sample['num_sample']

        init_depth, guide = self.backbone(rgb, dep, dep_MD)
        mask = (dep > 0) * 1.0
        if self.args.mode == 'conventional':
            pred_init = init_depth * (1 - mask) + dep * mask
        else:
            pred_init = dep_MD * (1 - mask) + dep * mask

        # Diffusion
        y_inter = [pred_init, ]
        mask_inter = [mask_init, ]
        if self.args.prop_time > 0:
            y, y_inter, mask_inter = self.prop_layer(pred_init, y_inter, mask_inter, guide, dep, mask_init, num_sample)
        else:
            y = pred_init

        # Remove negative depth
        y = torch.clamp(y, min=0)
        # best at first
        y_inter.reverse()
        mask_inter.reverse()

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter, 'mask_inter': mask_inter,
                  'guidance': guide, 'num_sample': num_sample}

        return output
