"""
    CompletionFormer
    ======================================================================

    SI log loss implementation
"""


import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self, args):
        super(SILogLoss, self).__init__()
        self.variance_focus = 0.85
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        relu = nn.ReLU()
        pred = relu(pred - 0.0000000001) + 0.0000000001
        gt = relu(gt - 0.0000000001) + 0.0000000001

        mask = (gt > self.t_valid).to(torch.bool)

        d = torch.log(pred[mask]) - torch.log(gt[mask])

        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
