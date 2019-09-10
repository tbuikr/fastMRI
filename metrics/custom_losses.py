from torch import nn
import torch.nn.functional as F

from metrics.ssim import ssim
from metrics.tv_loss import TVLoss

class CSSIM(nn.Module):  # Complementary SSIM
    def __init__(self, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val
        return 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size,
                             sigma=self.sigma, reduction=self.reduction)


class L1CSSIM(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)

        return cssim + self.l1_weight * l1_loss


class L1CSSIMTV(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean', tvloss_weight=1e-4, p=2):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction
        self.tvloss_weight = tvloss_weight
        self.p = p

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim(input, target, max_val=max_val, filter_size=self.filter_size, sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)
        
        tv_loss = TVLoss(input, self.tvloss_weight, self.p)

        return cssim + self.l1_weight * l1_loss + tv_loss
