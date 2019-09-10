import torch
import torch.nn as nn
import math
import numbers
from torch.nn import functional as F
import numpy as np

#class TVLoss(nn.Module):
#     def __init__(self, tvloss_weight=0.1, p=1):
#         super(TVLoss, self).__init__()
#         self.tvloss_weight = tvloss_weight
#         assert p in [1, 2]
#         self.p = p

def TVLoss(x, tvloss_weight, p):    
    if p == 1:
        loss = torch.sum(torch.abs(x[:,:-1,:] - x[:,1:,:])) + torch.sum(torch.abs(x[:,:,:-1] - x[:,:,1:]))
    else:
        loss = torch.sum(torch.sqrt((x[:,:-1,:] - x[:,1:,:])**2) + torch.sum((x[:,:,:-1] - x[:,:,1:])**2))
            
    loss = loss / x.size(0) / (x.size(1)-1) / (x.size(2)-1)
    return tvloss_weight * 2 *loss


