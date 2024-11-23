import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        
        self.smooth = smooth
    
    def forward(self, pred: Tensor, mask: Tensor):    
        pred = torch.sigmoid(pred)
        intersection = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        dice = 1 - (2*intersection + self.smooth)/(union + self.smooth)
        
        return dice.mean()