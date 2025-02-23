import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class AFC_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: Tensor, mask: Tensor):
        pred = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(pred, mask, reduction='none')
        focal = self.alpha * (1 - pred) ** self.gamma * bce
        
        return focal.mean()