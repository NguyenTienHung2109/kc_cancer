import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import FocalLoss

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, mode='binary', ignore_index=-1, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma,mode=mode, ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits, targets):
        return self.focal_loss(logits, targets)
