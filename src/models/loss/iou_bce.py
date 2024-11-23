import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, JaccardLoss

class IOU_BCE(nn.Module):
    def __init__(self, smooth=1, epsilon=1e-8):
        super().__init__()
        
        self.smooth = smooth
        self.epsilon = epsilon
    
    def forward(self, pred: Tensor, mask: Tensor):    
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + self.smooth)/(union - inter + self.smooth)

        return (wbce + wiou).mean()
    
class IoU_BCE_Loss(nn.Module):

    def __init__(self) -> None:
        self.iou = JaccardLoss(mode="binary")
        self.bce = SoftBCEWithLogitsLoss()

    def forward(self, logits: Tensor, targets: Tensor):
        return (self.iou(logits, targets) + self.bce(logits, targets)) / 2