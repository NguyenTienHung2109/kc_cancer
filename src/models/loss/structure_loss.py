import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StructureLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, mask: Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        
        return (wbce + wiou).mean()
    
class MulticlassStructureLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, mask: Tensor):
        # pred: (b, c, w, h)
        # mask: (b, c, w, h)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) # (b,c,w,h)
        
        # Dice Loss
        pred_softmax = torch.softmax(pred, dim=1) # (b, c, w, h)
        inter = (pred_softmax * mask * weit).sum(dim=(-2, -1))
        union = ((pred_softmax + mask) * weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        # Weighted Cross-Entropy Loss
        weit = weit.max(dim=1)[0] # (b, w, h)
        print(pred.shape, mask.shape, weit.shape)
        ce = F.cross_entropy(pred, mask, reduction='none')  # (b, w, h)
        wce = (weit * ce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))  # (b, )
        
        # Total Loss
        return (wce.mean() + wiou.mean())