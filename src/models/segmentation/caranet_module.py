from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

import rootutils
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torch.optim import Optimizer, lr_scheduler
from torchmetrics.classification import Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch.nn.functional as F
from src.models.loss import IOU_BCE

from segmentation_models_pytorch.losses import (SoftCrossEntropyLoss,
                                                SoftBCEWithLogitsLoss, 
                                                DiceLoss, 
                                                FocalLoss,
                                                JaccardLoss)

from src.models.loss import StructureLoss

from contextlib import contextmanager

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma

# def structure_loss(pred, mask):
    
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
    
#     return (wbce + wiou).mean()


# def ange_structure_loss(pred, mask, smooth=1):

#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + smooth)/(union - inter + smooth)

#     threshold = torch.tensor([0.5])
#     threshold = threshold
#     pred = pred
#     pred = (pred > threshold).float() * 1
#     pred = pred

#     pred = pred.data.cpu().numpy().squeeze()
#     pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
#     # print(type(pred))
#     return (wbce + wiou).mean(), torch.from_numpy(pred)


def dice_loss_coff(pred, target, smooth = 0.0001):

    num = target.size(0)
    pred_cont = pred.contiguous()
    target_cont = target.contiguous()

    intersection = (pred_cont * target_cont).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred_cont.sum(dim=2).sum(dim=2) + target_cont.sum(dim=2).sum(dim=2) + smooth)

    return loss.sum()/num


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class CaraNetModule(LightningModule):
    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        # criterion: nn.Module,
        size_rates = [0.75, 1, 1.25],
        image_size:int = 352,
        use_ema: bool = False,
    ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # CaraNet model
        self.net = net
        print(self.net.resnet.conv1)

        # assert isinstance(criterion, (SoftCrossEntropyLoss, 
        #                               SoftBCEWithLogitsLoss, 
        #                               DiceLoss, 
        #                               StructureLoss,
        #                               IOU_BCE,
        #                               FocalLoss,
        #                               JaccardLoss)), \
        #     NotImplementedError("Only implemented for [CrossEntropyLoss, SoftBCEWithLogitsLoss, \
        #                         DiceLoss, FocalLoss, JaccardLoss, StructureLoss]")
        
        # # loss function
        # self.criterion = criterion

        # multi-scale training
        self.size_rates = size_rates
        self.image_size = image_size

        # metric objects for calculating and averaging accuracy across batches
        self.train_dice = Dice(ignore_index=0)
        self.val_dice = Dice(ignore_index=0)
        self.test_dice = Dice(ignore_index=0)

        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()
        self.test_iou = BinaryJaccardIndex()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_loss1 = MeanMetric()
        self.train_loss2 = MeanMetric()
        self.train_loss3 = MeanMetric()
        self.train_loss5 = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_dice_best = MaxMetric()

        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

        # manual optimization for multi-scale training
        self.automatic_optimization = False

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.net)
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: Two tensor of noise
        """

        return self.net(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store accuracy from these checks
        self.val_loss.reset()
        self.val_dice_best.reset()

    def model_step(self, batch: Any):
        images, gts = batch

        pred_map_5, _, _, _ = self.forward(images)

        # loss = self.criterion(pred_map_5, gts)
        loss_fn = IOU_BCE()
        loss = loss_fn(pred_map_5, gts)
        preds = torch.sigmoid(pred_map_5)
        preds = (preds > 0.5).to(torch.int64)
        gts = (gts > 0.5).to(torch.int64)

        return loss, preds, gts

    @torch.no_grad()
    def predict(self, x: Tensor, threshold: float = 0.5, dtype: torch.dtype = torch.float32) -> Tensor:
        # return mask
        if self.use_ema:
            with self.ema_scope():
                logits, _, _, _ = self.net(x)
        else:
            logits, _, _, _ = self.net(x)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).to(dtype)

        return preds
    
    def training_step(self, batch: Any, batch_idx: int):
        images, gts = batch

        # multi-scale training
        for rate in self.size_rates:
            # manual optimization
            opt = self.optimizers()
            opt.zero_grad()

            # rescale
            trainsize = int(round(self.image_size*rate/32)*32)
            if rate != 1:
                imgs = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                targets = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            else:
                imgs = images
                targets = gts

            # forward
            pred_map_5, pred_map_3, pred_map_2, pred_map_1 = self.forward(imgs)
            loss_fn = IOU_BCE()
            # loss function
            loss5 = loss_fn(pred_map_5, targets)
            loss3 = loss_fn(pred_map_3, targets)
            loss2 = loss_fn(pred_map_2, targets)
            loss1 = loss_fn(pred_map_1, targets)

            loss = loss5 +loss3 + loss2 + loss1

            # backward
            self.manual_backward(loss)

            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            
            # recording loss
            if rate == 1:
                pred_map_r1 = torch.sigmoid(pred_map_5)
                target_r1 = targets
                loss_r1 = loss

        # updadte and log metrics
        self.train_loss(loss_r1)
        
        pred_map_r1 = (pred_map_r1 > 0.5).to(torch.int64)
        target_r1_int = (target_r1 > 0.5).to(torch.int64)

        self.train_dice(pred_map_r1, target_r1_int)
        self.train_iou(pred_map_r1, target_r1_int)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss_r1, "preds": pred_map_r1, "targets": target_r1}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_dice(preds, targets)
        self.val_iou(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_dice.compute()  # get current val acc
        self.val_dice_best(acc)  # update best so far val acc
        # log `val_dice_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/dice_best", self.val_dice_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_dice(preds, targets)
        self.test_iou(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":

    import hydra
    from omegaconf import DictConfig

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "segmentation")

    @hydra.main(version_base=None, config_path=config_path, config_name="caranet_module.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        caranet_model = hydra.utils.instantiate(cfg)
        x = torch.randn(1, 3, 128, 128)

        logits, _, _, _ = caranet_model(x)
        print(logits.shape)

    main()
        