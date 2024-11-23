from typing import Any, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import rootutils

from lightning import LightningModule
from torchmetrics import MeanMetric, Dice, JaccardIndex
from torch.optim import Optimizer, lr_scheduler
from contextlib import contextmanager

from segmentation_models_pytorch.losses import (SoftCrossEntropyLoss,
                                                SoftBCEWithLogitsLoss, 
                                                DiceLoss, 
                                                FocalLoss, 
                                                JaccardLoss)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.loss import StructureLoss
from src.models.segmentation.net import (UNet,
                                         UNetAttention,
                                         UNetPlusPlus)
from src.utils.ema import LitEma


class SegmentationModule(LightningModule):

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        criterion: nn.Module,
        metrics: List[str],
        use_ema: bool = False,
        compile: bool = False) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        assert isinstance(net, (UNet, UNetAttention, UNetPlusPlus)), \
            NotImplementedError(f"Only implemented for [UNet, UNetAttention, UNetPlusPlus]")
        
        # UNet model
        self.net = net

        assert isinstance(criterion, (SoftCrossEntropyLoss, 
                                      SoftBCEWithLogitsLoss, 
                                      DiceLoss, 
                                      StructureLoss,
                                      FocalLoss,
                                      JaccardLoss)), \
            NotImplementedError("Only implemented for [CrossEntropyLoss, SoftBCEWithLogitsLoss, \
                                DiceLoss, FocalLoss, JaccardLoss, StructureLoss]")
        
        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        for metric in metrics:
            if metric == "dice":
                self.train_dice = Dice(ignore_index=0)
                self.val_dice = Dice(ignore_index=0)
                self.test_dice = Dice(ignore_index=0)
        
            elif metric =="iou":
                self.train_iou = JaccardIndex(task="binary")
                self.val_iou = JaccardIndex(task="binary")
                self.test_iou = JaccardIndex(task="binary")

            else:
                NotImplementedError(f"Not implemented for {metric} metric")

        self.metrics = metrics

        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

    def on_train_batch_end(self, *args, **kwargs) -> None:
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
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: Tensor, threshold: float = 0.5, dtype: torch.dtype = torch.float32) -> Tensor:
        # return mask

        if self.use_ema:
            with self.ema_scope():
                logits = self.net(x)
        else:
            logits = self.net(x)

        if isinstance(self.criterion, SoftCrossEntropyLoss) or \
            (isinstance(self.criterion, (DiceLoss, FocalLoss, JaccardLoss)) and self.criterion.mode == "multiclass"):
            # multi-class
            preds = nn.functional.softmax(logits, dim=1)
        else:
            # binary or multi-label
            preds = nn.functional.sigmoid(logits)

        return (preds > threshold).to(dtype)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        if "dice" in self.metrics:
            self.val_dice.reset()
        
        if "iou" in self.metrics:
            self.val_iou.reset()

    def model_step(
            self, batch: Tuple[Tensor,
                               Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        images, targets = batch

        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        preds = nn.functional.sigmoid(logits)

        preds = (preds > 0.5).to(torch.int64)
        targets = (targets > 0.5).to(torch.int64)

        return loss, preds, targets

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        if "dice" in self.metrics:
            self.train_dice(preds, targets)
            self.log("train/dice",
                    self.train_dice,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True)

        if "iou" in self.metrics:
            self.train_iou(preds, targets)
            self.log("train/iou",
                     self.train_iou,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     sync_dist=True)

        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        if "dice" in self.metrics:
            self.val_dice(preds, targets)
            self.log("val/dice",
                     self.val_dice,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     sync_dist=True)

        if "iou" in self.metrics:
            self.val_iou(preds, targets)
            self.log("val/iou",
                     self.val_iou,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)

        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        if "dice" in self.metrics:
            self.test_dice(preds, targets)
            self.log("test/dice",
                     self.test_dice,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     sync_dist=True)

        if "iou" in self.metrics:
            self.test_iou(preds, targets)
            self.log("test/iou",
                     self.test_iou,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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

    root = rootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "segmentation")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet_module.yaml")
    def main1(cfg: DictConfig):
        print(cfg)

        unet_module: SegmentationModule = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        out = unet_module(x)
        print('*' * 20, ' UNET MODULE ', '*' * 20)
        print('Input:', x.shape)
        print('Output:', out.shape)
        print('-' * 100)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet_plus_plus_module.yaml")
    def main2(cfg: DictConfig):
        print(cfg)

        unet_plus_plus_module: SegmentationModule = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        out = unet_plus_plus_module(x)
        print('*' * 20, ' UNET PLUS PLUS MODULE ', '*' * 20)
        print('Input:', x.shape)
        print('Output:', out.shape)
        print('-' * 100)
    
    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet_attention_module.yaml")
    def main3(cfg: DictConfig):
        print(cfg)

        unet_module: SegmentationModule = hydra.utils.instantiate(cfg)
        image = torch.randn(2, 1, 32, 32)
        mask = torch.ones((2, 1, 32, 32))

        logits = unet_module(image)
        loss, _, _ = unet_module.model_step(batch=(image, mask))

        print('*' * 20, ' UNet Attention Module ', '*' * 20)
        print('Input:', image.shape)
        print('Output:', logits.shape)
        print(f"{unet_module.criterion._get_name()}:", loss)

    main1()
    main2()
    main3()