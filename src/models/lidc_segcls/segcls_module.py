from typing import Any, List

import torch
from torch import Tensor

import rootutils
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torch.optim import Optimizer, lr_scheduler
from torchmetrics.classification import Dice
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.specificity import Specificity
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from contextlib import contextmanager

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma

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

class LIDCSegClsModule(LightningModule):

    def __init__(
        self, 
        net: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        groups: List[str],
        n_classes: List[int],
        weights_classes: List[List[float]],
        image_size:int = 352,
        size_rates: List[float] = [0.75, 1, 1.25],
        use_ema: bool = False,
    ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # multi-scale training
        self.groups = groups
        self.n_classes = n_classes
        self.weights_classes = weights_classes
        self.size_rates = size_rates
        self.image_size = image_size
        self.cls_loss = CrossEntropyLoss()
        
        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

        # manual optimization for multi-scale training
        self.automatic_optimization = False

        ############### Segmentation ###############
        # loss
        self.train_seg_loss = MeanMetric()
        self.val_seg_loss = MeanMetric()
        self.test_seg_loss = MeanMetric()

        # metric
        self.train_dice, self.val_dice, self.test_dice  = Dice(ignore_index=0), Dice(ignore_index=0), Dice(ignore_index=0)
        self.train_iou, self.val_iou, self.test_iou = BinaryJaccardIndex(), BinaryJaccardIndex(), BinaryJaccardIndex()
        self.val_dice_best = MaxMetric()

        ############### Classification ###############
        # loss
        self.train_cls_loss = MeanMetric()
        self.val_cls_loss = MeanMetric()
        self.test_cls_loss = MeanMetric()

        # metric
        self.train_acc, self.val_acc, self.test_acc = {}, {}, {}
        self.train_precision, self.val_precision, self.test_precision = {}, {}, {}
        self.train_recall_sens, self.val_recall_sens, self.test_recall_sens = {}, {}, {}
        self.train_spec, self.val_spec, self.test_spec = {}, {}, {}

        for group, n_class in zip(self.groups, self.n_classes):
            self.train_acc[group] = Accuracy(task="multiclass", num_classes=n_class)
            self.val_acc[group] = Accuracy(task="multiclass", num_classes=n_class)
            self.test_acc[group] = Accuracy(task="multiclass", num_classes=n_class)

            self.train_precision[group] = Precision(task="multiclass", num_classes=n_class)
            self.val_precision[group] = Precision(task="multiclass", num_classes=n_class)
            self.test_precision[group] = Precision(task="multiclass", num_classes=n_class)

            self.train_recall_sens[group] = Recall(task="multiclass", num_classes=n_class)
            self.val_recall_sens[group] = Recall(task="multiclass", num_classes=n_class)
            self.test_recall_sens[group] = Recall(task="multiclass", num_classes=n_class)

            self.train_spec[group] = Specificity(task="multiclass", num_classes=n_class)
            self.val_spec[group] = Specificity(task="multiclass", num_classes=n_class)
            self.test_spec[group] = Specificity(task="multiclass", num_classes=n_class)

    def on_train_batch_end(self, *args, **kwargs):
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

    def forward(self, images: Tensor, nodules_infos=None) -> Tensor:
        return self.net(images, nodules_infos)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store accuracy from these checks
        self.val_seg_loss.reset()
        self.val_cls_loss.reset()
        self.val_dice.reset()
        self.val_iou.reset()
        self.val_dice_best.reset()

        for group in self.groups:
            self.val_acc[group]
            self.val_precision[group].reset()
            self.val_recall_sens[group].reset()
            self.val_spec[group].reset()

    def get_seg_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        def structure_loss(pred, mask):

            weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
            wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
            wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

            pred = torch.sigmoid(pred)
            inter = ((pred * mask)*weit).sum(dim=(2, 3))
            union = ((pred + mask)*weit).sum(dim=(2, 3))
            wiou = 1 - (inter + 1)/(union - inter+1)
            
            return (wbce + wiou).mean()

        pred_map_5, pred_map_3, pred_map_2, pred_map_1 = logits

        loss5 = structure_loss(pred_map_5, targets)
        loss3 = structure_loss(pred_map_3, targets)
        loss2 = structure_loss(pred_map_2, targets)
        loss1 = structure_loss(pred_map_1, targets)
        loss = loss5 + loss3 + loss2 + loss1

        preds = (F.sigmoid(pred_map_5) > 0.5).to(torch.int64)
        targets = (targets > 0.5).to(torch.int64)

        return loss, preds, targets

    def get_cls_loss(self, logits: torch.Tensor, targets):
        # return {"loss": 0}, None, None
        if targets is None: 
            return {"loss": 0}, None, None

        losses = {}
        preds = {}
        for i in range(len(self.groups)):
            group = self.groups[i]
            weight_classes = torch.tensor(self.weights_classes[i], device=self.device) \
                            if self.weights_classes[i] is not None else None
            # print("targets:", targets[group])
            losses[group] = self.cls_loss(logits[i], targets[group])
            prob = F.softmax(logits[i], dim=1) 
            preds[group] = torch.argmax(prob, dim=1)
            targets[group] = torch.argmax(targets[group], dim=1)
            if losses[group].isnan().any():
                print("losses:", losses)
                print("logits:", logits)
                print("targets:", targets)
                raise ValueError("Loss is NaN")

        return losses, preds, targets


    def predict(self, images: Tensor, threshold: float = 0.5, dtype: torch.dtype = torch.float32):
        (logits, _, _, _), _ = self.net(images, None)
        preds = (F.sigmoid(logits) > threshold).to(dtype)
        return preds

    def model_step(self, batch: Any):
        seg_inputs, seg_targets, nodules_infos = batch
        seg_preds, (cls_preds, cls_targets) = self.forward(seg_inputs, nodules_infos)
        return self.get_seg_loss(seg_preds, seg_targets), self.get_cls_loss(cls_preds, cls_targets)

    def training_step(self, batch: Any, batch_idx: int):
        seg_inputs, seg_targets, nodules_infos = batch
        torch.autograd.set_detect_anomaly(True)    

        # multi-scale training
        for rate in self.size_rates:
            # manual optimization
            opt = self.optimizers()
            opt.zero_grad()

            # rescale
            train_size = int(round(self.image_size*rate/32)*32)

            nodules = nodules_infos.copy()
            if rate != 1:
                images = F.interpolate(seg_inputs, size=(train_size, train_size), mode='bilinear', align_corners=True)
                masks = F.interpolate(seg_targets, size=(train_size, train_size), mode='bilinear', align_corners=True)
                nodules["bbox"] = (nodules["bbox"] * rate).to(torch.int64)
            else:
                images = seg_inputs
                masks = seg_targets

            seg_info, cls_info = self.model_step(batch=(images, masks, nodules))
            seg_loss, _, _ = seg_info
            cls_loss, _, _ = cls_info

            # backward

            self.manual_backward(seg_loss + sum(cls_loss.values()))

            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            
            # recording loss
            if rate == 1:
                seg_info_r1 = seg_info
                cls_info_r1 = cls_info

        seg_loss, seg_preds, seg_targets = seg_info_r1
        cls_loss, cls_preds, cls_targets = cls_info_r1

        ############### Segmentation ###############
        self.train_seg_loss(seg_loss)
        self.train_dice(seg_preds, seg_targets)
        self.train_iou(seg_preds, seg_targets)

        self.log("train/seg_loss", self.train_seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        if cls_preds is None: return

        ############### Classification ###############
        self.train_cls_loss(sum(cls_loss.values()))
        self.log("train/cls_loss", self.train_cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        for group in self.groups:
            # self.log(f"train/cls_loss/{group}",cls_loss_r1[group].detach(), on_step=False, on_epoch=True)

            self.train_acc[group].to(self.device)
            self.train_acc[group](cls_preds[group], cls_targets[group])
            self.log(f"train/acc/{group}", self.train_acc[group], on_step=False, on_epoch=True, metric_attribute=f"train_acc_{group}")

            self.train_precision[group].to(self.device)
            self.train_precision[group](cls_preds[group], cls_targets[group])
            self.log(f"train/precision/{group}", self.train_precision[group], on_step=False, on_epoch=True, metric_attribute=f"train_precision_{group}")

            self.train_recall_sens[group].to(self.device)
            self.train_recall_sens[group](cls_preds[group], cls_targets[group])
            self.log(f"train/recall_sens/{group}", self.train_recall_sens[group], on_step=False, on_epoch=True, metric_attribute=f"train_recall_sens_{group}")

            self.train_spec[group].to(self.device)
            self.train_spec[group](cls_preds[group], cls_targets[group])
            self.log(f"train/spec/{group}", self.train_spec[group], on_step=False, on_epoch=True, metric_attribute=f"train_spec_{group}")

    def validation_step(self, batch: Any, batch_idx: int):
        seg_info, cls_info = self.model_step(batch)

        seg_loss, seg_preds,seg_targets = seg_info
        cls_loss, cls_preds, cls_targets = cls_info

        ############### Segmentation ###############
        self.val_seg_loss(seg_loss)
        self.val_dice(seg_preds, seg_targets)
        self.val_iou(seg_preds, seg_targets)

        self.log("val/seg_loss", self.val_seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        if cls_preds is None: return

        ############### Classification ###############
        self.val_cls_loss(sum(cls_loss.values()))
        self.log("val/cls_loss", self.val_cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        for group in self.groups:
            # self.log(f"val/cls_loss/{group}",cls_loss[group].detach(), on_step=False, on_epoch=True)

            self.val_acc[group].to(self.device)
            self.val_acc[group](cls_preds[group], cls_targets[group])
            self.log(f"val/acc/{group}", self.val_acc[group], on_step=False, on_epoch=True, metric_attribute=f"val_acc_{group}")

            self.val_precision[group].to(self.device)
            self.val_precision[group](cls_preds[group], cls_targets[group])
            self.log(f"val/precision/{group}", self.val_precision[group], on_step=False, on_epoch=True, metric_attribute=f"val_precision_{group}")

            self.val_recall_sens[group].to(self.device)
            self.val_recall_sens[group](cls_preds[group], cls_targets[group])
            self.log(f"val/recall_sens/{group}", self.val_recall_sens[group], on_step=False, on_epoch=True, metric_attribute=f"val_recall_sens_{group}")

            self.val_spec[group].to(self.device)
            self.val_spec[group](cls_preds[group], cls_targets[group])
            self.log(f"val/spec/{group}", self.val_spec[group], on_step=False, on_epoch=True, metric_attribute=f"val_spec_{group}")

    def on_validation_epoch_end(self):
        acc = self.val_dice.compute()  # get current val acc
        self.val_dice_best(acc)  # update best so far val acc
        # log `val_dice_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch        
        self.log("val/dice_best", self.val_dice_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        seg_info, cls_info = self.model_step(batch)

        seg_loss, seg_preds,seg_targets = seg_info
        cls_loss, cls_preds, cls_targets = cls_info

        ############### Segmentation ###############
        self.test_seg_loss(seg_loss)
        self.test_dice(seg_preds, seg_targets)
        self.test_iou(seg_preds, seg_targets)

        self.log("test/seg_loss", self.test_seg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        if cls_preds is None: return

        ############### Classification ###############
        self.test_cls_loss(sum(cls_loss.values()))
        self.log("test/cls_loss", self.test_cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        for group in self.groups:
            # self.log(f"test/cls_loss/{group}",cls_loss[group].detach(), on_step=False, on_epoch=True)

            self.test_acc[group].to(self.device)
            self.test_acc[group](cls_preds[group], cls_targets[group])
            self.log(f"test/acc/{group}", self.test_acc[group], on_step=False, on_epoch=True, metric_attribute=f"test_acc_{group}")

            self.test_precision[group].to(self.device)
            self.test_precision[group](cls_preds[group], cls_targets[group])
            self.log(f"test/precision/{group}", self.test_precision[group], on_step=False, on_epoch=True, metric_attribute=f"test_precision_{group}")

            self.test_recall_sens[group].to(self.device)
            self.test_recall_sens[group](cls_preds[group], cls_targets[group])
            self.log(f"test/recall_sens/{group}", self.test_recall_sens[group], on_step=False, on_epoch=True, metric_attribute=f"test_recall_sens_{group}")

            self.test_spec[group].to(self.device)
            self.test_spec[group](cls_preds[group], cls_targets[group])
            self.log(f"test/spec/{group}", self.test_spec[group], on_step=False, on_epoch=True, metric_attribute=f"test_spec_{group}")

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
    config_path = str(root / "configs" / "model" / "lidc_segcls")

    @hydra.main(version_base=None, config_path=config_path, config_name="lidc_segcls.yaml")
    def main(cfg: DictConfig):
        cfg["image_size"] = 352
        cfg["net"]["cls_net"]["n_classes"] = [4, 7]
        cfg["net"]["matching_threshold"] = 0.
        print(cfg)

        segcls_module: LIDCSegClsModule = hydra.utils.instantiate(cfg)
        
        images = torch.randn(1, 1, 352, 352)
        masks = torch.ones((1, 1, 352, 352))
        nodules_infos = {
            "bbox": torch.tensor([[[10, 10, 350, 30], 
                                [1, 1, 350, 350],
                                [1, 1, 350, 350]]]),
            "label":{
                "cancer": torch.tensor([[[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]]]),
                "malignancy": torch.tensor([[[1, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1]]]),

            }
        }

        seg_info, cls_info = segcls_module.model_step(batch=(images, masks, nodules_infos))
        seg_loss, seg_preds,seg_targets = seg_info
        cls_loss, cls_preds, cls_targets = cls_info

        print("Segmentation Loss:", seg_loss)
        print("Segmentation Pred: ", seg_preds.shape, seg_preds.dtype)

        print("Classification Loss: ", cls_loss)
        for group in cls_preds:
            print(f"Classification Pred {group}: ", cls_preds[group].shape, cls_preds[group].dtype)
    main()
