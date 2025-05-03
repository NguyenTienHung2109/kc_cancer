from typing import Any, List, Dict

import copy
import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule

from torchmetrics import MaxMetric, MeanMetric
from torch.optim import Optimizer, lr_scheduler
from torchmetrics.classification import Dice, BinaryJaccardIndex, JaccardIndex
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.specificity import Specificity
import torch.nn.functional as F
from contextlib import contextmanager

import rootutils
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

class SegClsModule(LightningModule):

    def __init__(
        self, 
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        size_rates: List[float],
        use_ema: bool,
        seg_nodule_loss_func: nn.Module,
        use_nodule_cls: bool = False,
        nodule_cls_groups: List[str] | None = None,
        nodule_cls_classes: List[int] | None = None,
        nodule_cls_weights: List[List[float]] | None = None,
        use_lung_pos: bool = False,
        lung_pos_groups: List[str] | None = None,
        lung_pos_weights: List[List[float]] | None = None,
        use_lung_loc: bool = False,
        seg_lung_loc_loss_func: nn.Module = None,
    ) -> None:
        
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        ### model
        self.net = net
        self.use_nodule_cls = use_nodule_cls # classification nodule
        self.use_lung_pos = use_lung_pos # classification lung position
        self.use_lung_loc = use_lung_loc # segmentation lung location
        self.use_ema = use_ema # ema
        self.size_rates = size_rates # multi-scale training
        self.automatic_optimization = False # manual optimization for multi-scale training

        ################## Group loss and metric ###############
        ### loss
        self.train_loss, self.val_loss, self.test_loss = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        
        ### segmentation metric
        self.train_dice, self.val_dice, self.test_dice = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        self.train_iou, self.val_iou, self.test_iou = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        
        ### classification metric
        self.train_acc, self.val_acc, self.test_acc = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        self.train_precision, self.val_precision, self.test_precision = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        self.train_recall_sens, self.val_recall_sens, self.test_recall_sens = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        self.train_spec, self.val_spec, self.test_spec = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        ############### Segmentation Nodule ###############
        ### loss
        self.seg_nodule_loss_func = seg_nodule_loss_func
        self.train_loss["seg_nodule"] = MeanMetric()
        self.val_loss["seg_nodule"] = MeanMetric()
        self.test_loss["seg_nodule"] = MeanMetric()

        ### metric
        self.train_dice["seg_nodule"] = Dice(ignore_index=0)
        self.val_dice["seg_nodule"] = Dice(ignore_index=0)
        self.test_dice["seg_nodule"] = Dice(ignore_index=0)
        
        self.train_iou["seg_nodule"] = BinaryJaccardIndex()
        self.val_iou["seg_nodule"] = BinaryJaccardIndex()
        self.test_iou["seg_nodule"] = BinaryJaccardIndex()
        
        self.val_dice_best = MaxMetric() # save weight best with segmentation nodule

        ############### Classification Nodule ###############
        if use_nodule_cls:
            self.nodule_cls_groups = nodule_cls_groups
            self.nodule_cls_classes = nodule_cls_classes
            self.nodule_cls_weights = nodule_cls_weights
            
            ### loss
            self.train_loss["cls_nodule"] = MeanMetric()
            self.val_loss["cls_nodule"] = MeanMetric()
            self.test_loss["cls_nodule"] = MeanMetric()
            
            ### metric
            for group, n_class in zip(nodule_cls_groups, nodule_cls_classes):
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


        ############### Classification Lung Position ###############
        if use_lung_pos:
            self.lung_pos_groups = lung_pos_groups
            self.lung_pos_weights = lung_pos_weights
            
            ### loss
            self.train_loss["cls_lung_pos"] = MeanMetric()
            self.val_loss["cls_lung_pos"] = MeanMetric()
            self.test_loss["cls_lung_pos"] = MeanMetric()
            
            ### metric
            for position in self.lung_pos_groups:
                self.train_acc[position] = Accuracy(task="multiclass", num_classes=2)
                self.val_acc[position] = Accuracy(task="multiclass", num_classes=2)
                self.test_acc[position] = Accuracy(task="multiclass", num_classes=2)

        ############### Segmentation Lung Location ###############
        if use_lung_loc:
            ### loss
            self.seg_lung_loc_loss_func = seg_lung_loc_loss_func
            self.train_loss["seg_lung_loc"] = MeanMetric()
            self.val_loss["seg_lung_loc"] = MeanMetric()
            self.test_loss["seg_lung_loc"] = MeanMetric()
            
            ### metric
            self.train_dice["seg_lung_loc"] = Dice(ignore_index=0, num_classes=6)
            self.val_dice["seg_lung_loc"] = Dice(ignore_index=0, num_classes=6)
            self.test_dice["seg_lung_loc"] = Dice(ignore_index=0, num_classes=6)
            
            self.train_iou["seg_lung_loc"] = JaccardIndex(num_classes=6, task="multiclass")
            self.val_iou["seg_lung_loc"] = JaccardIndex(num_classes=6, task="multiclass")
            self.test_iou["seg_lung_loc"] = JaccardIndex(num_classes=6, task="multiclass")

        ### exponential moving average
        if use_ema:
            self.model_ema = LitEma(self.net)

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

    def forward(self, images: Tensor, nodules_infos=None) -> Tensor:
        return self.net(images, nodules_infos)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store accuracy from these checks
        
        ############### Segmentation Nodule ###############
        self.val_loss["seg_nodule"].reset()
        self.val_dice["seg_nodule"].reset()
        self.val_iou["seg_nodule"].reset()
        self.val_dice_best.reset()

        ############### Classification Nodule ###############
        if self.use_nodule_cls:
            self.val_loss["cls_nodule"].reset()
            for group in self.nodule_cls_groups:
                self.val_acc[group]

        ############### Classification Lung Position ###############
        if self.use_lung_pos:
            self.val_loss["cls_lung_pos"].reset()
            for position in self.lung_pos_groups:
                self.val_acc[position]

        ############### Segmentation Lung Location ###############
        if self.use_lung_loc:
            self.val_loss["seg_lung_loc"].reset()
            self.val_dice["seg_lung_loc"].reset()
            self.val_iou["seg_lung_loc"].reset()

    def get_seg_nodule_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        pred_map_5, pred_map_3, pred_map_2, pred_map_1 = logits
        
        loss5 = self.seg_nodule_loss_func(pred_map_5, targets)
        loss3 = self.seg_nodule_loss_func(pred_map_3, targets)
        loss2 = self.seg_nodule_loss_func(pred_map_2, targets)
        loss1 = self.seg_nodule_loss_func(pred_map_1, targets)
        loss = loss5 + loss3 + loss2 + loss1
        
        preds = (F.sigmoid(pred_map_5) > 0.5).to(torch.int64)
        targets = (targets > 0.5).to(torch.int64)
        
        return loss, preds, targets

    def get_cls_nodule_loss(self, logits: List[Tensor], targets: Dict[str, Tensor]):
        if len(logits[0]) == 0: # clean
            return {"loss": 0}, None, None
        
        losses, preds = {}, {}
        for i in range(len(self.nodule_cls_groups)):
            group = self.nodule_cls_groups[i]
            weight = torch.tensor(self.nodule_cls_weights[i], device=self.device) \
                    if self.nodule_cls_weights is not None else None
            
            losses[group] = F.cross_entropy(logits[i], targets[group], weight=weight)
            prob = F.softmax(logits[i], dim=1) 
            preds[group] = torch.argmax(prob, dim=1)
            targets[group] = torch.argmax(targets[group], dim=1)
        
        return losses, preds, targets

    def get_cls_lung_pos_loss(self, logits: torch.Tensor, targets):
        losses, preds = {}, {}
        for i in range(len(self.lung_pos_groups)):
            position = self.lung_pos_groups[i]
            weight = torch.tensor(self.lung_pos_weights[i], device=self.device) \
                    if self.lung_pos_weights is not None else None
            losses[position] = F.cross_entropy(logits[i], targets[position], weight=weight)
            prob = F.softmax(logits[i], dim=1) 
            preds[position] = torch.argmax(prob, dim=1)
            targets[position] = torch.argmax(targets[position], dim=1)
        
        return losses, preds, targets

    def get_seg_lung_loc_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        pred_map_5, pred_map_3, pred_map_2, pred_map_1 = logits
        
        loss5 = self.seg_lung_loc_loss_func(pred_map_5, targets)
        loss3 = self.seg_lung_loc_loss_func(pred_map_3, targets)
        loss2 = self.seg_lung_loc_loss_func(pred_map_2, targets)
        loss1 = self.seg_lung_loc_loss_func(pred_map_1, targets)
        loss = loss5 + loss3 + loss2 + loss1
        
        probs = F.softmax(pred_map_5, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(targets, dim=1)
        
        return loss, preds, targets

    def predict(self, images: Tensor, dtype: torch.dtype = torch.float32):
        out = self.net(images, nodules_infos=None, training=False)
        res = {
            "seg_nodule": torch.from_numpy(out["seg_nodule"]).to(dtype),
        }
        if self.use_lung_loc:
            res["seg_lung_loc"] = torch.from_numpy(out["seg_lung_loc"]).to(dtype)
        return res

    def model_step(self, batch: Any):
        nodules_infos = None
        if self.use_nodule_cls:
            nodules_infos = batch["nodule_info"]
        out = self.forward(batch["slice"], nodules_infos)
        
        # only segment for inside slice
        if self.use_lung_pos:
            outside = (batch["cls_lung_pos"]["right_lung"][:, 0] == 1) & \
                        (batch["cls_lung_pos"]["left_lung"][:, 0] == 1)
            ids_outside = torch.nonzero(outside, as_tuple=False).squeeze()
            for i in range(len(out["seg_nodule"])):
                out["seg_nodule"][i][ids_outside, ...] = 0
        
        # only back propagation for no clean lung_loc_mask -> check one hot for background
        if self.use_lung_loc:
            ids = torch.where(torch.all(batch["seg_lung_loc"][:, 0] == 1, dim=(1, 2)))[0]  
            for i in range(len(out["seg_lung_loc"])):
                out["seg_lung_loc"][i][ids, ...] = 0
        
        res = {
                "seg_nodule": self.get_seg_nodule_loss(out["seg_nodule"], batch["seg_nodule"]), 
            }
        if self.use_nodule_cls:
            res["cls_nodule"] = self.get_cls_nodule_loss(out["cls_nodule"][0], out["cls_nodule"][1])
        if self.use_lung_pos:
            res["cls_lung_pos"] = self.get_cls_lung_pos_loss(out["cls_lung_pos"], batch["cls_lung_pos"])
        if self.use_lung_loc:
            res["seg_lung_loc"] = self.get_seg_lung_loc_loss(out["seg_lung_loc"], batch["seg_lung_loc"])
        
        return res

    def training_step(self, batch: Any, batch_idx: int):
        
        # multi-scale training
        for rate in self.size_rates:
            # manual optimization
            opt = self.optimizers()
            opt.zero_grad()

            train_size = int(round(batch["slice"].shape[-1]*rate/32)*32) # rescale

            slices = batch["slice"]
            seg_nodule_targets = batch["seg_nodule"]
            if self.use_nodule_cls:
                nodules_infos = batch["nodule_info"].copy()
            if self.use_lung_loc:
                seg_lung_loc_targets = batch["seg_lung_loc"]
            if self.use_lung_pos:
                cls_lung_pos_targets = batch["cls_lung_pos"].copy()
            if rate != 1:
                slices = F.interpolate(batch["slice"], size=(train_size, train_size), mode='bilinear', align_corners=True)
                seg_nodule_targets = F.interpolate(batch["seg_nodule"], size=(train_size, train_size), mode='bilinear', align_corners=True)
                if self.use_nodule_cls:
                    nodules_infos["bbox"] = (nodules_infos["bbox"] * rate).to(torch.int64)
                if self.use_lung_loc:
                    seg_lung_loc_targets = F.interpolate(batch["seg_lung_loc"], size=(train_size, train_size), mode='bilinear', align_corners=True)

            b = {
                "slice": slices,
                "slice_info": batch["slice_info"],
                "seg_nodule": seg_nodule_targets,
            }
            if self.use_nodule_cls:
                b["nodule_info"] = nodules_infos
            if self.use_lung_pos:
                b["cls_lung_pos"] = cls_lung_pos_targets
            if self.use_lung_loc:
                b["seg_lung_loc"] = seg_lung_loc_targets

            out = self.model_step(batch=b)
            seg_nodule_loss, _, _ = out["seg_nodule"]
            total_loss = seg_nodule_loss
            if self.use_nodule_cls:
                cls_nodule_loss, _, _ = out["cls_nodule"]
                total_loss += sum(cls_nodule_loss.values())
            if self.use_lung_pos:
                cls_lung_pos_loss, _, _ = out["cls_lung_pos"]
                total_loss += sum(cls_lung_pos_loss.values())
            if self.use_lung_loc:
                seg_lung_loc_loss, _, _ = out["seg_lung_loc"]
                total_loss += seg_lung_loc_loss

            # backward
            self.manual_backward(total_loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            
            # recording loss
            if rate == 1:
                out_r1 = out

        ############### Segmentation Nodule ###############
        seg_nodule_loss, seg_nodule_preds, seg_nodule_targets = out_r1["seg_nodule"]
        self.train_loss["seg_nodule"](seg_nodule_loss)

        # if self.use_lung_loc:
        #     _, seg_lung_loc_preds, _ = out_r1["seg_lung_loc"]
        #     seg_nodule_preds *= seg_lung_loc_preds

        self.train_dice["seg_nodule"](seg_nodule_preds, seg_nodule_targets)
        self.train_iou["seg_nodule"](seg_nodule_preds, seg_nodule_targets)

        self.log("train/loss/seg_nodule", self.train_loss["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_loss_seg_nodule")
        self.log("train/dice/seg_nodule", self.train_dice["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_dice_seg_nodule")
        self.log("train/iou/seg_nodule", self.train_iou["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_iou_seg_nodule")

        ############### Classification Nodule ###############
        if self.use_nodule_cls:
            cls_nodule_loss, cls_nodule_preds, cls_nodule_targets = out_r1["cls_nodule"]
            if cls_nodule_preds is not None:
                self.train_loss["cls_nodule"](sum(cls_nodule_loss.values()))
                self.log("train/loss/cls_nodule", self.train_loss["cls_nodule"], on_step=False, on_epoch=True, prog_bar=True,  metric_attribute=f"train_loss_cls_nodule")
                for group in self.nodule_cls_groups:
                    self.train_acc[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"train/acc/{group}", self.train_acc[group], on_step=False, on_epoch=True, metric_attribute=f"train_acc_{group}")
        
        ############### Classification Lung Position ###############
        if self.use_lung_pos:
            cls_lung_pos_loss, lung_pos_preds, lung_pos_labels = out_r1["cls_lung_pos"]
            self.train_loss["cls_lung_pos"](sum(cls_lung_pos_loss.values()))
            self.log("train/loss/cls_lung_pos", self.train_loss["cls_lung_pos"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_loss_cls_lung_pos")
            for position in self.lung_pos_groups:
                self.train_acc[position](lung_pos_preds[position], lung_pos_labels[position])
                self.log(f"train/acc/{position}", self.train_acc[position], on_step=False, on_epoch=True, metric_attribute=f"train_acc_{position}")

        ############### Segmentation Lung Location ###############
        if self.use_lung_loc:
            seg_lung_loc_loss, seg_lung_loc_preds, seg_lung_loc_targets = out_r1["seg_lung_loc"]

            self.train_loss["seg_lung_loc"](seg_lung_loc_loss)
            self.train_dice["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)
            self.train_iou["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)

            self.log("train/loss/seg_lung_loc", self.train_loss["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_loss_seg_lung_loc")
            self.log("train/dice/seg_lung_loc", self.train_dice["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_dice_seg_lung_loc")
            self.log("train/iou/seg_lung_loc", self.train_iou["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"train_iou_seg_lung_loc")

    def validation_step(self, batch: Any, batch_idx: int):
        out = self.model_step(batch)

        ############### Segmentation Nodule ###############
        seg_nodule_loss, seg_nodule_preds, seg_nodule_targets = out["seg_nodule"]
        self.val_loss["seg_nodule"](seg_nodule_loss)

        # if self.use_lung_loc:
        #     _, seg_lung_loc_preds, _ = out["seg_lung_loc"]
        #     seg_nodule_preds *= seg_lung_loc_preds

        self.val_dice["seg_nodule"](seg_nodule_preds, seg_nodule_targets)
        self.val_iou["seg_nodule"](seg_nodule_preds, seg_nodule_targets)

        self.log("val/loss/seg_nodule", self.val_loss["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_loss_seg_nodule")
        self.log("val/dice/seg_nodule", self.val_dice["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_dice_seg_nodule")
        self.log("val/iou/seg_nodule", self.val_iou["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_iou_seg_nodule")

        ############### Classification Nodule ###############
        if self.use_nodule_cls:
            cls_nodule_loss, cls_nodule_preds, cls_nodule_targets = out["cls_nodule"]
            if cls_nodule_preds is not None:
                self.val_loss["cls_nodule"](sum(cls_nodule_loss.values()))
                self.log("val/loss/cls_nodule", self.val_loss["cls_nodule"], on_step=False, on_epoch=True, prog_bar=True,  metric_attribute=f"val_loss_cls_nodule")
                for group in self.nodule_cls_groups:
                    self.val_acc[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"val/acc/{group}", self.val_acc[group], on_step=False, on_epoch=True, metric_attribute=f"val_acc_{group}")

        ############### Classification Lung Position ###############
        if self.use_lung_pos:
            cls_lung_pos_loss, lung_pos_preds, lung_pos_labels = out["cls_lung_pos"]
            self.val_loss["cls_lung_pos"](sum(cls_lung_pos_loss.values()))
            self.log("val/loss/cls_lung_pos", self.val_loss["cls_lung_pos"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_loss_cls_lung_pos")
            for position in self.lung_pos_groups:
                self.val_acc[position](lung_pos_preds[position], lung_pos_labels[position])
                self.log(f"val/acc/{position}", self.val_acc[position], on_step=False, on_epoch=True, metric_attribute=f"val_acc_{position}")

        ############### Segmentation Lung Location ###############
        if self.use_lung_loc:
            seg_lung_loc_loss, seg_lung_loc_preds, seg_lung_loc_targets = out["seg_lung_loc"]

            self.val_loss["seg_lung_loc"](seg_lung_loc_loss)
            self.val_dice["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)
            self.val_iou["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)

            self.log("val/loss/seg_lung_loc", self.val_loss["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_loss_seg_lung_loc")
            self.log("val/dice/seg_lung_loc", self.val_dice["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_dice_seg_lung_loc")
            self.log("val/iou/seg_lung_loc", self.val_iou["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"val_iou_seg_lung_loc")

    def on_validation_epoch_end(self):
        dice = self.val_dice["seg_nodule"].compute()  # get current val acc
        self.val_dice_best(dice)  # update best so far val acc
        # log `val_dice_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch        
        self.log("val/dice_best", self.val_dice_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        out = self.model_step(batch)

        ############### Segmentation Nodule ###############
        seg_nodule_loss, seg_nodule_preds, seg_nodule_targets = out["seg_nodule"]
        self.test_loss["seg_nodule"](seg_nodule_loss)

        # if self.use_lung_loc:
        #     _, seg_lung_loc_preds, _ = out["seg_lung_loc"]
        #     seg_nodule_preds *= seg_lung_loc_preds

        self.test_dice["seg_nodule"](seg_nodule_preds, seg_nodule_targets)
        self.test_iou["seg_nodule"](seg_nodule_preds, seg_nodule_targets)

        self.log("test/loss/seg_nodule", self.test_loss["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_loss_seg_nodule")
        self.log("test/dice/seg_nodule", self.test_dice["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_dice_seg_nodule")
        self.log("test/iou/seg_nodule", self.test_iou["seg_nodule"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_iou_seg_nodule")

        ############### Classification Nodule ###############
        if self.use_nodule_cls:
            cls_nodule_loss, cls_nodule_preds, cls_nodule_targets = out["cls_nodule"]
            if cls_nodule_preds is not None:
                self.test_loss["cls_nodule"](sum(cls_nodule_loss.values()))
                self.log("test/loss/cls_nodule", self.test_loss["cls_nodule"], on_step=False, on_epoch=True, prog_bar=True,  metric_attribute=f"test_loss_cls_nodule")
                for group in self.nodule_cls_groups:
                    self.test_precision[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"test/precision/{group}", self.test_precision[group], on_step=False, on_epoch=True, metric_attribute=f"test_precision_{group}")
                    self.test_acc[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"test/acc/{group}", self.test_acc[group], on_step=False, on_epoch=True, metric_attribute=f"test_acc_{group}")
                    self.test_recall_sens[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"test/recall_sens/{group}", self.test_recall_sens[group], on_step=False, on_epoch=True, metric_attribute=f"test_recall_sens_{group}")
                    self.test_spec[group](cls_nodule_preds[group], cls_nodule_targets[group])
                    self.log(f"test/spec/{group}", self.test_spec[group], on_step=False, on_epoch=True, metric_attribute=f"test_spec_{group}")

        ############### Classification Lung Position ###############
        if self.use_lung_pos:
            cls_lung_pos_loss, lung_pos_preds, lung_pos_labels = out["cls_lung_pos"]
            self.test_loss["cls_lung_pos"](sum(cls_lung_pos_loss.values()))
            self.log("test/loss/cls_lung_pos", self.test_loss["cls_lung_pos"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_loss_cls_lung_pos")
            for position in self.lung_pos_groups:
                self.test_acc[position](lung_pos_preds[position], lung_pos_labels[position])
                self.log(f"test/acc/{position}", self.test_acc[position], on_step=False, on_epoch=True, metric_attribute=f"test_acc_{position}")

        ############### Segmentation Lung Location ###############
        if self.use_lung_loc:
            seg_lung_loc_loss, seg_lung_loc_preds, seg_lung_loc_targets = out["seg_lung_loc"]

            self.test_loss["seg_lung_loc"](seg_lung_loc_loss)
            self.test_dice["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)
            self.test_iou["seg_lung_loc"](seg_lung_loc_preds, seg_lung_loc_targets)

            self.log("test/loss/seg_lung_loc", self.test_loss["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_loss_seg_lung_loc")
            self.log("test/dice/seg_lung_loc", self.test_dice["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_dice_seg_lung_loc")
            self.log("test/iou/seg_lung_loc", self.test_iou["seg_lung_loc"], on_step=False, on_epoch=True, prog_bar=True, metric_attribute=f"test_iou_seg_lung_loc")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
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

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "segcls")

    @hydra.main(version_base=None, config_path=config_path, config_name="segcls_module_new_models.yaml")
    def main(cfg: DictConfig):
        cfg["use_nodule_cls"] = True
        cfg["use_lung_pos"] = True
        cfg["use_lung_loc"] = True
        cfg["net"]["cls_nodule_net"]["n_classes"] = [4, 8, 3, 5, 4]
        cfg["net"]["matching_threshold"] = 0.5
        cfg["net"]["use_nodule_cls"] = True
        cfg["net"]["use_lung_pos"] = cfg["use_lung_pos"]
        cfg["net"]["use_lung_loc"] = cfg["use_lung_loc"]
        cfg["net"]["seg_net"]["use_lung_loc"] = cfg["use_lung_loc"]
        cfg["net"]["seg_net"]["nodule_decoder"]["channel"] = 32
        print(cfg)

        segcls_module: SegClsModule = hydra.utils.instantiate(cfg)

        slices = torch.randn(1, 352, 352)
        seg_nodule_targets = torch.ones((1, 352, 352))
        nodules_infos = {
            "bbox": torch.tensor([[[10, 10, 350, 30], 
                                [1, 1, 350, 350],
                                [1, 1, 350, 350]]]),
            "label":{
                "dam_do": torch.tensor([[[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]]], dtype=torch.float32),
                "voi_hoa": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1]]], dtype=torch.float32),
                "chua_mo": torch.tensor([[[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]]], dtype=torch.float32),
                "duong_vien": torch.tensor([[[1, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 1]]], dtype=torch.float32),
                "tao_hang": torch.tensor([[[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]]], dtype=torch.float32),
            }
        }

        batch = {
            "slice": slices,
            "slice_info": [None],
            "seg_nodule": seg_nodule_targets,
            "nodule_info": nodules_infos,
        }
        if cfg["use_lung_pos"]:
            cls_lung_pos_targets = {
                "right_lung": torch.tensor([[1, 0]], dtype=torch.float32),
                "left_lung": torch.tensor([[1, 0]], dtype=torch.float32)
            }
            batch["cls_lung_pos"] = cls_lung_pos_targets
        if cfg["use_lung_loc"]:
            seg_lung_loc_targets = torch.ones(1, 1, 352, 352)
            batch["seg_lung_loc"] = seg_lung_loc_targets

        import copy
        out = segcls_module.model_step(copy.deepcopy(batch))

        seg_nodule_loss, seg_nodule_preds, _ = out["seg_nodule"]
        print("******** Segmentation Nodule **********")
        print("Loss:", seg_nodule_loss)
        print("Pred: ", seg_nodule_preds.shape, seg_nodule_preds.dtype)

        cls_nodule_loss, nodule_preds, nodule_labels = out["cls_nodule"]
        print("******** Classification Nodule **********")
        for key in nodule_preds.keys():
            print(f"{key}: ", cls_nodule_loss[key], nodule_preds[key].shape, nodule_preds[key].dtype)

        if cfg["use_lung_pos"]:
            cls_lung_pos_loss, cls_lung_pos_preds, _ =  out["cls_lung_pos"]
            print("******** Classification Lung Position **********")
            for key in cls_lung_pos_preds.keys():
                print(f"{key}: ", cls_lung_pos_loss[key], cls_lung_pos_preds[key].shape, cls_lung_pos_preds[key].dtype)

        if cfg["use_lung_loc"]:
            seg_lung_loc_loss, seg_lung_loc_preds, _ = out["seg_lung_loc"]
            print("******** Segmentation Lung Location **********")
            print("Loss:", seg_lung_loc_loss)
            print("Pred: ", seg_lung_loc_preds.shape, seg_lung_loc_preds.dtype)

    main()
