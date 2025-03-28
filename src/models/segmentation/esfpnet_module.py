from typing import Any

import torch
import rootutils
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Dice
import torch.nn.functional as F
from contextlib import contextmanager

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.loss import IOU_BCE
from src.utils.ema import LitEma

def iou_score(pred, target, smooth=1e-8):
    """
    Compute the IoU score for binary segmentation.
    :param pred: Predicted tensor of shape [batch_size, height, width].
    :param target: Target tensor of shape [batch_size, 1, height, width].
    :param smooth: A small floating point value to avoid division by zero.
    :return: IoU score.
    """
    # Ensure target has the same dimensions as pred
    target = target.squeeze(1)  # Remove channel dim if it's 1
    
    # Convert predictions to binary by thresholding
    preds_binary = (pred > 0.5).float()
    
    # Flatten the tensors to simplify calculation
    if len(preds_binary.shape) == 2:  # Nếu batch chỉ có 1 mẫu, thêm dimension batch
        preds_binary = preds_binary.unsqueeze(0)
    pred_flat = preds_binary.view(preds_binary.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(1)  # Element-wise multiplication and sum over each image
    union = pred_flat.sum(1) + target_flat.sum(1) - intersection  # Sum of both pred and target minus intersection
    
    # Calculate IoU and average over batch
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

# def calculate_accuracy_sensitivity_specificity(pred_mask, true_mask):
def calculate_accuracy_sensitivity_specificity(pred, target):
    # Ensure pred and target are binary by applying a threshold if necessary
    epsilon = 1e-8
    # pred_binary = (pred > 0.5).float() # Assuming pred is a probability map
    target_binary = target.squeeze(1) # Adjusting target shape to match pred shape

    # # True Positives (TP): Correctly predicted positive observations
    # TP = (pred_binary * target_binary).sum(dim=[1, 2])
    
    # # True Negatives (TN): Correctly predicted negative observations
    # TN = ((1 - pred_binary) * (1 - target_binary)).sum(dim=[1, 2])
    
    # # False Positives (FP): Incorrectly predicted positive observations
    # FP = (pred_binary * (1 - target_binary)).sum(dim=[1, 2])
    
    # # False Negatives (FN): Incorrectly predicted negative observations
    # FN = ((1 - pred_binary) * target_binary).sum(dim=[1, 2])

    # # Sensitivity or Recall or True Positive Rate
    # sensitivity = TP / (TP + FN + epsilon)
    
    # # Specificity or True Negative Rate
    # specificity = TN / (TN + FP + epsilon)
    
    # return sensitivity.mean(), specificity.mean()
    binary_predictions = (pred > 0.5).long()
    
    # Flatten tensors to simplify comparison
    pred_flat = binary_predictions.view(-1)
    labels_flat = target_binary.view(-1)
    
    # True positives, false positives, true negatives, false negatives
    tp = ((pred_flat == 1) & (labels_flat == 1)).float().sum()
    fp = ((pred_flat == 1) & (labels_flat == 0)).float().sum()
    tn = ((pred_flat == 0) & (labels_flat == 0)).float().sum()
    fn = ((pred_flat == 0) & (labels_flat == 1)).float().sum()
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    # print(sensitivity.item(), specificity.item())
    return sensitivity.item(), specificity.item()

def calculate_accuracy(pred, target):
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Ensure pred and target are the same shape
    assert pred.shape == target.shape, "Predictions and targets must have the same shape"
    
    # Flatten tensors to simplify comparison
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute accuracy
    correct = (pred_flat == target_flat).float().sum()
    total = target_flat.numel()
    accuracy = correct / total
    
    return accuracy
class ESFPNetModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_ema: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.img_size= [352,352]
        # loss function
        self.criterion = IOU_BCE(smooth=1,epsilon=1e-8)

        # metric objects for calculating and averaging accuracy across batches
        self.train_dice = Dice()
        self.val_dice = Dice()
        self.test_dice = Dice()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_dice_best = MaxMetric()
        # self.val_miou_best = MaxMetric()
        # self.val_accuracy = MaxMetric()

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

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

    def forward(self, x: torch.Tensor):
        pred = self.net(x)
        pred = F.interpolate(pred, size=self.img_size, mode='bilinear', align_corners=False)
        return pred

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store accuracy from these checks
        self.val_loss.reset()
        self.val_dice_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        y = y.to(torch.int64)
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).to(torch.float32)
        preds = preds.squeeze()
        preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_dice(preds, targets)
        # jaccard_index_value = jaccard_index(preds, targets.squeeze(1),task='binary')
        iou = iou_score(preds,targets)
        
        # print(iou)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_epoch=True, on_step=False, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "iou": iou}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.val_dice(preds, targets)
        # jaccard_index_value = jaccard_index(preds, targets.squeeze(1),task='binary')
        iou = iou_score(preds,targets)
        sen, spe = calculate_accuracy_sensitivity_specificity(preds,targets)
        acc = calculate_accuracy(preds,targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/sensitivity', sen, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/specificity', spe, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets, "iou": iou,'accuracy':acc, 'sensitivity': sen, 'specificity':spe}

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
        # jaccard_index_value = jaccard_index(preds, targets.squeeze(1),task='binary')
        iou = iou_score(preds,targets)
        sen, spe = calculate_accuracy_sensitivity_specificity(preds,targets)
        acc = calculate_accuracy(preds,targets)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/sensitivity', sen, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/specificity', spe, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "iou": iou,'accuracy':acc, 'sensitivity': sen, 'specificity':spe}


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
    # _ = UNetPlusPlusModule(None, None, None)
    import hydra
    from omegaconf import DictConfig

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "segmentation")

    @hydra.main(version_base=None, config_path=config_path, config_name="esfpnet_module.yaml")
    def main(cfg: DictConfig):
        esfpnet_module = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 352, 352)
        logits = esfpnet_module(x)
        preds = torch.argmax(logits, dim=1)
        print(logits.shape, preds.shape)
    
    main()