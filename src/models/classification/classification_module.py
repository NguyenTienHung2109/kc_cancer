from typing import Any, Tuple, List, Dict

import torch
from torch import Tensor
import rootutils
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric
from torch.optim import Optimizer, lr_scheduler
from contextlib import contextmanager

from torch.nn.functional import sigmoid, softmax
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.specificity import Specificity


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma


class ClassificationModule(LightningModule):

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        groups: List[str],
        tasks: List[str],
        n_labels_or_classes: List[int],
        weight_loss: List[float],
        use_ema: bool = False,
        compile: bool = False,
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # classifier
        self.net = net

        self.criterion = {
            "binary": BCEWithLogitsLoss(),
            "multilabel": BCEWithLogitsLoss(),
            "multiclass": CrossEntropyLoss(),
        }

        self.groups = groups
        self.tasks = tasks
        self.n_labels_or_classes = n_labels_or_classes
        self.weight_loss = weight_loss

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc, self.val_acc, self.test_acc = {}, {}, {}
        self.train_precision, self.val_precision, self.test_precision = {}, {}, {}
        self.train_recall_sens, self.val_recall_sens, self.test_recall_sens = {}, {}, {}
        self.train_spec, self.val_spec, self.test_spec = {}, {}, {}

        for group, task, n_label_or_class in zip(self.groups, self.tasks, self.n_labels_or_classes):
            if task == "binary":
                self.train_acc[group] = Accuracy(task=task)
                self.val_acc[group] = Accuracy(task=task)
                self.test_acc[group] = Accuracy(task=task)

                self.train_precision[group] = Precision(task=task)
                self.val_precision[group] = Precision(task=task)
                self.test_precision[group] = Precision(task=task)

                self.train_recall_sens[group] = Recall(task=task)
                self.val_recall_sens[group] = Recall(task=task)
                self.test_recall_sens[group] = Recall(task=task)

                self.train_spec[group] = Specificity(task=task)
                self.val_spec[group] = Specificity(task=task)
                self.test_spec[group] = Specificity(task=task)


            elif task == "multiclass":
                self.train_acc[group] = Accuracy(task=task, num_classes=n_label_or_class)
                self.val_acc[group] = Accuracy(task=task, num_classes=n_label_or_class)
                self.test_acc[group] = Accuracy(task=task, num_classes=n_label_or_class)

                self.train_precision[group] = Precision(task=task, num_classes=n_label_or_class)
                self.val_precision[group] = Precision(task=task, num_classes=n_label_or_class)
                self.test_precision[group] = Precision(task=task, num_classes=n_label_or_class)

                self.train_recall_sens[group] = Recall(task=task, num_classes=n_label_or_class)
                self.val_recall_sens[group] = Recall(task=task, num_classes=n_label_or_class)
                self.test_recall_sens[group] = Recall(task=task, num_classes=n_label_or_class)

                self.train_spec[group] = Specificity(task=task, num_classes=n_label_or_class)
                self.val_spec[group] = Specificity(task=task, num_classes=n_label_or_class)
                self.test_spec[group] = Specificity(task=task, num_classes=n_label_or_class)

            elif task == "multilabel":
                self.train_acc[group] = Accuracy(task=task, num_labels=n_label_or_class)
                self.val_acc[group] = Accuracy(task=task, num_labels=n_label_or_class)
                self.test_acc[group] = Accuracy(task=task, num_labels=n_label_or_class)

                self.train_precision[group] = Precision(task=task, num_labels=n_label_or_class)
                self.val_precision[group] = Precision(task=task, num_labels=n_label_or_class)
                self.test_precision[group] = Precision(task=task, num_labels=n_label_or_class)

                self.train_recall_sens[group] = Recall(task=task, num_labels=n_label_or_class)
                self.val_recall_sens[group] = Recall(task=task, num_labels=n_label_or_class)
                self.test_recall_sens[group] = Recall(task=task, num_labels=n_label_or_class)

                self.train_spec[group] = Specificity(task=task, num_labels=n_label_or_class)
                self.val_spec[group] = Specificity(task=task, num_labels=n_label_or_class)
                self.test_spec[group] = Specificity(task=task, num_labels=n_label_or_class)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
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

    def forward(self,
                x: Tensor) -> Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
            self, batch: Tuple[Tensor,
                               Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        images, labels = batch

        logits = self.forward(images)

        preds = {}
        losses = {}

        for logit, task, group in zip(logits, self.tasks, self.groups):
            losses[group] = self.criterion[task](logit, labels[group])

            if task == "multiclass":
                pred = softmax(logit, dim=1) 
                pred = torch.argmax(pred, dim=1)
                labels[group] = torch.argmax(labels[group], dim=1)
            else: 
                pred = sigmoid(logit)
                pred = (pred > 0.5).to(torch.int64)

            preds[group] = pred

        return losses, preds, labels

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        # return mask

        if self.use_ema:
            with self.ema_scope():
                logits = self.net(x)
        else:
            logits = self.net(x)

        probs = {}
        preds = {}
        for logit, task, group in zip(logits, self.tasks, self.groups):
            if task == "multiclass":
                prob = softmax(logit, dim=1) 
                pred = torch.argmax(prob, dim=1)
            else: 
                prob = sigmoid(logit)
                pred = (prob > 0.5).to(torch.int64)

            probs[group] = prob
            preds[group] = pred

        return probs, preds
    
    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:

        losses, preds, labels = self.model_step(batch)

        # update and log metrics
        # loss = sum(losses.values()) / len(losses.keys())
        loss = sum(losses.values())
        self.train_loss(loss)

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        for group in losses.keys():
            self.log(f"train/loss/{group}",
                    losses[group].detach(),
                    on_step=False,
                    on_epoch=True)

            self.train_acc[group].to(self.device)
            self.train_acc[group](preds[group], labels[group])
            self.log(f"train/acc/{group}",
                    self.train_acc[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"train_acc_{group}")

            self.train_precision[group].to(self.device)
            self.train_precision[group](preds[group], labels[group])
            self.log(f"train/precision/{group}",
                    self.train_precision[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"train_precision_{group}")

            self.train_recall_sens[group].to(self.device)
            self.train_recall_sens[group](preds[group], labels[group])
            self.log(f"train/recall_sens/{group}",
                    self.train_recall_sens[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"train_recall_sens_{group}")

            self.train_spec[group].to(self.device)
            self.train_spec[group](preds[group], labels[group])
            self.log(f"train/spec/{group}",
                    self.train_spec[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"train_spec_{group}")

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, labels = self.model_step(batch)

        # update and log metrics
        # loss = sum(losses.values()) / len(losses.keys())
        loss = sum(losses.values())
        self.val_loss(loss)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        for group in losses.keys():
            self.log(f"val/loss/{group}",
                    losses[group],
                    on_step=False,
                    on_epoch=True)

            self.val_acc[group].to(self.device)
            self.val_acc[group](preds[group], labels[group])
            self.log(f"val/acc/{group}",
                    self.val_acc[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"val_acc_{group}")

            self.val_precision[group].to(self.device)
            self.val_precision[group](preds[group], labels[group])
            self.log(f"val/precision/{group}",
                    self.val_precision[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"val_precision_{group}")

            self.val_recall_sens[group].to(self.device)
            self.val_recall_sens[group](preds[group], labels[group])
            self.log(f"val/recall_sens/{group}",
                    self.val_recall_sens[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"val_recall_sens_{group}")

            self.val_spec[group].to(self.device)
            self.val_spec[group](preds[group], labels[group])
            self.log(f"val/spec/{group}",
                    self.val_spec[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"val_spec_{group}")


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, labels = self.model_step(batch)

        # update and log metrics
        # loss = sum(losses.values()) / len(losses.keys())
        loss = sum(losses.values())
        self.test_loss(loss)

        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        for group in losses.keys():
            self.log(f"test/loss/{group}",
                    losses[group],
                    on_step=False,
                    on_epoch=True)

            self.test_acc[group].to(self.device)
            self.test_acc[group](preds[group], labels[group])
            self.log(f"test/acc/{group}",
                    self.test_acc[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"test_acc_group")

            self.test_precision[group].to(self.device)
            self.test_precision[group](preds[group], labels[group])
            self.log(f"test/precision/{group}",
                    self.test_precision[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"test_precision_{group}")

            self.test_recall_sens[group].to(self.device)
            self.test_recall_sens[group](preds[group], labels[group])
            self.log(f"test/recall_sens/{group}",
                    self.test_recall_sens[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"test_recall_sens_{group}")

            self.test_spec[group].to(self.device)
            self.test_spec[group](preds[group], labels[group])
            self.log(f"test/spec/{group}",
                    self.test_spec[group],
                    on_step=False,
                    on_epoch=True,
                    metric_attribute=f"test_spec_{group}")


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
    config_path = str(root / "configs" / "model" / "classification")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="resnet_module.yaml")
    def main(cfg: DictConfig):
        cfg["net"]["n_labels_or_classes"] = [3, 7, 1, 4, 3]
        print(cfg)

        classifier_module: ClassificationModule = hydra.utils.instantiate(cfg)

        batch = 5
        labels = {
            "dam_do": torch.tensor(batch*[[1, 0, 0]], dtype=torch.float32), 
            "voi_hoa": torch.tensor(batch*[[1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            "chua_mo": torch.tensor(batch*[[1]], dtype=torch.float32), 
            "duong_vien": torch.ones(batch, 4), 
            "tao_hang": torch.tensor(batch*[[1, 0, 0]], dtype=torch.float32),
        }
        input = torch.randn(batch, 1, 224, 224)
        outputs = classifier_module(input)
        print('*' * 20, ' CLASSIFIER MODULE ', '*' * 20)
        print('Input:', input.shape)
        
        for out in outputs:
            print(out.shape)

        losses, preds, labels = classifier_module.model_step(batch=(input, labels))

        for key in preds.keys():
            print(preds[key].shape, labels[key].shape)

        print(losses)


    main()