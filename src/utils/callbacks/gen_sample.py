from typing import Any, Tuple, List, Dict

import torch
from torch import Tensor
from lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import Callback

from src.models.seg import CaraNetModule, UNetModule
from src.models.segcls import SegClsModule

def draw_masks(masks_batch):
    # masks: (w, h, c)
    colors = torch.tensor([(0, 0, 0), (0.5, 0.75, 1.0), (0.0, 0.5, 0.75), 
                            (0.0, 0.0, 1.0), (1.0, 0.75, 0.5), (1.0, 0.5, 0.0)], 
                        dtype=torch.float32, device=masks_batch.device)
    b, c, w, h = masks_batch.shape
    colored_masks = torch.zeros((b, 3, w, h), dtype=torch.float32, device=masks_batch.device)
    for i in range(c):
        mask = masks_batch[:, i:i+1, :, :]  # (b, 1, w, h)
        colored_masks += mask * colors[i].view(1, 3, 1, 1) 
    return colored_masks

class GenSample(Callback):

    def __init__(
        self,
        grid_shape: Tuple[int, int],
    ):
        """_summary_

        Args:
            grid_shape (Tuple[int, int]): _description_

        Raises:
            NotImplementedError: _description_
        """
        self.grid_shape = grid_shape

    # train
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                        outputs: Any, batch: Any, batch_idx: int) -> None:
        if batch_idx == 0:
            self.infer(pl_module, batch, mode="train")

    # validation
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, 
                                outputs: Tensor | Dict[str, Any] | None, batch: Any, 
                                batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.infer(pl_module, batch, mode="val")

    # test
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, 
                          outputs: Tensor | Dict[str, Any] | None, batch: Any, 
                          batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.infer(pl_module, batch, mode="test")

    @torch.no_grad() 
    def infer(self, pl_module: LightningModule, batch: Any, mode: str):
        slices, nodule_masks = batch["slice"], batch["seg_nodule"]

        # avoid out of memory
        n_samples = min(self.grid_shape[0] * self.grid_shape[1], slices.shape[0])

        slices = slices[:n_samples]
        nodule_masks = nodule_masks[:n_samples]

        if isinstance(pl_module, (CaraNetModule, UNetModule)):
            with pl_module.ema_scope():
                nodule_preds = pl_module.predict(slices)

                self.log_sample([slices, nodule_masks, nodule_preds],
                                pl_module=pl_module,
                                nrow=self.grid_shape[0],
                                mode=mode,
                                caption=['image', 'mask', 'pred'])

        elif isinstance(pl_module, SegClsModule):
            if pl_module.use_lung_loc:
                lung_loc_masks = batch["seg_lung_loc"][:n_samples]

            with pl_module.ema_scope():
                out = pl_module.predict(slices)
                nodule_preds = out["seg_nodule"]

                images = [slices, nodule_masks, nodule_preds]
                caption = ['image', 'nodule_mask', 'nodule_pred']

                if pl_module.use_lung_loc:
                    lung_loc_preds = out["seg_lung_loc"]
                    images += [draw_masks(lung_loc_masks), draw_masks(lung_loc_preds)]
                    caption += ["lung_loc_mask", "lung_loc_pred"]

                self.log_sample(images,
                                pl_module=pl_module,
                                nrow=self.grid_shape[0],
                                mode=mode,
                                caption=caption)

    def log_sample(self,
                    images: Tensor,
                    pl_module: LightningModule,
                    nrow: int,
                    mode: str,
                    caption=List[str]):

        images = [make_grid(image, nrow=nrow, pad_value=1) for image in images]

        # logging
        pl_module.logger.log_image(key=mode + '/inference',
                                    images=images,
                                    caption=caption)
