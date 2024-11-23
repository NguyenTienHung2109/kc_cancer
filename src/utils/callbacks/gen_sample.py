from typing import Any, Tuple, List, Dict
import cv2
import numpy as np
import torch
from torch import Tensor
from lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import Callback



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
        # images, masks, _, _ = batch
        images, masks = batch
        # avoid out of memory
        n_samples = min(self.grid_shape[0] * self.grid_shape[1], images.shape[0])

        images = images[:n_samples]
        masks = masks[:n_samples]

        with pl_module.ema_scope():
            preds = pl_module.predict(images)
            def draw_contour(images, mask, pred):
                """
                Vẽ contour của pred (màu đỏ) và mask (màu xanh) lên ảnh gốc.

                Args:
                    images (torch.Tensor): Ảnh gốc dạng Tensor, kích thước (B, 1, H, W).
                    mask (torch.Tensor): Mask gốc dạng Tensor, kích thước (B, 1, H, W).
                    pred (torch.Tensor): Mask dự đoán dạng Tensor, kích thước (B, 1, H, W).

                Returns:
                    list of numpy arrays: Danh sách ảnh với contour đã được vẽ.
                """
                result_images = []

                # Đảm bảo đầu vào là tensor
                if not (torch.is_tensor(images) and torch.is_tensor(mask) and torch.is_tensor(pred)):
                    raise ValueError("images, mask, and pred must be PyTorch tensors.")

                # Chuyển đổi tensor về NumPy
                images = images.cpu().numpy()  # (B, 1, H, W)
                mask = mask.cpu().numpy()      # (B, 1, H, W)
                pred = pred.cpu().numpy()      # (B, 1, H, W)

                # Duyệt qua từng ảnh trong batch
                for i in range(images.shape[0]):
                    # Lấy ảnh, mask, và predict tương ứng
                    img = images[i, 0]  # (H, W)
                    msk = mask[i, 0]    # (H, W)
                    prd = pred[i, 0]    # (H, W)

                    # Đảm bảo mask và pred là nhị phân (0 hoặc 255) và kiểu uint8
                    msk = ((msk > 0.5) * 255).astype(np.uint8)  # Chuyển nhị phân và nhân lên 255
                    prd = ((prd > 0.5) * 255).astype(np.uint8)

                    # Chuyển ảnh về định dạng BGR để OpenCV có thể hiển thị
                    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    # Tìm contour của mask và predict
                    contours_mask, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_pred, _ = cv2.findContours(prd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Vẽ contour lên ảnh
                    img_with_contours = img_bgr.copy()
                    cv2.drawContours(img_with_contours, contours_mask, -1, (0, 0, 255), 2)  # Màu đỏ cho mask
                    cv2.drawContours(img_with_contours, contours_pred, -1, (255, 0, 0), 2)  # Màu xanh cho predict

                    # Thêm ảnh đã vẽ vào danh sách kết quả
                    result_images.append(img_with_contours)
                    
                result_images_tensor = torch.tensor(np.stack(result_images)).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    
                return result_images_tensor
            
            images_with_contours = draw_contour(images, masks, preds)   
            
            self.log_sample([images_with_contours, masks, preds],
                            pl_module=pl_module,
                            nrow=self.grid_shape[0],
                            mode=mode,
                            caption=['image', 'mask', 'pred'])

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
