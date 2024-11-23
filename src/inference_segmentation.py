from typing import List, Literal
import os
import cv2
import torch
import torch.nn as nn
import rootutils
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from torchmetrics import Dice

import random
random.seed(12345)

from tqdm import tqdm
from pathlib import Path
from torchmetrics.functional import jaccard_index

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.segmentation import SegmentationModule, CaraNetModule

def visualize(images: List[np.array], names: List[str]):

    _, axs = plt.subplots(1, len(images), figsize=(15, 10))
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(names[i])
        axs[i].axis("off")
    plt.show()

def find_bbox(mask: np.array):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]

def expand_bbox(bbox, expand_w, expand_h, max_size=512):
    x, y, w, h = bbox
    c_x, c_y = x + int(w / 2), y + int(h / 2)

    x1 = c_x - int(expand_w / 2)
    y1 = c_y - int(expand_h / 2)
    
    x2 = x1 + expand_w
    y2 = y1 + expand_h

    if x1 < 0: 
        x1, x2 = 0, expand_w

    if y1 < 0:
        y1, y2 = 0, expand_h

    if x2 > max_size:
        x1, x2 = max_size - expand_w, max_size

    if y2 > max_size:
        y1, y2 = max_size - expand_h, max_size

    return x1, y1, x2, y2

def crop_bbox(pred, mask, bboxes, image, threshold: float = 0.5, size=128):
    for bbox in bboxes:
        cropped_mask = torch.zeros_like(mask)
        x1, y1, x2, y2 = expand_bbox(bbox, expand_w=size, expand_h=size)

        cropped_mask[y1:y2, x1:x2] = pred[y1:y2, x1:x2]

        iou = jaccard_index(cropped_mask, mask, task="binary", average="micro")

        if iou > threshold:
            nodule = np.zeros_like(image)
            nodule[y1:y2, x1:x2] = image[y1:y2, x1:x2]

            expand_nodule = image[y1:y2, x1:x2]

            scale = 1.5
            x1, y1, x2, y2 = expand_bbox(bbox, expand_w = int(bbox[2]*scale), expand_h=int(bbox[3]*scale))
            cropped_nodule = image[y1:y2, x1:x2]

            return nodule, expand_nodule, cropped_nodule, bbox

    return None, None, None, None

class SegmentationModel(nn.Module):
    def __init__(self,
                checkpoint: str,
                net: Literal["unet", "unet-plus-plus", "unet-attention", "caranet"]) -> None:
        super().__init__()

        if net == "caranet":
            self.segmentation_module: CaraNetModule = CaraNetModule.load_from_checkpoint(checkpoint)
            self.input_size = (352, 352)
        else:
            self.segmentation_module: SegmentationModule = SegmentationModule.load_from_checkpoint(checkpoint)
            self.input_size = (128, 128)

        self.net = net
        self.original_size = (512, 512)
        self.segmentation_module.eval()
        self.segmentation_module.model_ema.copy_to(self.segmentation_module.net)

    def export_onnx(self, filepath: str):
        input_sample = torch.randn((1, 1) + self.input_size)

        if self.net == "caranet":
            output_names=["pred_map_5", "pred_map_3", "pred_map_2", "pred_map_1"]
            dynamic_axes={
                "norm_ct_image": {0: "batch_size"},
                "pred_map_5": {0: " batch_size"},
                "pred_map_3": {0: " batch_size"},
                "pred_map_2": {0: " batch_size"},
                "pred_map_1":{0: " batch_size"},
            }
        else:
            output_names=["logits"]
            dynamic_axes={
                "norm_ct_image": {0: "batch_size"},
                "logits": {0: " batch_size"},
            }

        self.segmentation_module.to_onnx(filepath, 
                                        input_sample, 
                                        export_params=True,
                                        input_names=["norm_ct_image"],
                                        output_names=output_names,
                                        dynamic_axes = dynamic_axes)

    # function for each image
    def ct_normalize(self, image: torch.Tensor): # (c, h, w) or (b, c, h, w) 
        if len(image.shape) == 4:
            min_val = image.amin(dim=(1, 2, 3), keepdim=True)
            max_val = image.amax(dim=(1, 2, 3), keepdim=True)
        else:
            min_val = image.min()
            max_val = image.max()

        if (max_val - min_val).item == 0:
            return image

        return (image - min_val) / (max_val - min_val)

    def preprocess(self, raw_images: torch.Tensor): # n, c, h, w = n, 1, 512, 512 -> n, 3, 352, 352
        norm_images = self.ct_normalize(raw_images)
        downsampled_images = nn.functional.interpolate(norm_images, self.input_size, mode="bilinear")

        return downsampled_images

    def postprocess(self, logits: torch.Tensor): # n, 1, 352, 352
        logits = logits.detach().cpu()

        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).to(torch.float32)

        upsampled_pred_masks = nn.functional.interpolate(pred_masks, self.original_size, mode="bilinear")

        return upsampled_pred_masks

    @torch.no_grad()
    def forward(self, raw_images: torch.Tensor): # n, 1, 512, 512
        preprocessed_images = self.preprocess(raw_images)        
        preprocessed_images = preprocessed_images.to(self.segmentation_module.device)

        logits = self.segmentation_module(preprocessed_images)
        if self.net == "caranet":
            logits = logits[0]

        pred_masks = self.postprocess(logits)

        return pred_masks

def main(segmentation_model: SegmentationModel,
        data_dir: str,
        meta_file: str, 
        threshold: int,
        version: str,
        crop_size: int = 128,
        batch_size: int = 1):

    meta_path = os.path.join(data_dir, meta_file)
    df = pd.read_csv(meta_path)

    is_segment = []
    x, y, width, height = [], [], [], []    
    n_clean = 0
    n_nodule = 0
    n_pred_nodule = 0
    n_pred_nodule_overlap = 0
    
    dice = Dice(ignore_index=0)

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):

        code = row["code"]
        study_id = row["image_name"][0:10]
        image_name = row["image_name"]
        mask_name = row["mask_name"]
        slice_id = row["slice_id"]
        nodule_id = row["nodule_id"]
        group = "nhom_benh" if "CTB" in code else "nhom_chung"
    
        is_segment.append(0)
        x.append(-1)
        y.append(-1)
        width.append(-1)
        height.append(-1)
    
        if row["is_clean"]: 
            n_clean += 1

            # negative sampling
            study_folder = os.path.join(data_dir, f"{group}/Clean/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

            # filter_data -> clean
            if not os.path.exists(ct_image_path):
                ct_image_path = ct_image_path.replace("Clean", "")

            raw_ct_image = np.load(ct_image_path).astype(np.float32)
            mask = torch.zeros(raw_ct_image.shape, dtype=torch.int64)

            norm_ct_image = segmentation_model.ct_normalize(raw_ct_image)

            x1, y1 = random.randint(100, 272), random.randint(100, 272)
            x2, y2 = x1 + crop_size, y1 + crop_size

            nodule_image = np.zeros_like(norm_ct_image)
            nodule_image[y1:y2, x1:x2] = norm_ct_image[y1:y2, x1:x2]

            expand_nodule_image = norm_ct_image[y1:y2, x1:x2]
            cropped_nodule_image = expand_nodule_image

            study_folder = os.path.join(data_dir, f"{group}/Nodule_{threshold}_{version}_caranet/Clean", study_id)
            Path(study_folder).mkdir(parents=True, exist_ok=True)

        else:
            n_nodule += 1

            study_folder = os.path.join(data_dir, f"{group}/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")
            mask_path = os.path.join(study_folder.replace("Image", "Mask"), f"{mask_name}.npy")

            raw_ct_image = np.load(ct_image_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.int64)

            norm_ct_image = segmentation_model.ct_normalize(raw_ct_image)
            mask = torch.from_numpy(mask)

        pred_masks = segmentation_model(torch.from_numpy(raw_ct_image[None, None, ...]))

        gt = mask[None, ...]
        dice.update(pred_masks[0], gt)
        pred_mask = pred_masks[0][0]# c, h, w -> h, w
        bboxes = find_bbox((pred_mask.numpy() > 0.5).astype(np.uint8))

        check_result = False
        if check_result and len(bboxes) > 0:
            image = np.stack([norm_ct_image, norm_ct_image, norm_ct_image], axis=-1)
            for bbox in bboxes:
                cv2.rectangle(image, (bbox[0], bbox[1]),
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                (255, 0, 0), 1)
            
            visualize(images=[mask, pred_mask, image], 
                    names=["Mask", "Pred", "Image"])

        if len(bboxes) > 0:
            n_pred_nodule += 1

        if row["is_clean"]:
            if len(bboxes) > 0:
                print(ct_image_path)
        else:
            if len(bboxes) == 0: continue
            nodule_image, expand_nodule_image, cropped_nodule_image, bbox = crop_bbox(pred_mask, 
                                                                                        mask, 
                                                                                        bboxes, 
                                                                                        norm_ct_image, 
                                                                                        threshold, size=crop_size)

            if nodule_image is None: continue
            n_pred_nodule_overlap += 1
            
            is_segment[-1] = 1
            x[-1] = bbox[0]
            y[-1] = bbox[1]
            width[-1] = bbox[2]
            height[-1] = bbox[3]

            study_folder = os.path.join(data_dir, f"{group}/Nodule_{threshold}_{version}_caranet/Image", study_id)
            Path(study_folder).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(study_folder, f"slice_{slice_id:04d}_nodule_{nodule_id:01d}.npy"), 
                nodule_image)
        np.save(os.path.join(study_folder, f"slice_{slice_id:04d}_expand_nodule_{nodule_id:01d}.npy"), 
                expand_nodule_image)
        np.save(os.path.join(study_folder, f"slice_{slice_id:04d}_cropped_nodule_{nodule_id:01d}.npy"), 
                cropped_nodule_image)

    print("Dice:", dice.compute())
    print(n_clean, n_nodule, n_pred_nodule, n_pred_nodule_overlap)

    df["is_segment"] = is_segment
    df["x"] = x
    df["y"] = y
    df["width"] = width
    df["height"] = height
    df.to_csv(os.path.join(data_dir, meta_file.split('/')[0], f"segmentation_meta_info_{version}.csv"))

def export_onnx(model: nn.Module, filepath: str):
    print('-' * 10, "Start export onnx", '-' * 10)
    input_sample = torch.randn((1, 1, 512, 512))
    torch.onnx.export(model,
                    input_sample,
                    filepath,
                    export_params=True,
                    opset_version=14,
                    input_names=["images"],
                    output_names=["pred_masks"],
                    # dynamic_axes={
                    #     "images": {0: " batch_size"},
                    #     "logits": {0 : "batch_size"},
                    # }
                    )

    print('-' * 10, "Finish export onnx", '-' * 10)

if __name__ == "__main__":

    threshold = 0.2
    version = "2.5"

    net = "unet"
    checkpoint = "logs/segmentation/runs/2024-09-06_01-10-35/checkpoints/epoch_099.ckpt"

    net = "caranet"
    checkpoint = "logs/segmentation/runs/2024-11-21_18-37-59/checkpoints/epoch_047.ckpt"

    segmentation_model = SegmentationModel(checkpoint, net)
    segmentation_model.export_onnx(filepath=f"logs/segmentation/{version}/{net}.onnx")
    # export_onnx(model=segmentation_model, filepath=f"logs/{version}/{net}.onnx")

    main(segmentation_model,
        data_dir="data/kc_cancer",
        meta_file=f"nhom_chung/meta_info_{version}.csv", 
        threshold=threshold,
        version=version)

    main(segmentation_model,
        data_dir="data/kc_cancer", 
        meta_file=f"nhom_benh/meta_info_{version}.csv",
        threshold=threshold,
        version=version)