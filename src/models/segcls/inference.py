from typing import List, Dict, Tuple
import re
import os
import cv2
import torch
import shutil
import pydicom
import zipfile
import argparse
import rootutils
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn.functional as F
import random


from torchmetrics import Dice
from torchmetrics.classification.accuracy import BinaryAccuracy

import warnings
warnings.filterwarnings("ignore")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.segcls.segcls_module_new_models import SegClsModule
from src.models.segcls.new_net.onnx import InferSliceModule, InferNoduleModule

# TODO: swap 6.0 and 6.1
dict_columns = {
    "slice_info_columns": ["bnid", "raw_id"],
    "nodule": {
        "dam_do": ["Nhóm 1 - Đậm độ - 1.1 Đặc", "Nhóm 1 - Đậm độ - 1.2 Bán đặc", "Nhóm 1 - Đậm độ - 1.3 Kính mờ", "Nhóm 1 - Đậm độ - 1.4 Đông đặc"], 
        "voi_hoa": ["Nhóm 2 - Đậm độ vôi - 2.1 Không có vôi", "Nhóm 2 - Đậm độ vôi - 2.2 Vôi trung tâm"],
        "chua_mo": ["Nhóm 3 - Đậm độ mỡ - 3.1 Không chứa mỡ", "Nhóm 3 - Đậm độ mỡ - 3.2 Có chứa mỡ"],
        "duong_vien": ["Nhóm 4 - Bờ và Đường viền - 4.1 Tròn đều", "Nhóm 4 - Bờ và Đường viền - 4.2 Đa thuỳ", "Nhóm 4 - Bờ và Đường viền - 4.3 Bờ không đều", "Nhóm 4 - Bờ và Đường viền - 4.4 Tua gai"],
        "tao_hang":["Nhóm 5 - Tạo hang - 5.1 Không có", "Nhóm 5 - Tạo hang - 5.2 Hang lành tính", "Nhóm 5 - Tạo hang - 5.3 Hang ác tính"],
        "di_can": ["Nhóm 6 - Di căn phổi - 6.1 - Di căn cùng bên", "Nhóm 6 - Di căn phổi - 6.0 - Không di căn", "Nhóm 6 - Di căn phổi - 6.2 - Di căn đối bên"],
    },
    "bbox": ["left", "top", "width", "height"],
    "lung_pos": ["right_lung", "left_lung"],
    "lung_loc": ["Vị trí giải phẫu - 1. Thuỳ trên phải", "Vị trí giải phẫu - 2. Thuỳ giữa phải", "Vị trí giải phẫu - 3. Thuỳ dưới phải", "Vị trí giải phẫu - 4. Thuỳ trên trái", "Vị trí giải phẫu - 5. Thuỳ dưới trái"],
    "lung_damage": {
        "dong_dac": ["not_dong_dac", "Tổn thương viêm - 1. Đông đặc"], 
        "kinh_mo": ["not_kinh_mo", "Tổn thương viêm - 2. Kính mờ"], 
        "phe_quan_do": ["not_phe_quan_do", "Tổn thương viêm - 3. Hình phế quản đồ"], 
        "nu_tren_canh": ["not_nu_tren_canh", "Tổn thương viêm - 4. Nốt mờ dạng nụ trên cành"],
    },
}

colors = {
    "pred_nodule": (1., 1., 0.),
    "gt_nodule": (0., 1., 0.),
    "pred_lung_loc": [(0.5, 0.75, 1.0), (0.0, 0.5, 0.75), 
                    (0.0, 0.0, 1.0), (1.0, 0.75, 0.5), (1.0, 0.5, 0.0)],
    "gt_lung_loc": [(0.5, 1.0, 0.5), (0.3, 0.8, 0.3), 
                    (0.1, 0.6, 0.1), (0.8, 0.6, 0.9), (0.5, 0.2, 0.7)]
}

def find_bbox(mask: np.array, erode=False):
    mask = (mask > 0.5).astype(np.uint8)
    if erode:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.erode(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, [cv2.boundingRect(contour) for contour in contours]

def extend_label(labels: torch.Tensor):
    labels_arr = []
    for label in labels:
        # negative
        if label[0] == -1:
            labels_arr.append([0] * len(label) + [1])
        else:
            labels_arr.append(label.tolist() + [0])
    return torch.tensor(labels_arr, dtype=torch.float32)

def create_label(label_ids: List[int], n_class: int):
    labels = []
    for index in label_ids:
        if index == -1:
            label = [-1] * n_class
        else:
            label = [0] * n_class
            label[int(index)] = 1
        labels.append(label)
    return torch.tensor(labels, dtype=torch.float32)

def get_masks(mask_folder, mask_names):
    masks = []
    print(mask_names)
    for mask_name in mask_names:
        print(mask_name)
        mask_path = f"{mask_folder}/{mask_name}.npy"
        masks.append(np.load(mask_path))
    return masks

def draw_masks(slice ,masks, colors, thickness, erode=False):
    # slice: (w, h)
    # masks: (w, h, c)
    # colors: tuple()

    masks = masks[:, :, 1:] # first dim is background
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        if mask.sum() > 0:
            contour, _ = find_bbox(mask, erode)
            cv2.drawContours(slice, contour, -1, colors[i], thickness)
    return slice

class SegClsModel(nn.Module):   
    def __init__(self, 
                segcls_checkpoint) -> None:
        
        super().__init__()
        
        self.segcls_module: SegClsModule = SegClsModule.load_from_checkpoint(segcls_checkpoint)
        self.segcls_module.eval()
        # self.segcls_module.model_ema.copy_to(self.segcls_module.net)

        self.original_size = (512, 512)
        self.preprocessed_size = (352, 352)
        self.init_transform()

    def init_transform(self):
        bbox_params = A.BboxParams(format="coco", label_fields=["class_labels"])
        self.transform_img = A.Compose(transforms=[A.Resize(*self.preprocessed_size, p=1),
                                                    A.pytorch.transforms.ToTensorV2()])
        self.rev_transform_mask = A.Compose(transforms=[A.Resize(*self.original_size, p=1)])
        self.transform_img_box = A.Compose(transforms=[A.Resize(*self.preprocessed_size, p=1),
                                                        A.pytorch.transforms.ToTensorV2()], 
                                            bbox_params=bbox_params)
        self.rev_transform_mask_box = A.Compose(transforms=[A.Resize(*self.original_size, p=1)], 
                                                bbox_params=bbox_params)

    def export_onnx(self, onnx_dir: str):
        # segmentation
        print(self.segcls_module.net.seg_net)
        print(self.segcls_module.net.cls_lung_pos_net)
        print(self.segcls_module.use_lung_loc)
        print(self.segcls_module.use_lung_pos)
        print(self.segcls_module.use_nodule_cls)
        infer_slice_model = InferSliceModule(input_shape=(1, *self.preprocessed_size),
                                            seg_net=self.segcls_module.net.seg_net, 
                                            use_lung_pos=self.segcls_module.use_lung_pos,
                                            cls_lung_pos_net=self.segcls_module.net.cls_lung_pos_net,
                                            use_lung_loc=self.segcls_module.use_lung_loc,
                                            get_fm=self.segcls_module.use_nodule_cls)
        infer_slice_model.export_onnx(onnx_dir / "infer_slice.onnx")
        infer_slice_model.test_onnx(onnx_dir / "infer_slice.onnx", device="gpu")

        infer_nodule_model = InferNoduleModule(input_size=1024,
                                            cls_nodule_net=self.segcls_module.net.cls_nodule_net)
        infer_nodule_model.export_onnx(onnx_dir / "infer_nodule.onnx")
        infer_nodule_model.test_onnx(onnx_dir / "infer_nodule.onnx", device="gpu")

    # function for each image
    def ct_normalize(self, image: torch.Tensor): # (h, w) or (c, h, w) or (b, c, h, w) 
        if len(image.shape) == 4:
            min_val = image.amin(dim=(1, 2, 3), keepdim=True)
            max_val = image.amax(dim=(1, 2, 3), keepdim=True)
        else:
            min_val = image.min()
            max_val = image.max()

        if (max_val - min_val).item == 0:
            return image

        return (image - min_val) / (max_val - min_val)

    def preprocess(self, raw_images, nodules_infos=None): # 512, 512 -> 1, 1, 352, 352
        norm_images = self.ct_normalize(raw_images)
        
        bboxes = []
        if self.segcls_module.use_nodule_cls and nodules_infos is not None:
            bboxes = nodules_infos["bbox"]
        if len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            fake_label = [i for i in range(bboxes.shape[0])]
            transformed = self.transform_img_box(image=norm_images[:, :, None], 
                                                bboxes=bboxes, class_labels=fake_label)
        else:
            transformed = self.transform_img(image=norm_images[:, :, None])
        
        downsampled_images = transformed["image"][None, ...]
        if self.segcls_module.use_nodule_cls and nodules_infos is not None:
            nodules_infos["bbox"]  = bboxes.astype(np.int64)
        
        return downsampled_images, nodules_infos

    def seg_postprocess(self, mask: torch.Tensor, bboxes: torch.Tensor = torch.tensor([])) -> Tuple[np.ndarray, np.ndarray]:
        mask = torch.from_numpy(mask).to(torch.float32)
        if mask.shape[1] == 1: # binary mask
            mask = mask[0][0].numpy() # (1, 1, h, w) -> (h, w)
            if len(bboxes) and len(bboxes[0]) != 0:
                
                # convert list to numpy array
                bboxes = np.array(bboxes, dtype=np.float32)
                # erase dim from (1, 1, 4) to (1, 4)
                print("bboxes", bboxes.shape)
                bboxes = bboxes.squeeze(0)
                fake_label = [i for i in range(bboxes.shape[0])]
                
                transformed = self.rev_transform_mask_box(image=mask, 
                                                        bboxes=bboxes, class_labels=fake_label)
            else:
                transformed = self.rev_transform_mask(image=mask)
            
            upsampled_mask = transformed["image"]
            if len(bboxes) and len(bboxes[0]) != 0:
                bboxes = transformed["bboxes"].astype(np.int64)
            return upsampled_mask, bboxes
        
        elif mask.shape[1] == 6:
            upsampled_mask = F.interpolate(mask, size=self.original_size, mode='bilinear')
            return upsampled_mask[0].permute(1, 2, 0).numpy() # (1, c, h, w) -> (h, w, c)

    def cls_postprocess(self, logits: List[torch.Tensor], groups: List[str]):
        probs, preds = {}, {}
        for logit, group in zip(logits, groups):
            # check if logit is a list or NoneType
            if isinstance(logit, list) or logit is None:
                continue
            logit = logit.detach().cpu()
            prob = torch.softmax(logit, dim=1) 
            pred_label = torch.argmax(prob, dim=1)
            probs[group] = prob
            preds[group] = pred_label
        return probs, preds

    # for one slice
    def forward(self, raw_ct_images: np.array, nodules_infos=None): # 512, 512
        # preprocess bbox
        preprocessed_images, nodules_infos = self.preprocess(raw_ct_images, nodules_infos)
        preprocessed_images = preprocessed_images.to(self.segcls_module.device)
        
        if nodules_infos:
            nodules_infos["bbox"] = nodules_infos["bbox"].to(self.segcls_module.device)
            for group in self.segcls_module.nodule_cls_groups:
                nodules_infos["label"][group] = nodules_infos["label"][group].to(self.segcls_module.device)
        
        # inference
        out = self.segcls_module.net(preprocessed_images, nodules_infos, training=False)
        
        # segmentation nodule
        bboxes = out["bboxes"] if self.segcls_module.use_nodule_cls else []
        pred_nodule_mask, bboxes = self.seg_postprocess(out["seg_nodule"], bboxes)
        res = {"seg_nodule": pred_nodule_mask}

        # classification nodule
        if self.segcls_module.use_nodule_cls:
            cls_nodule_logits = out["cls_nodule"]
            cls_nodule_probs, cls_nodule_preds = None, None
            if len(cls_nodule_logits[0]): # not clean and postprocess
                cls_nodule_probs, cls_nodule_preds = self.cls_postprocess(cls_nodule_logits, 
                                                                        self.segcls_module.nodule_cls_groups)
            res["cls_nodule"] = (cls_nodule_probs, cls_nodule_preds)
            res["bboxes"] = bboxes

        # classification lung position
        if self.segcls_module.use_lung_pos:
            res["cls_lung_pos"] = self.cls_postprocess(out["cls_lung_pos"], self.segcls_module.lung_pos_groups)
        
        # segmentation lung location
        if self.segcls_module.use_lung_loc:
            res["seg_lung_loc"] = self.seg_postprocess(out["seg_lung_loc"])
        
        return res

def get_result(segcls_model: SegClsModel, pred, datapoint):
    norm_image = segcls_model.ct_normalize(datapoint["slice"]) # normalize
    image = np.stack([norm_image, norm_image, norm_image], axis=-1) # RGB image
    
    # draw gt segmentation nodule
    if datapoint["seg_nodule"] is not None and datapoint["seg_nodule"].sum() > 0:
        contours, _ = find_bbox(datapoint["seg_nodule"])
        cv2.drawContours(image, contours, -1, colors["gt_nodule"], 1)
    
    # draw predict segmentation nodule
    if pred["seg_nodule"].sum() > 0:
        contours, _ = find_bbox(pred["seg_nodule"])
        cv2.drawContours(image, contours, -1, colors["pred_nodule"], 1)

    # draw predict classification nodule
    if segcls_model.segcls_module.use_nodule_cls:
        # draw probability of classification not nodule
        cls_nodule_probs, _ = pred["cls_nodule"]
        # check if cls_nodule_probs is an empty dict
        if cls_nodule_probs is not None and len(cls_nodule_probs) > 0:
            text_position_x, text_position_y, line_spacing = 5, 50, 20
            cv2.putText(image, "No Label: ", (text_position_x, text_position_y), cv2.FONT_HERSHEY_PLAIN, 1, (1, 1, 1), 2)
            
            not_nodule_probs = [cls_nodule_probs[group][:, -1] \
                                for group in segcls_model.segcls_module.nodule_cls_groups]
            not_nodule_probs = torch.stack(not_nodule_probs, dim=1)
            
            for probs in not_nodule_probs:
                text_position_y += line_spacing
                text = '-'.join([f"{prob:.2f}" for prob in probs])
                cv2.putText(image, text, (text_position_x, text_position_y), cv2.FONT_HERSHEY_PLAIN, 1, (1, 1, 1), 2)

    # draw predict classification lung location
    if segcls_model.segcls_module.use_lung_pos:
        lung_pos_prob, _ = pred["cls_lung_pos"]
        
        text_position_x, text_position_y, line_spacing = 360, 20, 20
        for group in segcls_model.segcls_module.lung_pos_groups:
            lung_pos_text = f"{group}: {lung_pos_prob[group][0, 1]:.2f}"
            cv2.putText(image, lung_pos_text, (text_position_x, text_position_y), cv2.FONT_HERSHEY_PLAIN, 1, (1, 1, 1), 2)
            text_position_y += line_spacing

    if segcls_model.segcls_module.use_lung_loc:
        # draw predict segmentation lung location
        if  datapoint["seg_lung_loc"] is not None:
            image = draw_masks(image, datapoint["seg_lung_loc"], colors["gt_lung_loc"], thickness=2, erode=True)
        
        # draw predict segmentation lung location
        image = draw_masks(image, pred["seg_lung_loc"], colors["pred_lung_loc"], thickness=2)

    return image

def voting_nodule(nodule_probs, threshold: float):
    voted_probs = [prob for prob in nodule_probs if prob > threshold]
    pos_votes = len(voted_probs)
    total_votes = len(nodule_probs)
    avg_prob = sum(voted_probs) / len(voted_probs) if voted_probs else 0.0
    return pos_votes * 2 > total_votes, avg_prob

def lung_loc_postprocess(lung_loc_mask):
    for i in range(lung_loc_mask.shape[-1]):
        mask = lung_loc_mask[:, :, i]
        contours, _ = find_bbox(mask)
        if len(contours):
            max_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
            lung_loc_mask[:, :, i] = (np.array(mask) / 255.0).astype(lung_loc_mask.dtype)
    
    return lung_loc_mask

def infer_slice(model: SegClsModel, datapoint, des_path: Path,
            show_result: bool = False,
            save_result: bool = False):

    # THRESHOLD
    CLS_NODULE_THRESHOLD = 0.5
    CLS_LUNG_POS_THRESHOLD = [0.35, 0.75]

    out = model(datapoint["slice"], datapoint["nodule_info"])

    # get all inference_result and not filter
    image = get_result(model, pred=out, datapoint=datapoint)
    if show_result:
        plt.imshow(image)
        plt.show()

    if save_result:
        des_path_image = des_path / "image" / f"{datapoint['bnid']}"
        des_path_image.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{des_path_image}/{datapoint['raw_id']}.jpg", image * 255)

    res_slice = [[datapoint["bnid"], datapoint["raw_id"]]]
    if model.segcls_module.use_lung_pos:
        lung_pos_probs, _ = out["cls_lung_pos"]
        lung_pos_res = []
        for group in model.segcls_module.lung_pos_groups:
            # first dim: batch
            # second dim: 0 for probability of outside, 1 for probability of inside
            lung_pos_res.append(lung_pos_probs[group][0, 1].item())
        res_slice[0].extend(lung_pos_res)

        # outside slice
        if lung_pos_res[0] < CLS_LUNG_POS_THRESHOLD[0] and lung_pos_res[1] < CLS_LUNG_POS_THRESHOLD[0]:
            out["seg_nodule"] = np.zeros_like(out["seg_nodule"]) # clean mask
            if model.segcls_module.use_nodule_cls: # remove nodule
                out["bboxes"] = []
                out["cls_nodule"] = None, None
        # outside slice
        if lung_pos_res[0] < CLS_LUNG_POS_THRESHOLD[1] and lung_pos_res[1] < CLS_LUNG_POS_THRESHOLD[1]:
            if model.segcls_module.use_lung_loc:
                out["seg_lung_loc"] = np.zeros_like(out["seg_lung_loc"]) # clean mask

    if model.segcls_module.use_nodule_cls:
        cls_nodule_probs, _ = out["cls_nodule"]
        
        error = False
        # bboxes must match with cls_nodule_probs
        if len(out["bboxes"]) == 0 and cls_nodule_probs is not None:
            error = True
        if cls_nodule_probs is not None and len(cls_nodule_probs) != 0:
            if len(out["bboxes"]) > 0 and (cls_nodule_probs is None or \
                (len(out["bboxes"]) != len(cls_nodule_probs[model.segcls_module.nodule_cls_groups[0]]))):
                error = True
        if error:
            with open(log_filepath, "a", encoding="utf-8") as file:
                file.write(f"{datapoint['bnid']} - {datapoint['raw_id']}: Not enough nodule is classified\n")
        
        slice_info, res_slice, nodule_id = res_slice[0], [], -1
        for i, bbox in enumerate(out["bboxes"]):
            if len(cls_nodule_probs) > 0:
                _cls_nodule_probs = [cls_nodule_probs[group][i].tolist()[:-1] for group in model.segcls_module.nodule_cls_groups]
                nodule_id += 1
                nodule_info = [nodule_id] + list(bbox) + sum(_cls_nodule_probs, [])
                res_slice.append(slice_info + nodule_info)
            
        # no nodule -> clean slice
        if len(res_slice) == 0:
            _cls_nodule_probs = [-1] * (sum(model.segcls_module.nodule_cls_classes) - len(model.segcls_module.nodule_cls_classes))
            bbox = [-1] * len(dict_columns["bbox"])
            no_nodule_info = [nodule_id] + bbox + _cls_nodule_probs
            res_slice.append(slice_info + no_nodule_info)

    res = {
        "res_slice": res_slice,
        "seg_nodule": out["seg_nodule"]
    }
    if model.segcls_module.use_nodule_cls:
        res["cls_nodule"] = out["cls_nodule"]
    if model.segcls_module.use_lung_pos:
        res["cls_lung_pos"] = out["cls_lung_pos"]
    if model.segcls_module.use_lung_loc:
        res["seg_lung_loc"] = lung_loc_postprocess(out["seg_lung_loc"])
    
    image_postprocessed = get_result(model, pred=out, datapoint=datapoint)
    if save_result:
        des_path_image = des_path / "image_postprocessed" / f"{datapoint['bnid']}"
        des_path_image.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(image_postprocessed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{des_path_image}/{datapoint['raw_id']}.jpg", image * 255)

    return res

def infer_meta_file(model: SegClsModel, data_dir: Path, des_path: Path,
                    meta_file: str, args_results: Dict[str, bool]):
    # metric
    dice = {
        "nodule": Dice(ignore_index=0),
    }    
    acc = {}

    df = pd.read_csv(data_dir / meta_file)
    # Convert stringified lists in DataFrame columns back to lists
    df_columns = dict_columns["slice_info_columns"]
    str_columns = ["mask_name", "lung_damage_name"]
    if model.segcls_module.use_lung_pos:
        df_columns.extend(dict_columns["lung_pos"])
        acc["right_lung"] = BinaryAccuracy()
        acc["left_lung"] = BinaryAccuracy()
    
    # columns slice info before nodule info
    if model.segcls_module.use_nodule_cls:
        use_nodule_cls = "dam_do" in model.segcls_module.nodule_cls_groups
        use_lung_damage_cls = "dong_dac" in model.segcls_module.nodule_cls_groups
        df_columns.extend(["nodule_id"] + dict_columns["bbox"])
        str_columns.extend(dict_columns["bbox"])
        
        if use_lung_damage_cls:
            for _, group_cols in dict_columns["lung_damage"].items():
                df_columns.extend(group_cols)
                str_columns.append(group_cols[1])
        
        if use_nodule_cls:
            for group, group_cols in dict_columns["nodule"].items():
                str_columns.extend(group_cols)
                df_columns.extend(group_cols)
    
    if model.segcls_module.use_lung_loc:
        str_columns.extend(["lung_loc_name"] + dict_columns["lung_loc"])
        dice["lung_loc"] = Dice(ignore_index=0, num_classes=6)
    
    def should_eval(x):
        return isinstance(x, str) and x.strip().startswith(('[' , '{', '(', '"', "'"))

    for col in str_columns:
        df[col] = df[col].apply(lambda x: eval(x) if should_eval(x) else x)


    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        code, bnid, raw_id = row["code"], row["bnid"], row["raw_id"]
        study_id = row["image_name"][0:10]
        image_name = row["image_name"]
        if "NC" in code:
            group = "nc_bo_sung"
        elif "CTB" in code:
            group = "nhom_benh"
        else:
            group = "nhom_chung" 
        
        if "cvat" in image_name: # for data from cvat
            ct_image_path = image_name
        elif "2.6" in image_name: # for data from kc version 2.6
            ct_image_path = image_name
        else:
            if row["is_outside"]:
                ct_image_path = data_dir / group / "Outside" / study_id / f"{image_name}.npy"    
            else:
                ct_image_path = data_dir / group / "Image" / study_id / f"{image_name}.npy"
        raw_ct_image = np.load(ct_image_path).astype(np.float32)

        # nodule segmetation ground-truth
        nodule_masks = [np.zeros_like(raw_ct_image)]
        nodule_masks.extend(get_masks(mask_folder=data_dir / group / "Mask" / study_id,
                                    mask_names=row["mask_name"]))
        nodule_masks.extend(get_masks(mask_folder=data_dir / group / "LungDamage" / study_id,
                                    mask_names=row["lung_damage_name"]))
        nodule_mask = np.logical_or.reduce(nodule_masks)
        nodule_mask = torch.from_numpy(nodule_mask)

        # a datapoint for each slice
        datapoint = {
            "slice": raw_ct_image,
            "bnid": bnid,
            "raw_id": raw_id,
            "seg_nodule": nodule_mask,
            "nodule_info": None,
        }
        
        if model.segcls_module.use_lung_pos:
            # classification lung position ground-truth
            lung_pos_labels = {
                lung: torch.tensor([0 if row.at[col] == 0 else 1], dtype=torch.int8)
                for lung, col in zip(["right_lung", "left_lung"], dict_columns["lung_loc"])
            }

        if model.segcls_module.use_lung_loc:
            # segmetation lung location ground-truth
            lung_loc_masks = get_masks(mask_folder=data_dir / group / "LungLoc" /study_id,
                                    mask_names=row["lung_loc_name"])
            lung_loc_mask = np.zeros((*raw_ct_image.shape, len(dict_columns["lung_loc"])))
            for lung_loc_id, col in enumerate(dict_columns["lung_loc"]):
                if 1 in row[col]:
                    lung_loc_mask[:, :, lung_loc_id] = lung_loc_masks[row[col].index(1)]
            background_mask = np.logical_not(np.any(lung_loc_mask, axis=-1))[..., None]
            lung_loc_mask = np.concatenate([background_mask, lung_loc_mask], axis=-1)
            # datapoint["seg_lung_loc"] = torch.from_numpy(lung_loc_mask)

        out = infer_slice(model, datapoint, des_path = des_path / group,
                        save_result=args_results["save_res"], 
                        show_result=args_results["show_res"])
        results.extend(out["res_slice"])

        # update dice for segmentation nodule
        targets = torch.from_numpy(datapoint["seg_nodule"], dtype=torch.int64)
        dice["nodule"].update(out["seg_nodule"], targets)
        
        if model.segcls_module.use_lung_pos:
            # update accuracy for classification lung position
            _, lung_pos_preds = out["cls_lung_pos"]
            acc["right_lung"].update(lung_pos_preds["right_lung"], lung_pos_labels["right_lung"])
            acc["left_lung"].update(lung_pos_preds["left_lung"], lung_pos_labels["left_lung"])
        
        if model.segcls_module.use_lung_loc:
            # update dice for segmentation lung location
            pred = torch.argmax(out["seg_lung_loc"], dim=-1)
            target = torch.argmax(datapoint["seg_lung_loc"], dim=-1)
            dice["lung_loc"].update(pred, target)

    print("Segmentation Nodule Dice:",  dice["nodule"].compute())

    if model.segcls_module.use_lung_pos:
        print("Classification Right Lung Accuracy:", acc["right_lung"].compute())
        print("Classidication Left Lung Accuracy:", acc["left_lung"].compute())
    
    if model.segcls_module.use_lung_loc:
        print("Segmentation Lung Location Dice:",  dice["lung_loc"].compute())
    
    if args_results["save_csv"]:
        out_df = pd.DataFrame(results, columns=df_columns)
        des_path_csv = os.path.join(des_path, meta_file.split('.')[0] + ".csv")
        out_df.to_csv(des_path_csv, index=False)

def infer_dicom_file(model: SegClsModel, zip_path: str, src_path: str, 
                    des_path: str, args_results: Dict[str, bool], 
                    n_case: int = -1, skip_cases: List[str] = [], 
                    range_infer: Tuple[int, int] = (0, -1)):

    df_columns = dict_columns["slice_info_columns"]
    if model.segcls_module.use_lung_pos:
        df_columns.extend(dict_columns["lung_pos"])
    # columns slice info before nodule info
    if model.segcls_module.use_nodule_cls:
        use_nodule_cls = "dam_do" in model.segcls_module.nodule_cls_groups
        use_lung_damage_cls = "dong_dac" in model.segcls_module.nodule_cls_groups
        df_columns.extend(["nodule_id"] + dict_columns["bbox"])
        if use_lung_damage_cls:
            for _, group_cols in dict_columns["lung_damage"].items():
                df_columns.extend(group_cols)
        
        if use_nodule_cls:
            for group, group_cols in dict_columns["nodule"].items():
                df_columns.extend(group_cols)

    study_paths = get_studies(zip_path)
    study_paths = sorted(study_paths)[range_infer[0]:range_infer[1]]
    for study_path in tqdm(study_paths, total=len(study_paths)):
        results_case = []
        
        # get folder name
        bnid = os.path.basename(study_path).split(".zip")[0]
        if bnid in skip_cases: continue
        
        # try:
        # unzip folder if needed
        is_unzip = False
        if ".zip" in study_path:
            study_path = unzip_study(study_path, src_path)
            is_unzip = True
        
        dicom_paths = get_dicoms(study_path)
        dicom_paths = sorted_dicom(dicom_paths)
        
        with open(log_filepath, "a", encoding="utf-8") as file:
            file.write(f"{bnid}\n")
        
        if len(dicom_paths) == 0:
            with open(log_filepath, "a", encoding="utf-8") as file:
                file.write(f"Empty case, zero slice\n")

        # -1 -> infer all
        n_case -= 1
        if n_case == 0: break

        # rand = random.choices([0, 1], weights=[0.95, 0.05], k=1)[0] # only save 50 case
        rand = 1
        for dicom_path, raw_id in tqdm(dicom_paths, total=len(dicom_paths)):
            # checking valid ct
            is_ct, _ = check_dicom(dicom_path)
            if not is_ct: continue  
            
            # data for a slice
            raw_data = read_input(dicom_path).astype(np.float32)
            datapoint = {
                "slice": raw_data,
                "bnid": bnid,
                "raw_id": raw_id,
                "seg_nodule": None,
                "nodule_info": None,
            }
            if model.segcls_module.use_lung_loc:
                datapoint["seg_lung_loc"] = None
            
            # model inference
            out = infer_slice(model, datapoint, des_path,
                            save_result=args_results["save_res"] & rand, 
                            show_result=args_results["show_res"])
            results_case.extend(out["res_slice"])
            
            if args_results["save_mask"]:
                if out["seg_nodule"].sum() > 0:
                    # save nodule mask prediction
                    des_path_mask = des_path / "mask" / bnid
                    des_path_mask.mkdir(parents=True, exist_ok=True)
                    pred_nodule_mask = (out["seg_nodule"] > 0.5).astype(np.uint8)
                    cv2.imwrite(des_path_mask / f"{raw_id}.jpg", pred_nodule_mask * 255)
                    
                if model.segcls_module.use_lung_loc:
                    # save lung location mask prediction
                    des_path_mask = des_path / "lung_loc" / bnid
                    des_path_mask.mkdir(parents=True, exist_ok=True)
                    for i in range(1, out["seg_lung_loc"].shape[-1]):
                        pred_lung_loc_mask = (out["seg_lung_loc"][:,:,i] > 0.5).astype(np.uint8)
                        if pred_lung_loc_mask.sum() > 0:
                            cv2.imwrite(des_path_mask / f"{raw_id}_{i}.jpg", pred_lung_loc_mask * 255)
        
        if is_unzip:
            shutil.rmtree(study_path)
        
        if args_results["save_csv"]:
            out_df = pd.DataFrame(results_case, columns=df_columns)
            des_path_csv = des_path / "csv"
            des_path_csv.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(des_path_csv / f"{bnid}.csv", index=False)
        
        # except Exception as e:
        #     with open(log_filepath, "a", encoding="utf-8") as file:
        #         file.write(f"Bnid: {bnid} --- Error: {e}\n")
        #     print(f"Bnid: {bnid} --- Error: {e}\n")

def get_studies(src_path: str):
    ''' 
    Get remain studies in src_path, return a list that contain path to each remained study.
    '''
    studies = os.listdir(src_path)
    study_paths = []
    for study in studies:
        study_paths.append(os.path.join(src_path, study))

    return study_paths

def unzip_study(study_zip_path: str, src_path: str):
    study_name = os.path.basename(study_zip_path).split(".zip")[0]
    study_path = os.path.join(src_path, study_name)
    if not os.path.exists(study_path):
        os.makedirs(study_path)

    with zipfile.ZipFile(study_zip_path, 'r') as zip_ref:
        zip_ref.extractall(study_path)

    return study_path

def get_dicoms(study_path: str):
    '''
    Get all dicoms path from study_path, return a full image paths list.
    '''

    dicom_paths = []
    for root, _, files in os.walk(study_path):
        for file in files:
            dicom_path = os.path.join(root, file)
            dicom_paths.append(dicom_path)

    return dicom_paths

def sorted_alphanumeric(data):
    '''
    Sort data input in alphanumeric order (default: increase). 
    '''
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                    for c in re.split('([0-9]+)', key)]

    return sorted(data, key=alphanum_key)

def check_dicom(dicom_path):
    '''
    Check if dicom_path is ct image dicom without any wrong tag. Return 
    '''
    try:
        dcm = pydicom.dcmread(dicom_path)
        dcm_tag = str(dcm.dir)

        contains = ''
        status = True

        # Check "Patient Protocol" tag
        if "Patient Protocol" in dcm_tag:
            status = False
            contains += "Patient Protocol"

        # Check "Dose Report" tag
        if "Dose Report" in dcm_tag:
            status = False
            if len(contains) != 0:
                contains += ", "
            contains += "Dose Report"

        # Check "pdf" tag
        if "pdf" in dcm_tag or "PDF" in dcm_tag:
            status = False
            if len(contains) != 0:
                contains += ", "
            contains += "PDF"

        if not status:
            return False, "Contain " + contains

        # Check "Pixel Array" length
        pixel_array = dcm.pixel_array
        if len(pixel_array) != 512:
            return False, "Length of pixel array is not 512"

        # Check "PixelSpacing"
        pixel_spacing = dcm.get("PixelSpacing", "-1")
        if pixel_spacing == "-1":
            return False, "Don't have pixel spacing"

        # Check "Modality" tag
        modality = dcm.Modality
        if modality != "CT":
            return False, "Is not CT Modality"

        return True, ""

    except Exception as e:
        return False, str(e)

def read_input(dicom_path: str):
    '''
    Read raw dicom data from dicom_path, return pixel_array of that dicom, an unique_id and pixel spacing tag.
    '''
    raw_data = pydicom.dcmread(dicom_path)

    # series_number = raw_data.SeriesNumber
    # instance_number = raw_data.InstanceNumber

    # wc = raw_data.WindowCenter
    # ww = raw_data.WindowWidth

    raw_data = np.array(raw_data.pixel_array)
    return raw_data #, series_number, instance_number, wc, ww

def sorted_dicom(dicom_paths: List[str], is_filter: bool = True):
    paths = []
    for dicom_path in dicom_paths:
        raw_data = pydicom.dcmread(dicom_path, force=True)
        
        valid = True
        if is_filter:
            try:
                wc = min(raw_data.WindowCenter)
                ww = max(raw_data.WindowWidth)
                valid = -200 >= wc and wc >= -650 and 1000 <= ww and ww <= 1600
            except:
                valid = False
            
            if not valid:
                continue
            
        try:
            img_data = np.array(raw_data.pixel_array)
            if img_data.shape != (512, 512):
                valid = False
            
            series_number = raw_data.SeriesNumber
            instance_number = raw_data.InstanceNumber
        except:
            valid = False
        
        if not valid:
            continue

        paths.append(
            (dicom_path, f"{series_number:02d}_{instance_number:04d}"))

    if is_filter and len(paths) <= 100:
        return sorted_dicom(dicom_paths, is_filter=False)

    def key(element):
        return element[1]

    return sorted(paths, key=key)

def read_log(log_filepath):
    with open(log_filepath, "r", encoding="utf-8") as file:
        cases = file.read().split('\n')
        # drop last empty row
        cases = cases[:-1]
        cases = [case for case in cases if len(case) < 10]
        if len(cases) > 0:
            last_case = cases[-1]
            cases = [case for case in cases if case != last_case]
    cases = list(set(cases))
    print("Finished case:", cases)
    return cases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data based on version")
    parser.add_argument("--version_kc",
                        "-kc",
                        default=3.1,
                        type=float,
                        help="Specify the version data kc")
    parser.add_argument("--metafile",
                        "-meta",
                        default=None,
                        type=str,
                        help="Specify the metainfo file")
    parser.add_argument("--checkpoint",
                        "-ckpt",
                        default=None,
                        type=str,
                        help="Specify the checkpoint")
    parser.add_argument("--dicom",
                        action="store_true",
                        help="Specify dicom data or not")
    parser.add_argument("--group",
                        "-g",
                        default="nc_bo_sung",
                        type=str,
                        choices=[
                            "nhom_benh", "nhom_chung", "nc_bo_sung",
                            "test_studies", "test_cases"
                        ],
                        help="Specify the group")
    parser.add_argument("--n_case",
                        "-n_case",
                        default=-1,
                        type=int,
                        help="Number of case for infer full-slice")
    args = parser.parse_args()

    print(f"Data version {args.version_kc}")

    weights_dir = Path("weights") / f"{args.version_kc}"
    model = SegClsModel(segcls_checkpoint=weights_dir / args.checkpoint)
    # model.export_onnx(onnx_dir=weights_dir)

    args_results = {
        "save_mask": True,
        "save_csv": True,
        "save_res": True,
        "show_res": False,
    }

    if args.dicom:
        print("aloooo")
        # zip_path = Path(f"/data/hpc/ba/kc_cancer_v2/data/compressed/{args.group}")
        zip_path = Path(f"/data/hpc/ba/kc_cancer_v4/data/compressed/{args.group}")
        src_path = zip_path.parent / "source"

        chunk_id, size = 1, 50
        start_id = (chunk_id - 1) * size
        range_infer = (start_id, -1)
        des_path = Path(
            f"data/kc_cancer_{args.version_kc}/inference_full/{args.group}")
        # des_path = des_path / f"{range_infer[0]}_{range_infer[1]}"  # for chunk
        log_filepath = des_path / "log.txt"
        des_path.mkdir(parents=True, exist_ok=True)

        finished_case = []
        if os.path.exists(log_filepath):
            finished_case = read_log(log_filepath)

        with open(log_filepath, "a", encoding="utf-8") as file:
            file.write(f"checkpoint: {args.checkpoint}\n")
            file.write(
                f"chunk: {chunk_id} --- Range: {range_infer[0]} {range_infer[1]} \n"
            )
        infer_dicom_file(model,
                         zip_path,
                         src_path,
                         des_path,
                         args_results=args_results,
                         n_case=args.n_case,
                         skip_cases=finished_case,
                         range_infer=range_infer)
        with open(log_filepath, "a", encoding="utf-8") as file:
            file.write("Finish\n")
    else:
        data_dir = Path(f"data/kc_cancer_{args.version_kc}")
        des_path = data_dir / "inference_slice"
        log_filepath = des_path / "log.txt"
        des_path.mkdir(parents=True, exist_ok=True)
        infer_meta_file(model,
                        data_dir,
                        des_path,
                        meta_file=args.metafile,
                        args_results=args_results)
