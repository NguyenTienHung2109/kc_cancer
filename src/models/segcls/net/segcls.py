import cv2
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class SegClsNet(nn.Module):

    def __init__(self, 
                seg_net: nn.Module, 
                cls_net: nn.Module, 
                matching_threshold: float = 0.5,
                additional_pos_bbox: bool = False,
                additional_neg_bbox: bool = False) -> None:
        super().__init__()

        self.seg_net = seg_net
        self.cls_net = cls_net
        self.matching_threshold = matching_threshold
        self.additional_pos_bbox, self.additional_neg_bbox = additional_pos_bbox, additional_neg_bbox

    def seg_postprocess(self, logits: torch.Tensor):
        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).to(torch.uint8)
        return pred_masks.detach().cpu().numpy()

    def find_bbox(self, mask: np.array):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(contour) for contour in contours]

    def get_training_cls(self, bboxes, pred_bboxes, labels):
        # bboxes: Tensor - shape = (3, 4)
        # pred_bboxes: List(tuple(int, int, int))
        # labels: Dict{
        #   "dam_do": Tensor - shape = (3, 4)
        #   "voi_hoa": Tensor - shape = (3, 8)
        #   "chua_mo": Tensor - shape = (3, 3)
        #   "duong_vien": Tensor - shape = (3, 5)
        #   "tao_hang": Tensor - shape = (3, 4)
        # }

        def calculate_iou(box1, box2):

            # (x, y, w, h) -> (x1, y1, x2, y2)
            x1_box1, y1_box1 = box1[0], box1[1]
            x2_box1, y2_box1 = box1[0] + box1[2], box1[1] + box1[3]
            x1_box2, y1_box2 = box2[0], box2[1]
            x2_box2, y2_box2 = box2[0] + box2[2], box2[1] + box2[3]

            x1_inter = max(x1_box1, x1_box2)
            y1_inter = max(y1_box1, y1_box2)
            x2_inter = min(x2_box1, x2_box2)
            y2_inter = min(y2_box1, y2_box2)

            intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

            box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
            box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

            union_area = box1_area + box2_area - intersection_area

            iou = intersection_area / union_area if union_area != 0 else 0
            return iou

        index = next((i for i, dam_do in enumerate(labels["dam_do"][:3]) if dam_do[-1] == 1), 3)
        pos_bboxes, neg_bboxes = bboxes[:index], bboxes[index:]
        pos_labels = {key: labels[key][:index] for key in labels.keys()}

        gt_ids, pos_ids = [], []
        for gt_id, bbox in enumerate(pos_bboxes):
            max_iou, max_id = 0, -1

            for pos_id, pred_bbox in enumerate(pred_bboxes):
                if pos_id in pos_ids:
                    continue

                iou = calculate_iou(box1=bbox, box2=pred_bbox)
                if iou > max_iou:
                    max_iou, max_id = iou, pos_id

            if max_iou > self.matching_threshold:
                gt_ids.append(gt_id)
                pos_ids.append(max_id)

        training_bboxes = []
        labels = {key: torch.tensor([]).to(bboxes.device) for key in labels.keys()}

        # positive
        if pos_ids:
            training_bboxes.extend([pred_bboxes[id] for id in pos_ids])
            for key in labels.keys():
                labels[key] = torch.cat((labels[key], pos_labels[key][gt_ids]), dim=0)

        elif self.additional_pos_bbox:
            training_bboxes.extend(pos_bboxes.detach().cpu().tolist())
            for key in labels.keys():
                labels[key] = torch.cat((labels[key], pos_labels[key]), dim=0)

        n_training_positive = len(training_bboxes)

        # negative
        neg_ids = [i for i in range(len(pred_bboxes)) if i not in pos_ids]
        neg_ids = neg_ids[:min(n_training_positive, len(neg_ids))]
        
        if neg_ids:
            training_bboxes.extend([pred_bboxes[id] for id in neg_ids])
            for key in labels.keys():
                neg_label = torch.zeros((len(neg_ids), labels[key].shape[1]), device=bboxes.device)
                neg_label[:, -1] = 1
                labels[key] = torch.cat((labels[key], neg_label), dim=0)

        if n_training_positive > len(neg_ids) and self.additional_neg_bbox:
            n_addition_neg_bbox = min(n_training_positive - len(neg_ids), neg_bboxes.shape[0])

            training_bboxes.extend(neg_bboxes[:n_addition_neg_bbox].detach().cpu().tolist())
            for key in labels.keys():
                neg_label = torch.zeros((n_addition_neg_bbox, labels[key].shape[1]), device=bboxes.device)
                neg_label[:, -1] = 1
                labels[key] = torch.cat((labels[key], neg_label), dim=0)

        return training_bboxes, labels

    def forward(self, images: torch.Tensor, nodules_infos=None):
        
        # segementation
        logits, feature_maps = self.seg_net(images)

        scale = int(images.shape[2] / feature_maps.shape[2])
        feature_maps = F.interpolate(feature_maps, scale_factor=scale, mode='bilinear') 
        pred_masks = self.seg_postprocess(logits[0])

        nodules, cls_pred_labels, cls_labels = [], None, None

        if nodules_infos is not None:
            # for training
            cls_labels = {key: torch.tensor([]).to(images.device) for key in nodules_infos["label"].keys()}

        # matching bbox
        for i in range(feature_maps.shape[0]):
            pred_mask = pred_masks[i][0]
            pred_bboxes = self.find_bbox(mask=pred_mask)
            feature_map = feature_maps[i]

            if nodules_infos is not None:
                # for training
                cur_bboxes = nodules_infos["bbox"][i]
                cur_labels = {
                    key: nodules_infos["label"][key][i] for key in nodules_infos["label"].keys()
                }

                crop_nodule_bboxes, labels = self.get_training_cls(cur_bboxes, pred_bboxes, cur_labels)
                for key in labels.keys():
                    cls_labels[key] = torch.cat((cls_labels[key], labels[key]), dim=0)

            else:
                # for inference
                crop_nodule_bboxes = pred_bboxes

            for x, y, w, h in crop_nodule_bboxes:
                nodules.append(feature_map[:, x: x + w, y: y + h].mean(dim=[1, 2]))

        # classification
        if len(nodules) > 0:
            nodules = torch.stack(nodules, dim=0)
            cls_pred_labels = self.cls_net(nodules)

        else: 
            cls_labels = None

        return logits, (cls_pred_labels, cls_labels)

if __name__ == '__main__':
    from src.models.segcls.net import CaraNet, MLP

    segcls_net = SegClsNet(seg_net=CaraNet(),
                            cls_net=MLP(input_size=1024, 
                                        hidden_size_list=[256, 64, 16], 
                                        n_classes=[4, 8, 3, 5, 4]),
                            matching_threshold=0.)

    images = torch.randn(1, 1, 352, 352)
    nodules_infos = {
        "bbox": torch.tensor([[[10, 10, 350, 30], 
                            [1, 1, 350, 350],
                            [1, 1, 350, 350]]]),
        "label":{
            "dam_do": torch.tensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]]]),
            "voi_hoa": torch.tensor([[[1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]]]),
            "chua_mo": torch.tensor([[[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]]),
            "duong_vien": torch.tensor([[[1, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 1]]]),
            "tao_hang": torch.tensor([[[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]]]),
        }
    }

    seg_preds, (cls_preds, cls_targets) = segcls_net(images, nodules_infos)

    print("----- Segmentation Output: -----")
    for out in seg_preds:
        print(out.shape)

    print("---- Classification Output: -----")
    for out in cls_preds:
        print(out.shape)

    print("----- Classification Input: -----")
    for key in cls_targets.keys():
        print(cls_targets[key].shape)
