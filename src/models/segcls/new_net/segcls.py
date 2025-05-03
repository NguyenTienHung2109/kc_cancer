import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.segcls.new_net import CaraNet, MLP

class SegClsNet(nn.Module):

    def __init__(self, 
                seg_net: CaraNet,
                use_nodule_cls: bool = True,
                matching_threshold: float | None = 0.5,
                additional_pos_bbox: bool | None = False,
                additional_neg_bbox: bool | None = False,
                scale_bbox: float = 1.0,
                cls_nodule_net: MLP | None = None,
                use_lung_pos: bool = False,
                cls_lung_pos_net: MLP | None = None,
                use_lung_loc: bool = False,) -> None:
        super().__init__()

        self.seg_net = seg_net

        self.use_nodule_cls = use_nodule_cls
        self.use_lung_pos = use_lung_pos
        self.use_lung_loc = use_lung_loc

        if use_nodule_cls:
            self.cls_nodule_net = cls_nodule_net
            self.matching_threshold = matching_threshold
            self.additional_pos_bbox = additional_pos_bbox
            self.additional_neg_bbox = additional_neg_bbox
            self.scale_bbox = scale_bbox

        if use_lung_pos:
            self.cls_lung_pos_net = cls_lung_pos_net

    def seg_postprocess(self, logits: torch.Tensor):
        if logits.shape[1] == 1: # binary segmentation
            probs = torch.sigmoid(logits)
            pred_masks = (probs > 0.5).to(torch.uint8)
        else:
            probs = F.softmax(logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1)
            pred_masks = F.one_hot(pred_masks, num_classes=probs.shape[1]).permute(0, 3, 1, 2)
        return pred_masks.detach().cpu().numpy()

    def check_tiny_nodule(self, bbox, min_area: int = 25): # for image (512, 512)
        return bbox[2] * bbox[3] < min_area

    def find_bbox(self, mask: np.array):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(contour) for contour in contours]

    def get_training_cls(self, bboxes, pred_bboxes, labels, image_size):
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

        def expand_bbox(bbox, max_size, scale=1.5):
            if scale == 1.0:
                return bbox
            
            x, y, w, h = bbox
            expand_w, expand_h= int(w*scale), int(h*scale)
            c_x, c_y = x + int(w / 2), y + int(h / 2)
            x1 = c_x - int(expand_w / 2)
            y1 = c_y - int(expand_h / 2)
            
            if x1 < 0: 
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x1 + expand_w > max_size:
                x1 = max_size - expand_w
            if y1 + expand_h > max_size:
                y1 = max_size - expand_h
            
            return x1, y1, expand_w, expand_h

        first_group = next(iter(labels.values()))
        last_group = next(reversed(labels.values()))
        index = 0
        for i in range(len(first_group)):
            if first_group[i][-1] == 1 and last_group[i][-1] == 1:
                break
            index += 1
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
            training_bboxes.extend(expand_bbox(pred_bboxes[id], scale=self.scale_bbox, max_size=image_size) 
                                for id in pos_ids)
            for key in labels.keys():
                labels[key] = torch.cat((labels[key], pos_labels[key][gt_ids]), dim=0)

        if not self.additional_pos_bbox:
            for i, bbox in enumerate(pos_bboxes.detach().cpu().tolist()):
                if i in pos_ids: continue # not use postive_bbox that not overlap
                training_bboxes.append(expand_bbox(bbox, scale=self.scale_bbox, max_size=image_size))
                for key in labels.keys():
                    labels[key] = torch.cat((labels[key], pos_labels[key][i:i+1]), dim=0)

        n_training_positive = len(training_bboxes)

        # negative
        neg_ids = [i for i in range(len(pred_bboxes)) if i not in pos_ids]
        neg_ids = neg_ids[:min(n_training_positive, len(neg_ids))]

        if neg_ids:
            # TODO: whether require expand_bbox
            training_bboxes.extend(expand_bbox(pred_bboxes[id], scale=self.scale_bbox, max_size=image_size) for id in neg_ids)
            for key in labels.keys():
                neg_label = torch.zeros((len(neg_ids), labels[key].shape[1]), device=bboxes.device)
                neg_label[:, -1] = 1
                labels[key] = torch.cat((labels[key], neg_label), dim=0)

        if self.additional_neg_bbox and n_training_positive > len(neg_ids):
            n_addition_neg_bbox = min(n_training_positive - len(neg_ids), neg_bboxes.shape[0])

            training_bboxes.extend(neg_bboxes[:n_addition_neg_bbox].detach().cpu().tolist())
            for key in labels.keys():
                neg_label = torch.zeros((n_addition_neg_bbox, labels[key].shape[1]), device=bboxes.device)
                neg_label[:, -1] = 1
                labels[key] = torch.cat((labels[key], neg_label), dim=0)

        return training_bboxes, labels

    def forward(self, images: torch.Tensor, nodules_infos=None, training: bool=True):
        # segmentation nodule
        out = self.seg_net(images)

        if training:
            res = {"seg_nodule": out["seg_nodule"]}
            if self.use_lung_loc:
                res["seg_lung_loc"] = out["seg_lung_loc"] # segmentation lung_loc
        else: # inference
            res = {"seg_nodule": self.seg_postprocess(out["seg_nodule"][0])}
            if self.use_lung_loc:
                res["seg_lung_loc"] = self.seg_postprocess(out["seg_lung_loc"][0]) # segmentation lung_loc

        if self.use_lung_pos: # classification lung position
            embedding = out["fm"].mean(dim=[2, 3])
            res["cls_lung_pos"] = self.cls_lung_pos_net(embedding)
        
        if self.use_nodule_cls:  # classification nodule
            scale = images.shape[2] // out["fm"].shape[2]
            scale = scale.item() if isinstance(scale, torch.Tensor) else scale
            
            feature_maps = F.interpolate(out["fm"], scale_factor=scale, mode='bilinear') 
            pred_nodule_masks = self.seg_postprocess(out["seg_nodule"][0])
            
            if training:
                cls_nodule_labels = {key: torch.empty((0, n_class), device=images.device) 
                                    for key, n_class in zip(nodules_infos["label"], self.cls_nodule_net.n_classes)}
            else:
                cls_nodule_labels = None
                res["bboxes"] = []

            nodules = []
            # create empty tensor
            
            # matching bbox
            for i in range(feature_maps.shape[0]):
                pred_bboxes = self.find_bbox(mask=pred_nodule_masks[i][0])
                pred_bboxes_filter = []
                
                for bbox in pred_bboxes:
                    if self.check_tiny_nodule(bbox, (25 * (images.shape[2] / 512))**2):
                        if not training: # assign tiny nodule to background
                            x, y, w, h = bbox
                            pred_nodule_masks[i, 0, y: y+h, x: x+w] = 0
                    else: # filter tiny nodule: min_area=25 for image (512x512)
                        pred_bboxes_filter.append(bbox)
                
                if training: # for training
                    cur_bboxes = nodules_infos["bbox"][i]
                    cur_labels = {
                        key: nodules_infos["label"][key][i] for key in nodules_infos["label"].keys()
                    }
                    crop_nodule_bboxes, labels = self.get_training_cls(cur_bboxes, 
                                                                        pred_bboxes_filter, 
                                                                        cur_labels, 
                                                                        image_size=images.shape[2])
                    for key in labels.keys():
                        cls_nodule_labels[key] = torch.cat((cls_nodule_labels[key], labels[key]), dim=0)
                else: # for inference
                    crop_nodule_bboxes = pred_bboxes_filter
                    res["bboxes"].append(crop_nodule_bboxes)
                
                feature_map = feature_maps[i]
                for x, y, w, h in crop_nodule_bboxes:
                    nodules.append(feature_map[:, y: y+h, x: x+w].mean(dim=[1, 2]))
            
            if len(nodules) > 0:
                nodules = torch.stack(nodules, dim=0)
                cls_nodule_preds = self.cls_nodule_net(nodules) # classification nodule
            else:
                cls_nodule_preds = [torch.empty(0, n_class) for n_class in self.cls_nodule_net.n_classes]
            
            if not training:
                res["seg_nodule"] = pred_nodule_masks
            res["cls_nodule"] = (cls_nodule_preds, cls_nodule_labels)
        
        return res

if __name__ == '__main__':
    from src.models.segcls.new_net import Encoder, Decoder

    use_lung_loc = True
    use_lung_pos = True
    segcls_nodule_net = SegClsNet(seg_net=CaraNet(encoder=Encoder(channel=112),
                                                decoder=Decoder(channel=112, out_channel=7),
                                                use_lung_loc=use_lung_loc),
                                cls_lung_pos_net=MLP(input_size=1024,
                                                    hidden_size_list=[256, 64, 16],
                                                    n_classes=[2, 2]),
                                cls_nodule_net=MLP(input_size=1024, 
                                            hidden_size_list=[256, 64, 16], 
                                            n_classes=[4, 8, 3, 5, 4]),
                                use_nodule_cls=True,
                                matching_threshold=0.5,
                                additional_pos_bbox=True,
                                additional_neg_bbox=True,
                                use_lung_loc=use_lung_loc,
                                use_lung_pos=use_lung_pos)

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

    res = segcls_nodule_net(images, nodules_infos)

    seg_nodule_logits = res["seg_nodule"]
    print("********** Segmentation Nodule Output **********")
    for logit in seg_nodule_logits:
        print(logit.shape)

    cls_nodule_preds, cls_nodule_labels = res["cls_nodule"]
    print("********** Classification Nodule Output **********")
    for pred in cls_nodule_preds:
        print(pred.shape)
    print("********** Classification Nodule Target **********")
    for key in cls_nodule_labels.keys():
        print(cls_nodule_labels[key].shape)

    if use_lung_pos:
        lung_pos_preds = res["cls_lung_pos"]
        print("********** Classification Lung Position Output **********")
        for pred in lung_pos_preds:
            print(pred.shape)

    if use_lung_loc:
        lung_loc_logits = res["seg_lung_loc"]
        print("********** Segmentation Lung Location Output: **********")
        for logit in lung_loc_logits:
            print(logit.shape)
