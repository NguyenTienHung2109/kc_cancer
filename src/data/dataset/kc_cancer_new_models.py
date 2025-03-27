from typing import List
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

def ct_normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val: return image
    return (image - min_val) / (max_val - min_val)

def extend_label(labels: np.array):
    if labels.size == 0:
        return np.empty((0, labels.shape[1] + 1), dtype=np.float32)
    labels_arr = []
    for label in labels:
        # negative
        if label[0] == -1:
            labels_arr.append([0] * len(label) + [1])
        else:
            labels_arr.append(label.tolist() + [0])
    return np.array(labels_arr, dtype=np.float32)

def create_label(label_ids: List[int], n_class: int):
    labels = []
    for index in label_ids:
        if index == -1:
            label = [-1] * n_class
        else:
            label = [0] * n_class
            label[int(index)] = 1
        labels.append(label)
    if labels == []:
        return np.empty((0, n_class), dtype=np.float32)
    return np.array(labels, dtype=np.float32)

def get_masks(mask_folder, mask_names):
    masks = []
    for mask_name in mask_names:
        mask_path = f"{mask_folder}/{mask_name}.npy"
        masks.append(np.load(mask_path))
    return masks

def find_bbox(mask: np.array):
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, [cv2.boundingRect(contour) for contour in contours]

def draw_masks(slice ,masks, colors):
    # slice: (w, h)
    # masks: (w, h, c)
    # colors: tuple()

    colored_mask = np.zeros((*masks.shape[:2], 3), dtype=np.float32)
    masks = masks[:, :, 1:] # first dim is background
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        if mask.sum() > 0:
            colored_mask += np.expand_dims(mask, axis=-1) * colors[i]
            contour, _ = find_bbox(mask)
            cv2.drawContours(slice, contour, -1, colors[i], 2)
    return slice, colored_mask

colors = {
    "nodule": (1., 1., 0.),
    "bbox": (0., 1., 0.),
    "lung_loc": [(0.5, 0.75, 1.0), (0.0, 0.5, 0.75), 
                (0.0, 0.0, 1.0), (1.0, 0.75, 0.5), (1.0, 0.5, 0.0)]
}

error_df = []

class KCSliceDataset(Dataset):
    df_columns = {
        "nodule": {
            "dam_do": ["Nhóm 1 - Đậm độ - 1.1 Đặc", "Nhóm 1 - Đậm độ - 1.2 Bán đặc", "Nhóm 1 - Đậm độ - 1.3 Kính mờ", "Nhóm 1 - Đậm độ - 1.4 Đông đặc"], 
            "voi_hoa": ["Nhóm 2 - Đậm độ vôi - 2.1 Không có vôi", "Nhóm 2 - Đậm độ vôi - 2.2 Vôi trung tâm", "Nhóm 2 - Đậm độ vôi - 2.3 Vôi dạng lá", "Nhóm 2 - Đậm độ vôi - 2.4 Vôi lan toả", "Nhóm 2 - Đậm độ vôi - 2.5 Vôi dạng bắp", "Nhóm 2 - Đậm độ vôi - 2.6 Vôi lấm tấm", "Nhóm 2 - Đậm độ vôi - 2.7 Vôi lệch tâm"],
            "chua_mo": ["Nhóm 3 - Đậm độ mỡ - 3.1 Không chứa mỡ", "Nhóm 3 - Đậm độ mỡ - 3.2 Có chứa mỡ"],
            "duong_vien": ["Nhóm 4 - Bờ và Đường viền - 4.1 Tròn đều", "Nhóm 4 - Bờ và Đường viền - 4.2 Đa thuỳ", "Nhóm 4 - Bờ và Đường viền - 4.3 Bờ không đều", "Nhóm 4 - Bờ và Đường viền - 4.4 Tua gai"],
            "tao_hang":["Nhóm 5 - Tạo hang - 5.1 Không có", "Nhóm 5 - Tạo hang - 5.2 Hang lành tính", "Nhóm 5 - Tạo hang - 5.3 Hang ác tính"],
            "di_can": ["Nhóm 6 - Di căn phổi - 6.1 - Di căn cùng bên", "Nhóm 6 - Di căn phổi - 6.0 - Không di căn", "Nhóm 6 - Di căn phổi - 6.2 - Di căn đối bên"]
        },
        "bbox": ["left", "top", "width", "height"],
        "lung_pos": ["right_lung", "left_lung"],
        "lung_loc": ["Vị trí giải phẫu - 1. Thuỳ trên phải", "Vị trí giải phẫu - 2. Thuỳ giữa phải", "Vị trí giải phẫu - 3. Thuỳ dưới phải", "Vị trí giải phẫu - 4. Thuỳ trên trái", "Vị trí giải phẫu - 5. Thuỳ dưới trái"],
        "lung_damage": {
            "dong_dac": "Tổn thương viêm - 1. Đông đặc", 
            "kinh_mo": "Tổn thương viêm - 2. Kính mờ", 
            "phe_quan_do": "Tổn thương viêm - 3. Hình phế quản đồ", 
            "nu_tren_canh": "Tổn thương viêm - 4. Nốt mờ dạng nụ trên cành",
        },
    }

    def __init__(self,
                data_dir: str,
                meta_file: str,
                use_nodule_cls: bool = False,
                use_lung_pos: bool = False,
                use_lung_loc: bool = False,
                use_lung_damage_cls: bool = False,
                cache_data: bool = False) -> None:
        
        super().__init__()
        
        self.dataset_dir = Path(data_dir)
        meta_path = self.dataset_dir / meta_file
        self.dataFrame = pd.read_csv(meta_path)

        # convert string columns
        str_cols = ["mask_name", "lung_damage_name"]
        str_cols.extend(self.df_columns["bbox"])
        for _, group_cols in self.df_columns["nodule"].items():
            str_cols.extend(group_cols)
        str_cols.extend(["lung_loc_name"] + self.df_columns["lung_loc"])
        str_cols.extend(list(self.df_columns["lung_damage"].values()))
        
        for col in str_cols:
            self.dataFrame[col] = self.dataFrame[col].apply(lambda x: eval(x))

        if cache_data:
            self.data = []
            for index in range(self.__len__()):
                self.data.append(self.__getitem__(index))
        
        self.use_nodule_cls = use_nodule_cls
        self.use_lung_pos = use_lung_pos
        self.use_lung_loc = use_lung_loc
        self.use_lung_damage_cls = use_lung_damage_cls
        self.cache_data = cache_data

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):

        if self.cache_data:
            return self.data[index]
        
        row = self.dataFrame.iloc[index]
        code = row["code"]
        study_id = row["image_name"][0:10]
        image_name = row["image_name"]
        slice_info = f"{row['bnid']}_{row['raw_id']}"

        if "NC" in code:
            group = "nc_bo_sung"
        elif "CTB" in code:
            group = "nhom_benh"
        else:
            group = "nhom_chung" 

        # read image of slice
        if "cvat" in image_name: # for data from cvat
            ct_image_path = image_name
        elif "2.6" in image_name: # for data from kc version 2.6
            ct_image_path = image_name
        else: # for data from the latest kc version 
            if row["is_outside"]:
                ct_image_path = self.dataset_dir/group/"Outside"/study_id/f"{image_name}.npy"    
            else:
                ct_image_path = self.dataset_dir/group/"Image"/study_id/f"{image_name}.npy"
        
        ct_image_raw = np.load(ct_image_path)
        slice = ct_normalize(ct_image_raw)

        # multi nodule in image -> merger multi mask
        # nodule_masks = [np.zeros_like(slice)]
        # for mask_name in nodule_masks_name:
        #     if "cvat" in image_name: # for data from cvat
        #         mask_path = mask_name
        #     elif "2.6" in image_name: # for data from kc version 2.6
        #         mask_path = mask_name
        #     else: # for data from the latest kc version 
        #         mask_path = f"{self.dataset_dir}/{group}/Mask/{study_id}/{mask_name}.npy"
        #     nodule_masks.append(np.load(mask_path))
        print(row["mask_name"])
        print(row["lung_damage_name"])
        nodule_masks = [np.zeros_like(slice)]
        nodule_masks.extend(get_masks(mask_folder=self.dataset_dir/group/"Mask"/study_id,
                                    mask_names=row["mask_name"]))
        nodule_masks.extend(get_masks(mask_folder=self.dataset_dir/group/"LungDamage"/study_id,
                                    mask_names=row["lung_damage_name"]))
        nodule_mask = np.logical_or.reduce(nodule_masks)

        datapoint = {
            "slice": slice.astype(np.float32),
            "slice_info": slice_info,
            "seg_nodule": nodule_mask.astype(np.float32),
        }

        # nodule and lung damage
        if self.use_nodule_cls or self.use_lung_damage_cls:
            bboxes = [row[col] for col in self.df_columns["bbox"]]
            bboxes = np.array(bboxes, dtype=np.float32).T
            labels = {}
            if self.use_lung_damage_cls:
                for key, col in self.df_columns["lung_damage"].items():
                    labels[key] = extend_label(create_label(label_ids=row[col], n_class=2))
            if self.use_nodule_cls:
                for key, group_cols in self.df_columns["nodule"].items():
                    nodule_label = []
                    for col in group_cols:
                        nodule_label.append(row[col])
                        if key == "voi_hoa":
                            nodule_label.append([-1 if value == -1 else 1 - value for value in row[col]])
                            break
                    labels[key] = extend_label(np.array(nodule_label, dtype=np.float32).T)
            if self.use_lung_damage_cls and self.use_nodule_cls:
                n_nodule = labels[next(iter(self.df_columns["nodule"]))].shape[0]
                n_lung_damage = labels[next(iter(self.df_columns["lung_damage"]))].shape[0]
                for key, group_cols in self.df_columns["nodule"].items():
                    n_class = 2 if key == "voi_hoa" else len(group_cols)
                    neg_label = extend_label(np.full((n_lung_damage, n_class), -1, dtype=np.float32))
                    labels[key] = np.concatenate((neg_label, labels[key]), axis=0)
                for key in self.df_columns["lung_damage"].keys():
                    neg_label = extend_label(np.full((n_nodule, 2), -1, dtype=np.float32))
                    labels[key] = np.concatenate((labels[key], neg_label), axis=0)
            
            nodules_infos = {
                "bbox": bboxes,
                "label": labels
            }
            datapoint["nodule_info"] = nodules_infos

        if self.use_lung_pos:
            # info about position of right lung and left lung
            lung_pos_labels = {}
            for col in self.df_columns["lung_pos"]:
                lung_pos_labels[col] = create_label(label_ids=[row[col]], n_class=2)[0]
                lung_pos_labels[col] = np.array(lung_pos_labels[col], dtype=np.float32)
            datapoint["cls_lung_pos"] = lung_pos_labels

        if self.use_lung_loc:
            lung_loc_masks = get_masks(mask_folder=self.dataset_dir/group/"LungLoc"/study_id,
                                    mask_names=row["lung_loc_name"])
            
            error = False
            lung_loc_mask = np.zeros((*slice.shape[:2], len(self.df_columns["lung_loc"])))
            for lung_loc_id, col in enumerate(self.df_columns["lung_loc"]):
                if sum(row[col]) > 1 or len(row[col]) > 5: 
                    error = True
                if 1 in row[col]:
                    lung_loc_mask[:, :, lung_loc_id] = lung_loc_masks[row[col].index(1)]
            background_mask = np.logical_not(np.any(lung_loc_mask, axis=-1))[..., None]
            datapoint["seg_lung_loc"] = np.concatenate([background_mask, lung_loc_mask], axis=-1).astype(np.float32)
            if error:
                error_df.append(row)

        return datapoint

if __name__ == "__main__":
    version = 4.3
    use_nodule_cls = True
    use_lung_loc = True
    use_lung_pos = True
    use_lung_damage_cls = True

    print(f"Data version {version}")
    dataset = KCSliceDataset(data_dir=f"data/kc_cancer_{version}", 
                            meta_file="test_meta_info_filter.csv", 
                            use_nodule_cls=use_nodule_cls,
                            use_lung_pos=use_lung_pos,
                            use_lung_loc=use_lung_loc,
                            use_lung_damage_cls=use_lung_damage_cls,
                            cache_data=False)
    print('Length Dataset:', len(dataset))
    # for id in range(len(dataset)):
    #     data_point = dataset[id]
    # dataset = KCSliceDataset(data_dir=f"data/kc_cancer_{version}", 
    #                         meta_file="val_meta_info_filter.csv", 
    #                         use_nodule_cls=use_nodule_cls,
    #                         use_lung_pos=use_lung_pos,
    #                         use_lung_loc=use_lung_loc,
    #                         use_lung_damage_cls=use_lung_damage_cls,
    #                         cache_data=False)
    # print('Length Dataset:', len(dataset))
    # for id in range(len(dataset)):
    #     data_point = dataset[id]
    dataset = KCSliceDataset(data_dir=f"data/kc_cancer_{version}", 
                            meta_file="train_meta_info_filter.csv", 
                            use_nodule_cls=use_nodule_cls,
                            use_lung_pos=use_lung_pos,
                            use_lung_loc=use_lung_loc,
                            use_lung_damage_cls=use_lung_damage_cls,
                            cache_data=False)
    print('Length Dataset:', len(dataset))
    # for id in range(len(dataset)):
    #     data_point = dataset[id]
    
    # error_df = pd.DataFrame(error_df)
    # error_df.to_csv("error.csv", index=False)

    id = random.randint(0, len(dataset) - 1)
    id  = 484
    datapoint = dataset[id]

    slice, slice_info, nodule_mask = datapoint["slice"], datapoint["slice_info"], datapoint["seg_nodule"]
    print(f"***** Slice Info: {slice_info}*****")
    print("Shape: {} --- Max: {:.1f} --- Min: {:.1f}".format(slice.shape, slice.max(), slice.min()))
    print(f"***** Nodule Segmentation: *****")
    print("Mask shape: {} --- Max: {:.1f} --- Min: {:.1f}".format(nodule_mask.shape, nodule_mask.max(), nodule_mask.min()))

    if use_nodule_cls or use_lung_damage_cls:
        nodules_infos = datapoint["nodule_info"]
        print("***** Nodule Classification: *****")
        print(datapoint["nodule_info"])

    if use_lung_pos:
        lung_pos_labels = datapoint["cls_lung_pos"]
        print("***** Lung Position Classification: *****")
        for key in lung_pos_labels:
            print(f"{key} shape: ", lung_pos_labels[key].shape)

    if use_lung_loc:
        lung_loc_mask = datapoint["seg_lung_loc"]
        print("***** Lung Location Segmentation: *****")
        print("Mask shape: {} --- Max: {:.1f} --- Min: {:.1f}".format(lung_loc_mask.shape, lung_loc_mask.max(), lung_loc_mask.min()))

    import cv2
    import matplotlib.pyplot as plt
    col_titles = ["Slices", "Nodule Mask"]
    if use_lung_loc:
        col_titles.append("Lung Location Mask")
    n_row , n_col= 1, len(col_titles)
    _, axes = plt.subplots(n_row, n_col, figsize=(15, 10))
    for i, title in enumerate(col_titles):
        axes[i].set_title(title, weight='bold')

    slice = np.stack([slice, slice, slice], axis=-1)
    nodule_contours, _ = find_bbox(nodule_mask)
    cv2.drawContours(slice, nodule_contours, -1, colors["nodule"], 1)
    
    nodule_mask = np.stack([nodule_mask, nodule_mask, nodule_mask], axis=-1)
    if use_nodule_cls or use_lung_damage_cls:
        for bbox in nodules_infos["bbox"]:
            bbox = [int(x) for x in bbox]
            cv2.rectangle(slice, (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            colors["bbox"], 1)
            cv2.rectangle(nodule_mask, (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            colors["bbox"], 1)
    axes[1].imshow(nodule_mask, cmap="gray")
    axes[1].axis("off")
    
    if use_lung_loc:
        slice ,colored_mask = draw_masks(slice, masks=lung_loc_mask, colors=colors["lung_loc"])
        axes[2].imshow(colored_mask)
        axes[2].axis("off")
    
    axes[0].text(-0.5, 0.25, slice_info, weight='bold', transform=axes[0].transAxes)
    axes[0].imshow(slice)
    axes[0].axis("off")
    # plt.show()
    plt.savefig("dataset.png")