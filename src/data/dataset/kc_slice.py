from typing import List, Literal
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def ct_normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val: return image
    return (image - min_val) / (max_val - min_val)

def extend_label(label):

    # negative
    if label[0] == -1:
        return [0]*3+[1] + [0]*7+[1] + [0]*2+[1] + [0]*4+[1] + [0]*3+[1]

    # positive
    else:
        return label[:3]+[0] + label[3:10]+[0] + label[10:12]+[0] + label[12:16]+[0] + label[16:19]+[0]

class KCSliceDataset(Dataset):

    dataset = "kc_cancer"

    def __init__(self, data_dir: str,
                meta_file: str,
                cache_data: bool = False) -> None:

        super().__init__()
        self.dataset_dir = os.path.join(data_dir, self.dataset)

        meta_path = os.path.join(self.dataset_dir, meta_file)
        self.dataFrame = pd.read_csv(meta_path)

        for col in self.dataFrame.columns[9:]:
            self.dataFrame[col] =  self.dataFrame[col].apply(lambda x: eval(x))

        self.cache_data = False
        if cache_data:
            self.data = []
            for index in range(self.__len__()):
                self.data.append(self.__getitem__(index))

            self.cache_data = cache_data

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):

        if self.cache_data:
            return self.data[index]

        row = row = self.dataFrame.iloc[index]
        code = row["code"]
        is_clean = row["is_clean"]
        study_id = row["image_name"][0:10]
        image_name = row["image_name"]
        mask_name = row["mask_name"]
        group = "nhom_benh" if "CTB" in code else "nhom_chung"

        # bbox = [[info[i] for info in row.iloc[-4:]] for i in range(len(row.iloc[-1]))]
        # label = [[info[i] for info in row.iloc[9:28]] for i in range(len(row.iloc[-1]))]
        bbox = [[info[i] for info in row.iloc[-4:]] for i in range(3)]
        label = [[info[i] for info in row.iloc[9:28]] for i in range(3)]
        label = [extend_label(l) for l in label]


        nodules_infos = {
            "bbox": np.array(bbox, dtype=np.float32),
            "label": np.array(label, dtype=np.float32),
            "image_name": image_name
        }

        if is_clean:
            study_folder = os.path.join(self.dataset_dir, f"{group}/Clean/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

            # filter_data -> clean
            if not os.path.exists(ct_image_path):
                ct_image_path = ct_image_path.replace("Clean", "")

        else:
            study_folder = os.path.join(self.dataset_dir, f"{group}/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")


        ct_image_raw = np.load(ct_image_path)
        ct_image_norm = ct_normalize(ct_image_raw)
        image = ct_image_norm

        if is_clean:
            mask = np.zeros_like(image)
        else:
            # multi nodule in image -> merger multi mask
            masks = []
            study_folder = os.path.join(self.dataset_dir, f"{group}/Mask", study_id)
            for i in range(int(mask_name[-1]) + 1):
                mask_path = os.path.join(study_folder, f"{mask_name[:-1]}{i}.npy")
                masks.append(np.load(mask_path))

            mask = np.sum(np.array(masks), axis=0)

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask, nodules_infos

if __name__ == "__main__":
    dataset = KCSliceDataset(data_dir="data/", 
                            meta_file="seg_cls_train_meta_info_2.5.csv", 
                            cache_data=False)
    print('Length Dataset:', len(dataset))

    # draw ['KC_CT_0021_IMAGE_SLICE_0001', 'KC_CT_0841_IMAGE_SLICE_0001', 'KC_CT_0253_IMAGE_SLICE_0006', 'KC_CT_0259_IMAGE_CLEAN_SLICE_0002', 'KC_CT_0831_IMAGE_SLICE_0004','KC_CT_0692_IMAGE_SLICE_0001', 'KC_CT_0329_IMAGE_CLEAN_SLICE_0006', 'KC_CT_0563_IMAGE_CLEAN_SLICE_0008'] and save jpg to debug/ folder
    import matplotlib.pyplot as plt
    import cv2
    import os
    error_images = ['KC_CT_0021_IMAGE_SLICE_0001', 'KC_CT_0841_IMAGE_SLICE_0001', 'KC_CT_0253_IMAGE_SLICE_0006', 'KC_CT_0259_IMAGE_CLEAN_SLICE_0002', 'KC_CT_0831_IMAGE_SLICE_0004','KC_CT_0692_IMAGE_SLICE_0001', 'KC_CT_0329_IMAGE_CLEAN_SLICE_0006', 'KC_CT_0563_IMAGE_CLEAN_SLICE_0008']
    for i in range(len(dataset)):
        image, mask, nodules_infos = dataset[i]
        image_name = nodules_infos['image_name']
        if image_name in error_images:
            print(image_name)
            print(image.shape, mask.shape)
            # save image to debug folder, knowing the shape is 512 x 512

            cv2.imwrite(f'debug/{image_name}_image.jpg', image*255)
            cv2.imwrite(f'debug/{image_name}_mask.jpg', mask*255)

            print(nodules_infos)
            print()
            # break