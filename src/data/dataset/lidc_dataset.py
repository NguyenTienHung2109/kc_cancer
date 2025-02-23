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
        return [0]*3+[1] + [0]*6+[1]

    # positive
    else:
        return label[:3]+[0] + label[3:9]+[0]

class LIDCDataset(Dataset):

    dataset = "LIDC"

    def __init__(self, data_dir: str,
                meta_file: str,
                cache_data: bool = False) -> None:

        super().__init__()
        self.dataset_dir = os.path.join(data_dir, self.dataset)

        meta_path = os.path.join(self.dataset_dir, meta_file)
        self.dataFrame = pd.read_csv(meta_path)

        for col in self.dataFrame.columns[6:]:
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
        is_clean = row["is_clean"]
        study_id = "LIDC-IDRI-" + row["original_image"][0:4]
        nodule = int(row["original_image"][-3:])
        image_name = row["original_image"]
        mask_name = row["mask_image"]

        # bbox = [[info[i] for info in row.iloc[-4:]] for i in range(len(row.iloc[-1]))]
        # label = [[info[i] for info in row.iloc[9:28]] for i in range(len(row.iloc[-1]))]
        bbox = [[info[i] for info in row.iloc[-4:]] for i in range(3)]
        label = [[info[i] for info in row.iloc[6:-4]] for i in range(3)]
        label = [extend_label(l) for l in label]
        
        nodules_infos = {
            "bbox": np.array(bbox, dtype=np.float32),
            "label": np.array(label, dtype=np.float32)
        }


        if is_clean:
            
            study_folder = os.path.join(self.dataset_dir, "Clean/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

            # filter_data -> clean
            if not os.path.exists(ct_image_path):
                ct_image_path = ct_image_path.replace("Clean", "")

        else:
            
            study_folder = os.path.join(self.dataset_dir, "Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

        ct_image_raw = np.load(ct_image_path)
        ct_image_norm = ct_normalize(ct_image_raw)
        image = ct_image_norm

        if is_clean:
            mask = np.zeros_like(image)
        else:
            # multi nodule in image -> merger multi mask
            masks = []
            study_folder = os.path.join(self.dataset_dir, "Mask", study_id)
            
            for i in range(nodule + 1):
                mask_path = os.path.join(study_folder, f"{mask_name[:-3]}{str(i).zfill(3)}.npy")
                # check if mask exists
                if not os.path.exists(mask_path):
                    continue
                masks.append(np.load(mask_path))

            mask = np.sum(np.array(masks), axis=0)
            # if a pixel has more than 1 nodule, set it to 1
            mask[mask > 1] = 1

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask, nodules_infos

if __name__ == "__main__":
    dataset = LIDCDataset(data_dir="data/", 
                            meta_file="train_meta_info_lidc.csv", 
                            cache_data=False)
    print('Length Dataset:', len(dataset))

    image, mask, nodules_infos = dataset[101]

    print(image.shape, mask.shape)
    print(image.max(), image.min(), mask.max(), mask.min())
    print(nodules_infos['bbox'].shape)

    import cv2
    image = np.stack([image, image, image], axis=-1)
    mask = np.stack([mask, mask, mask], axis=-1)
    for bbox in nodules_infos["bbox"]:
        bbox = [int(x) for x in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]),
                        (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                        (0, 1, 0), 1)
        cv2.rectangle(mask, (bbox[0], bbox[1]),
                        (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                        (0, 1, 0), 1)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis("off")
    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[1].axis("off")
    plt.show()