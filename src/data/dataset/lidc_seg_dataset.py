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

class LIDCSegDataset(Dataset):

    dataset = "LIDC"

    def __init__(self, data_dir: str,
                meta_file: str,
                cache_data: bool = False) -> None:

        super().__init__()
        self.dataset_dir = os.path.join(data_dir, self.dataset)

        meta_path = os.path.join(self.dataset_dir, meta_file)
        self.dataFrame = pd.read_csv(meta_path)
        
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

        if is_clean:
            
            study_folder = os.path.join(self.dataset_dir, "Clean/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

            # filter_data -> clean
            if not os.path.exists(ct_image_path):
                ct_image_path = ct_image_path.replace("Clean", "")

        else:
            
            study_folder = os.path.join(self.dataset_dir, "Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

        image = []
        ct_image_raw = np.load(ct_image_path)
        ct_image_norm = ct_normalize(ct_image_raw)
        image.append(ct_image_norm)

        if is_clean:
            mask = np.zeros_like(image[0])
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

        image = np.stack(image, axis=-1)
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask

if __name__ == "__main__":
    dataset = LIDCSegDataset(data_dir="data/", 
                            meta_file="segmentation_train_meta_info_lidc.csv", 
                            cache_data=False)
    print('Length Dataset:', len(dataset))
    
    image, mask = dataset[0]

    print(image.shape, mask.shape)

    print(image.max(), image.min(), mask.max(), mask.min())

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(image[:,:,0], cmap='gray')
    axs[0].set_title('Image')
    axs[0].axis("off")


    plt.show()