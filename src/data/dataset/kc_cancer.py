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

class KCCancerDataset(Dataset):

    dataset = "kc_cancer"

    def __init__(self, data_dir: str,
                meta_file: str,
                images: List[Literal["ct", "lung"]] = ["ct"],
                cache_data: bool = False) -> None:

        super().__init__()
        self.dataset_dir = os.path.join(data_dir, self.dataset)

        meta_path = os.path.join(self.dataset_dir, meta_file)
        self.dataFrame = pd.read_csv(meta_path)
        
        self.images = images

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

        if is_clean:
            study_folder = os.path.join(self.dataset_dir, f"{group}/Clean/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")

            # filter_data -> clean
            if not os.path.exists(ct_image_path):
                ct_image_path = ct_image_path.replace("Clean", "")
        else:
            study_folder = os.path.join(self.dataset_dir, f"{group}/Image", study_id)
            ct_image_path = os.path.join(study_folder, f"{image_name}.npy")


        image = []
        if "ct" in self.images:
            ct_image_raw = np.load(ct_image_path)
            ct_image_norm = ct_normalize(ct_image_raw)
            image.append(ct_image_norm)

        if "lung" in self.images:
            preprocess_path = ct_image_path.replace(".npy", "_preprocess.npy")
            lung_image_raw = np.load(preprocess_path)
            lung_image_norm = ct_normalize(lung_image_raw)
            image.append(lung_image_norm)

        if is_clean:
            mask = np.zeros_like(image[0])
        else:
            # multi nodule in image -> merger multi mask
            masks = []
            study_folder = os.path.join(self.dataset_dir, f"{group}/Mask", study_id)
            for i in range(int(mask_name[-1]) + 1):
                mask_path = os.path.join(study_folder, f"{mask_name[:-1]}{i}.npy")
                masks.append(np.load(mask_path))

            mask = np.sum(np.array(masks), axis=0)

        image = np.stack(image, axis=-1)
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask

if __name__ == "__main__":
    dataset = KCCancerDataset(data_dir="data/", 
                            meta_file="segmentation_train_meta_info_2.5.csv", 
                            images=["ct", "lung"],
                            cache_data=False)
    print('Length Dataset:', len(dataset))
    
    image, mask = dataset[0]

    print(image.shape, mask.shape)

    print(image.max(), image.min(), mask.max(), mask.min())

    ct_image, lung_image = np.split(image, 2, axis=-1)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(image[:,:,0], cmap='gray')
    axs[0].set_title('Image')
    axs[0].axis("off")

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis("off")

    axs[2].imshow(image[:,:,1], cmap='gray')
    axs[2].set_title('Preprocess')
    axs[2].axis("off")

    plt.show()