from typing import List
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class NoduleDataset(Dataset):
    
    dataset = "kc_cancer"
    nodule_folder = "Nodule_0.2_2.5_caranet"

    def __init__(self, data_dir: str, 
                 meta_file: str,
                 images: List[str] = ["nodule", "expand_nodule", "cropped_nodule"]) -> None:
        
        super().__init__()
        self.dataset_dir = os.path.join(data_dir, self.dataset)
        # print(self.dataset)
        meta_path = os.path.join(self.dataset_dir, meta_file)
        self.dataFrame = pd.read_csv(meta_path)
        self.images = images

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):
        row = self.dataFrame.iloc[index]
        code = row["code"]
        is_clean = row["is_clean"]
        study_id = row["image_name"][0:10]
        image_name = row["image_name"]
        mask_name = row["mask_name"]

        slice_id = row["slice_id"]
        nodule_id = row["nodule_id"]
        is_segment = row["is_segment"]

        group = "nhom_benh" if "CTB" in code else "nhom_chung"

        if not is_segment:
            label = {
                "dam_do": np.array([0]*3 + [1], dtype=np.float32),
                "voi_hoa": np.array([0]*7 + [1], dtype=np.float32),
                "chua_mo": np.array([0]*2 + [1], dtype=np.float32),
                "duong_vien": np.array([0]*4 + [1], dtype=np.float32),
                "tao_hang": np.array([0]*3 + [1], dtype=np.float32),
            }
        else:
            label = {
                "dam_do": np.array(list(row[9:12]) + [0], dtype=np.float32),
                "voi_hoa": np.array(list(row[12:19]) + [0], dtype=np.float32),
                "chua_mo": np.array(list(row[19:21]) + [0], dtype=np.float32),
                "duong_vien": np.array(list(row[21:25]) + [0], dtype=np.float32),
                "tao_hang": np.array(list(row[25:28]) + [0], dtype=np.float32),
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

        ct_image = np.load(ct_image_path).astype(np.float32)

        if is_clean:
            mask = np.zeros_like(ct_image)
        else:
            mask_path = os.path.join(study_folder.replace("Image", "Mask"), f"{mask_name}.npy")
            mask = np.load(mask_path).astype(np.float32)

        size = (512, 512)

        image_nodule = []
        study_folder = os.path.join(self.dataset_dir, 
                                    f"{group}/{self.nodule_folder}", 
                                    "Image" if is_segment else "Clean",
                                    study_id)
                    
        if "nodule" in self.images:
            nodule_path = os.path.join(study_folder, f"slice_{slice_id:04d}_nodule_{nodule_id:01d}.npy")
            nodule = np.load(nodule_path).astype(np.float32)
            nodule = cv2.resize(nodule, size)
            image_nodule.append(nodule)

        if "expand_nodule" in self.images:
            expand_nodule_path = os.path.join(study_folder, f"slice_{slice_id:04d}_expand_nodule_{nodule_id:01d}.npy")
            expand_nodule = np.load(expand_nodule_path).astype(np.float32)
            expand_nodule = cv2.resize(expand_nodule, size)
            image_nodule.append(expand_nodule)

        if "cropped_nodule" in self.images:
            cropped_nodule_path = os.path.join(study_folder, f"slice_{slice_id:04d}_cropped_nodule_{nodule_id:01d}.npy")
            cropped_nodule = np.load(cropped_nodule_path).astype(np.float32)
            cropped_nodule = cv2.resize(cropped_nodule, size)
            image_nodule.append(cropped_nodule)

        image_nodule = np.stack(image_nodule, axis=-1)

        return image_nodule, label, ct_image, mask

if __name__ == "__main__":
    dataset = NoduleDataset(data_dir="data/", 
                            meta_file="classification_nodule_val_meta_info_2.5.csv",
                            images=["nodule", "expand_nodule", "cropped_nodule"])
    print('Length Dataset:', len(dataset))
    
    image_nodule, label, image, mask = dataset[1000]

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0][0].imshow(image_nodule[:,:,0], cmap='gray')
    axs[0][0].set_title('Nodule')
    axs[0][0].axis("off")

    axs[0][1].imshow(image_nodule[:,:,1], cmap='gray')
    axs[0][1].set_title('Expand Nodule')
    axs[0][1].axis("off")

    axs[0][2].imshow(image_nodule[:,:,2], cmap='gray')
    axs[0][2].set_title('Cropped Nodule')
    axs[0][2].axis("off")

    axs[1][0].imshow(image, cmap='gray')
    axs[1][0].set_title('Image')
    axs[1][0].axis("off")

    axs[1][1].imshow(mask, cmap='gray')
    axs[1][1].set_title('Mask')
    axs[1][1].axis("off")

    plt.show()
    # save the plt
    plt.savefig("nodule_dataset.png")