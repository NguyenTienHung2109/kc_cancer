import os
import cv2
import numpy as np
import pandas as pd

def merge_bbox(bboxes, image):
    left = min([bbox[0] for bbox in bboxes])
    top = min([bbox[1] for bbox in bboxes])
    right = max([bbox[0] + bbox[2] for bbox in bboxes])
    bottom = max([bbox[1] + bbox[3] for bbox in bboxes])

    merged_bbox = (left, top, right - left, bottom - top)
    return merged_bbox

def find_bbox(image: np.array):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    bbox = merge_bbox(bboxes, image)
    return bbox

def main(meta_path: str, dataset_dir: str, version: str):
    df = pd.read_csv(meta_path)

    bbox_infos = [ [] for _ in range(4)]
    for _, row in df.iterrows():
        is_clean = row["is_clean"]

        if is_clean:
            for i in range(4):
                bbox_infos[i].append(-1)
        else:
            code = row["code"]
            study_id = row["image_name"][0:10]
            mask_name = row["mask_name"]
            group = "nhom_benh" if "CTB" in code else "nhom_chung"

            study_folder = os.path.join(dataset_dir, f"{group}/Mask", study_id)
            mask_path = os.path.join(study_folder, f"{mask_name}.npy")
            mask = np.load(mask_path)

            bbox = find_bbox(mask)
            for i in range(4):
                bbox_infos[i].append(bbox[i])

    df["left"] = bbox_infos[0]
    df["top"] = bbox_infos[1]
    df["width"] = bbox_infos[2]
    df["height"] = bbox_infos[3]

    df.to_csv(meta_path.replace("meta_info", f"meta_info_bbox_{version}"), index=False)

if __name__ == "__main__":

    version = 2.5
    print(f"Data version_{version}")
    # main(f"data/kc_cancer_v3/nhom_chung/meta_info_{version}.csv", dataset_dir="data/kc_cancer")
    main(f"data/kc_cancer/nhom_benh/meta_info.csv", dataset_dir="data/kc_cancer", version=2.5)
