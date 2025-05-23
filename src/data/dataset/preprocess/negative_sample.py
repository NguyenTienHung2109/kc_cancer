import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
random.seed(12345)

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

def bbox_intersect(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2


    if x1_max < x2 or x2_max < x1:
        return False

    if y1_max < y2 or y2_max < y1:
        return False

    return True

def bboxes_intersect(bboxes, bbox):
    for b in bboxes:
        if bbox_intersect(b, bbox): return True

    return False

def negative_sampling(file_path: str):
    df = pd.read_csv(file_path)

    for col in df.columns[9:]:
        df[col] =  df[col].apply(lambda x: eval(x))

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        is_clean = row["is_clean"]

        pos_bboxes = []
        if not is_clean:
            # multi nodule in image -> merger multi mask
            lefts = df.loc[idx, "left"]
            tops = df.loc[idx, "top"]
            widths = df.loc[idx, "width"]
            heights = df.loc[idx, "height"]

            # Create a list of indices to remove
            indices_to_remove = [i for i in range(len(heights)) if lefts[i] == -1]
            for i in sorted(indices_to_remove, reverse=True):
                for col in df.columns[9:-4]:
                    df.loc[idx, col].pop(i)
                lefts.pop(i)
                tops.pop(i)
                widths.pop(i)
                heights.pop(i)


            for i in range(len(df["height"][idx])):
                pos_bboxes.append((df.loc[idx, "left"][i], df.loc[idx, "top"][i], df.loc[idx, "width"][i], df.loc[idx, "height"][i]))
        else:
            df.at[idx, "left"] = []
            df.at[idx, "top"] = []
            df.at[idx, "width"] = []
            df.at[idx, "height"] = []

        neg_bboxes = []
        min_w, max_w = 20, 128
        for _ in range(3 - len(pos_bboxes)):
            # sampling
            while True:
                w, scale = random.randint(min_w, max_w), random.uniform(0.5, 2)
                h = int(w * scale)
                x, y = random.randint(64, 448 - w), random.randint(128, 384 - h)

                if bboxes_intersect(pos_bboxes, bbox=(x, y, w, h)):
                    continue

                if bboxes_intersect(neg_bboxes, bbox=(x, y, w, h)):
                    continue

                if max_w > min_w + 5:
                    max_w -= 5

                break

            neg_bboxes.append((x, y, w, h))

        for bbox in neg_bboxes:
            for col in df.columns[9:-4]:
                df.loc[idx, col].append(-1)

            df.loc[idx, "left"].append(bbox[0])
            df.loc[idx, "top"].append(bbox[1])
            df.loc[idx, "width"].append(bbox[2])
            df.loc[idx, "height"].append(bbox[3])
                


    df.to_csv(file_path, index=False)

if __name__ == "__main__":

    version = 2.5
    print(f"Data version_{version}")

    negative_sampling(file_path = f"data/kc_cancer/seg_cls_train_meta_info_{version}.csv")
    negative_sampling(file_path = f"data/kc_cancer/seg_cls_val_meta_info_{version}.csv")
    negative_sampling(file_path = f"data/kc_cancer/seg_cls_test_meta_info_{version}.csv")
