import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(12345)

def label2string_classification(labels):
    classification_labels = {
        "dam_do": ['1.1 Dac', '1.2 Ban dac', '1.3 Kinh mo'],
        "voi_hoa": ['2.1 Khong co voi', '2.2 Voi trung tam', '2.3 Voi dang la', '2.4 Voi lan toa', '2.5 Voi dang bap', '2.6 Voi lam tam', '2.7 Voi lech tam'],
        "chua_mo": ['3.1 Khong chua mo', '3.2 Co chua mo'],
        "duong_vien": ['4.1 Tron deu', '4.2 Da thuy', '4.3 Bo khong deu', '4.4 Tua gai'],
        "tao_hang": ['5.1 Khong co', '5.2 Hang lanh tinh', '5.3 Hang ac tinh'],
    }

    label_str = []

    if labels["dam_do"] != [0, 0, 0]:
        id = np.argmax(labels["dam_do"])
        label_str.append(classification_labels["dam_do"][id])

    if labels["voi_hoa"] != [0, 0, 0, 0, 0, 0, 0]:
        id = np.argmax(labels["voi_hoa"])
        label_str.append(classification_labels["voi_hoa"][id])

    if labels["chua_mo"] != [0, 0]:
        id = np.argmax(labels["chua_mo"])
        label_str.append(classification_labels["chua_mo"][id])

    for i, value in enumerate(labels["duong_vien"]):
        if value == 1:
            label_str.append(classification_labels["duong_vien"][i])

    if labels["tao_hang"] != [0, 0, 0]:
        id = np.argmax(labels["tao_hang"])
        label_str.append(classification_labels["tao_hang"][id])

    return label_str

def find_bbox(pred: np.array):
    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, [cv2.boundingRect(contour) for contour in contours]

def ct_normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val: return image
    return (image - min_val) / (max_val - min_val)

def draw_labels(image, labels, position, nodule_id):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    font_thickness = 1
    text_color = (0, 0, 0)
    line_spacing = 20
    rectangle_color = (1, 1, 0)

    bbox_x = position[0]
    bbox_y = position[1]

    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + 20, bbox_y + 20), rectangle_color, thickness=-1)
    cv2.putText(image, str(nodule_id), (bbox_x + 5, bbox_y + 15), font, font_scale, text_color, 2)

    labels = [f"Ton Thuong {nodule_id}:"] + labels
    text_x, text_y = 520, (nodule_id - 1) * 180 + 50
    bbox_x, bbox_y = 512, text_y - 20
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + 220, bbox_y + 20 * (len(labels) + 1)), (1, 1, 1), thickness=-1)

    for label in labels:
        cv2.putText(image, f" + {label}" if label != labels[0] else label, (text_x, text_y), 
                    font, 
                    font_scale, 
                    text_color, 
                    font_thickness if label != labels[0] else font_thickness + 1)

        text_y += line_spacing

    return image

def check_data():
    df_nc = pd.read_csv("data/kc_cancer/nhom_chung/meta_info.csv")
    df_nc["group"] = "nc"
    df_nb = pd.read_csv("data/kc_cancer/nhom_benh/meta_info.csv")
    df_nb["group"] = "nb"
    df = pd.concat((df_nc, df_nb))

    group_columns = list(df.columns[9:-2])
    print(group_columns)

    grouped_df = df.groupby(group_columns).agg(lambda x: list(x))
    grouped_df = grouped_df.sort_values(by='code', key=lambda x: x.apply(len), ascending=False)
    grouped_df = grouped_df.reset_index()

    for i in range(len(grouped_df)):
        group_labels = grouped_df.iloc[i]

        n_slice = len(group_labels["code"])

        id = random.randrange(0, n_slice)

        code = group_labels["code"][id]
        is_clean = group_labels["is_clean"][id]
        study_id = group_labels["image_name"][id][0:10]
        image_name = group_labels["image_name"][id]
        mask_name = group_labels["mask_name"][id]
        group = "nhom_benh" if "CTB" in code else "nhom_chung"

        if is_clean:
            print('!' * 20, "Clean", '!' * 20)
            continue

        labels = {
            "dam_do": list(group_labels[0:3]),
            "voi_hoa": list(group_labels[3:10]),
            "chua_mo": list(group_labels[10:12]),
            "duong_vien": list(group_labels[12:16]),
            "tao_hang": list(group_labels[16:19]),
        }

        img_path = os.path.join(f"data/kc_cancer/{group}/Image", study_id, f"{image_name}.npy")
        mask_path = os.path.join(f"data/kc_cancer/{group}/Mask", study_id, f"{mask_name}.npy")

        raw_ct_image = np.load(img_path)
        norm_ct_image = ct_normalize(raw_ct_image)
        rgb_ct_image = np.stack([norm_ct_image, norm_ct_image, norm_ct_image], axis=-1)
        ct_image = np.concatenate([rgb_ct_image, np.zeros([512, 256, 3])] ,axis=1)
        mask = np.load(mask_path)

        print(group_labels)
        print(img_path)
        print("Length:", n_slice, '~' * 20)
        contours, bboxes = find_bbox((mask > 0.5).astype(np.uint8))
        
        if i < 97: continue

        for i, bbox in enumerate(bboxes):

            label_str = label2string_classification(labels)

            cv2.drawContours(ct_image, contours, i, (1, 1, 0), 2)

            ct_image = draw_labels(ct_image, label_str, 
                                    (bbox[0] + bbox[2], bbox[1] + bbox[3]), i + 1)

        plt.imshow(ct_image)
        plt.show()

        if list(group_labels[0:19]) == [0, 0, 0, 1] + [0] * 15 or \
            list(group_labels[0:19]) == [0] * 16 + [1, 0, 0] or \
            list(group_labels[0:19]) == [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 5 + [1, 0, 0] or \
            list(group_labels[0:19]) == [0] * 10 + [1] + [0] * 8 or \
            list(group_labels[0:19]) == [0, 0, 0, 1] + [0] * 12 + [1, 0, 0] or \
            list(group_labels[0:19]) == [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 8:

            print("+" * 20, "Start Check", "+" * 20)

            for i in range(5):
                id = random.randrange(0, n_slice)

                code = group_labels["code"][id]
                is_clean = group_labels["is_clean"][id]
                study_id = group_labels["image_name"][id][0:10]
                image_name = group_labels["image_name"][id]
                mask_name = group_labels["mask_name"][id]
                group = "nhom_benh" if "CTB" in code else "nhom_chung"

                labels = {
                    "dam_do": list(group_labels[0:3]),
                    "voi_hoa": list(group_labels[3:10]),
                    "chua_mo": list(group_labels[10:12]),
                    "duong_vien": list(group_labels[12:16]),
                    "tao_hang": list(group_labels[16:19]),
                }

                img_path = os.path.join(f"data/kc_cancer/{group}/Image", study_id, f"{image_name}.npy")
                mask_path = os.path.join(f"data/kc_cancer/{group}/Mask", study_id, f"{mask_name}.npy")

                raw_ct_image = np.load(img_path)
                norm_ct_image = ct_normalize(raw_ct_image)
                rgb_ct_image = np.stack([norm_ct_image, norm_ct_image, norm_ct_image], axis=-1)
                ct_image = np.concatenate([rgb_ct_image, np.zeros([512, 256, 3])] ,axis=1)
                mask = np.load(mask_path)

                contours, bboxes = find_bbox((mask > 0.5).astype(np.uint8))

                for i, bbox in enumerate(bboxes):

                    label_str = label2string_classification(labels)

                    cv2.drawContours(ct_image, contours, i, (1, 1, 0), 2)

                    ct_image = draw_labels(ct_image, label_str, 
                                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), i + 1)

                plt.imshow(ct_image)
                plt.show()

            print("+" * 20, "End Check", "+" * 20)

def filter(meta_path: str, version: float):
    df = pd.read_csv(meta_path)
    if version == 2.4:
        error_labels = [
            [0, 0, 0, 1] + [0] * 15,
            [0] * 16 + [1, 0, 0],
            [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 5 + [1, 0, 0],
            [0] * 10 + [1] + [0] * 8,
            [0, 0, 0, 1] + [0] * 12 + [1, 0, 0],
            [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 8,
            [0, 0, 1, 1] + [0] * 6 + [1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0] * 10 + [1] + [0] * 5 + [1, 0, 0],
            [0, 0, 1, 1] + [0] * 6 + [1] + [0] * 5 + [1, 0, 0],
            [1, 0, 0, 1] + [0] * 15,
        ]

        condition = df.apply(lambda row: list(row[9:28]) in error_labels, axis=1)
        new_df = df[~condition]

    elif version in [2.5, 2.6]:
        error_labels = [
            [0, 0, 0, 1] + [0] * 15,
            [0] * 16 + [1, 0, 0],
            [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 5 + [1, 0, 0],
            [0] * 10 + [1] + [0] * 8,
            [0, 0, 0, 1] + [0] * 12 + [1, 0, 0],
            [0, 0, 0, 1] + [0] * 6 + [1] + [0] * 8,
            [0, 0, 1, 1] + [0] * 6 + [1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0] * 10 + [1] + [0] * 5 + [1, 0, 0],
            [0, 0, 1, 1] + [0] * 6 + [1] + [0] * 5 + [1, 0, 0],
            [1, 0, 0, 1] + [0] * 15,
        ]

        condition = df.apply(lambda row: list(row[9:28]) in error_labels, axis=1)
        df.loc[condition, df.columns[4:7]] = [None, -1, 1]
        df.loc[condition, df.columns[9:]] = [-1] * 20

        def is_label_invalid(row):
            labels = {
                "dam_do": list(row[9:12]),
                "voi_hoa": list(row[12:19]),
                "chua_mo": list(row[19:21]),
                "duong_vien": list(row[21:25]),
                "tao_hang": list(row[25:28]),
            }

            # Kiểm tra nếu bất kỳ nhóm nào chứa toàn 0
            return any(all(val == 0 for val in label) for label in labels.values())

        # Xác định hàng lỗi
        condition = df.apply(is_label_invalid, axis=1)

        # Xử lý các hàng lỗi theo yêu cầu
        df.loc[condition, df.columns[4:7]] = [None, -1, 1]
        df.loc[condition, df.columns[9:]] = [-1] * 20

        # from IPython import embed
        # embed()

        new_df = df.drop_duplicates()

        if version == 2.6:
            print(df.columns[21:25])
            condition = new_df.apply(lambda row: row[21:25].sum() > 1 or row[21:25].sum() == 0, axis=1)
            new_df = new_df[~condition]

    print("Length before: {} ----- Length after: {} ----- n_drop: {}".format(len(df), len(new_df), len(df) - len(new_df)))

    new_df.to_csv(meta_path.replace("meta_info", f"meta_info_{version}"), index=False)

if __name__ == "__main__":
    # check_data()

    # 2.1 1458
    # 5.1 1206
    # 2.1 3.1 5.1 268
    # 3.1 131
    # 2.1 5.1 13
    # 2.1 3.1 8
    # 1.3 2.1 3.1 4.1 5.1 5
    # 3.1 5.1 3
    # 1.3 2.1 3.1 5.1 1
    # 1.1 2.1 1

    version = 2.5
    print(f"Data version_{version}")
    filter("data/kc_cancer/nhom_chung/meta_info.csv", version=version)
    filter("data/kc_cancer/nhom_benh/meta_info.csv", version=version)

