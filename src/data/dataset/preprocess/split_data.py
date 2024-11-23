from typing import Literal
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
np.random.seed(12345)

test_code = ['CTB-331', 'CTB-332', 'CTB-334', 'CTB-336', 'CTB-337', 'CTB-345', 'CTB-347', 'CTB-348', 
             'CTB-349', 'CTB-350', 'CTB-354', 'CTB-361', 'CTB-367', 'CTB-371', 'CTB-372', 'CTB-373', 
             'CTB-374', 'CTB-376', 'CTB-388', 'CTB-457', 'CTB-489', 'CTB-491', 'CTB-493', 'CTB-502', 
             'CTB-505', 'CTB-506', 'CTB-509', 'CTB-517', 'CTB-237', 'CTB-200', 'CTB-9', 'CTB-224', 
             'CTB-84', 'CTB-193', 'CTB-279', 'CTB-128', 'CTB-534', 'CTB-535', 'CTB-537', 'CTB-551', 
             'CTB-596', 'CTB-660', 'CTB-118', 'CTB-699', 'CTB-325', 'CTB-76', 'CTB-71', 'CTB-111', 
             'CTB-50', 'CTB-87', 'CTB-244', 'CTB-69', 'CTB-90', 'CTB-83', 'CTB-129', 'CTB-229', 
             'CTB-48', 'CTB-56', 'CTB-122', 'CTB-23', 'CTB-153', 'CTB-127', 'CTB-276', 'CTB-74', 
             'CTB-278', 'CTB-281', 'CTB-62', 'CTB-284', 'CTB-528', 'CTB-533', 'CTB-671', 'CTB-183', 
             'CTB-288', 'CTB-292', 'CTB-688', 'CTB-137', 'CTB-33', 'CTB-97', 'CTBA-1', 'CTBA-2', 
             'CTBA-3', 'CTBA-4', 'CTBA-5', 'CTBA-6', 'CTBA-7', 'CTBA-8', 'CTBA-9', 'CTBA-10', 
             'CTBA-11', 'CTBA-12', 'CTBA-13', 'CTBA-14', 'CTBA-15', 'CTBA-16', 'CTBA-17', 'CTBA-18', 
             'CTBA-19', 'CTBA-20', 'CTBA-21', 'CTBA-22', 'CTBA-23', 'CTBA-24', 'CTBA-25', 'CTBA-26', 
             'CTBA-27', 'CTBA-28', 'CTBA-29', 'CTBA-30', 'CTBA-31', 'CTBA-32', 'CTBA-33', 'CTBA-34', 
             'CTBA-35', 'CTBA-36', 'CTBA-37', 'CTBA-38', 'CTBA-39', 'CTBA-40', 'CTBA-41', 'CTBA-42', 
             'CTBA-43', 'CTBA-44', 'CTBA-45', 'CTBA-46', 'CTBA-47', 'CTBA-48', 'CTBA-49', 'CTBA-50', 
             'CTBA-51', 'CTBA-52', 'CTBA-53', 'CTBA-54', 'CTBA-55', 'CTBA-56', 'CTBA-57', 'CTBA-58', 
             'CTBA-59', 'CTBA-60', 'CTBA-61', 'CTBA-62', 'CTBA-64', 'CTBA-63', 'CTBA-65', 'CTBA-66', 
             'CTBA-67', 'CTBA-68', 'CTBA-69', 'CTBA-70', 'CTBA-71', 'CTBA-72', 'CTBB-27', 'CTBB-30', 
             'CTBB-32', 'CTBB-86', 'CTBB-36', 'CTBB-37', 'CTBB-38', 'CTBB-92', 'CTBB-10', 'CTBB-44', 
             'CTBB-17', 'CTBB-18', 'CTBB-22', 'CTBB-28']

def split_data(meta_path: str, task = "segmentation", split_type: Literal["case", "slice"] = "case"):
    df = pd.read_csv(meta_path)

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    if task == "segmentation":

        if split_type == "case":
            df_groupby = df.groupby("code")
            df_codes = [df_code for _, df_code in df_groupby]
            np.random.shuffle(df_codes)

            for df_code in df_codes:
                
                groups = df_code.groupby("image_name")

                new_df = pd.DataFrame(columns=df.columns)

                for _, group in groups:
                    # if len(group) > 1:
                    group = pd.DataFrame([group.iloc[-1]])
                    new_df = pd.concat([new_df, group])

                if df_code["code"].iloc[0] in test_code:
                    test_df = pd.concat([test_df, new_df])
                elif len(val_df) * 10 < len(df):
                    val_df = pd.concat([val_df, new_df])
                else:
                    train_df = pd.concat([train_df, new_df])

        elif split_type == "slice":
            df_groupby = df.groupby("image_name")
            df_groups = [group for _, group in df_groupby]

            for df_group in df_groups:
                df_group = pd.DataFrame([group.iloc[-1]])

                if df_group["code"] in test_code:
                    test_df.loc[len(test_df)] = df_group
                elif len(val_df) * 10 < len(df):
                    val_df.loc[len(val_df)] = df_group
                else:
                    train_df.loc[len(train_df)] = df_group

    elif task == "classification_nodule":
        df = pd.concat((df[df["is_clean"]==1], df[df["is_segment"]==1]))
        if split_type == "case":
            df_groupby = df.groupby("code")
            df_codes = [df_code for _, df_code in df_groupby]
            np.random.shuffle(df_codes)

            for df_code in df_codes:
                if df_code["code"].iloc[0] in test_code:
                    test_df = pd.concat([test_df, df_code])
                elif len(val_df) * 10 < len(df):
                    val_df = pd.concat([val_df, df_code])
                else:
                    train_df = pd.concat([train_df, df_code])

        elif split_type == "slice":
            shuffled_df = df.sample(frac=1)

            for _, row in shuffled_df.iterrows():
                if row["code"] in test_code:
                    test_df.loc[len(test_df)] = row
                elif len(val_df) * 10 < len(df):
                    val_df.loc[len(val_df)] = row
                else:
                    train_df.loc[len(train_df)] = row

    elif task == "seg_cls":
        nodule_cls_columns = ["Nhóm 1 - Đậm độ - 1.1 Đặc", "Nhóm 1 - Đậm độ - 1.2 Bán đặc", "Nhóm 1 - Đậm độ - 1.3 Kính mờ", \
                            "Nhóm 2 - Đậm độ vôi - 2.1 Không có vôi", "Nhóm 2 - Đậm độ vôi - 2.2 Vôi trung tâm", "Nhóm 2 - Đậm độ vôi - 2.3 Vôi dạng lá", "Nhóm 2 - Đậm độ vôi - 2.4 Vôi lan toả", "Nhóm 2 - Đậm độ vôi - 2.5 Vôi dạng bắp", "Nhóm 2 - Đậm độ vôi - 2.6 Vôi lấm tấm", "Nhóm 2 - Đậm độ vôi - 2.7 Vôi lệch tâm", \
                            "Nhóm 3 - Đậm độ mỡ - 3.1 Không chứa mỡ", "Nhóm 3 - Đậm độ mỡ - 3.2 Có chứa mỡ", \
                            "Nhóm 4 - Bờ và Đường viền - 4.1 Tròn đều", "Nhóm 4 - Bờ và Đường viền - 4.2 Đa thuỳ", "Nhóm 4 - Bờ và Đường viền - 4.3 Bờ không đều", "Nhóm 4 - Bờ và Đường viền - 4.4 Tua gai", \
                            "Nhóm 5 - Tạo hang - 5.1 Không có", "Nhóm 5 - Tạo hang - 5.2 Hang lành tính", "Nhóm 5 - Tạo hang - 5.3 Hang ác tính"]
        bbox_columns = ["left", "top", "width", "height"]

        # only for case
        df_groupby = df.groupby("code")
        df_codes = [df_code for _, df_code in df_groupby]
        np.random.shuffle(df_codes)

        for df_code in tqdm(df_codes):
            groups = df_code.groupby("image_name")
            new_df = pd.DataFrame(columns=df_code.columns)

            for _, group in groups:
                last_row = group.iloc[-1].copy()
                last_row.update({col: group[col].tolist() for col in nodule_cls_columns + bbox_columns})
                new_df = pd.concat([new_df, last_row.to_frame().T])
    
            if len(test_df) * 10 < len(df):
                test_df = pd.concat([test_df, new_df])
            elif len(val_df) * 10 < len(df):
                val_df = pd.concat([val_df, new_df])
            else:
                train_df = pd.concat([train_df, new_df])

    else:
        raise NotImplementedError(f"Not implemented split_data for {task} task")

    return train_df, val_df, test_df

if __name__ == "__main__":
    data_dir = "data/kc_cancer"
    version = 2.5
    split_type = "case"

    print(f"Data version_{version}")

    # task = "segmentation"
    # meta_name = f"meta_info_{version}.csv"

    # task = "classification_nodule"
    # meta_name = f"segmentation_meta_info_{version}.csv"

    task = "segmentation"
    meta_name = f"meta_info_bbox_{version}.csv"

    train_df, val_df, test_df = split_data(meta_path=os.path.join(data_dir, f"nhom_benh/{meta_name}"), 
                                            task=task,
                                            split_type=split_type)
    # train_df_1, val_df_1, test_df_1 = split_data(meta_path=os.path.join(data_dir, f"nhom_chung/{meta_name}"), 
    #                                             task=task,
    #                                             split_type=split_type)

    # train_df = pd.concat([train_df, train_df_1])
    train_df.reset_index(inplace=True, drop=True)
    train_df.to_csv(os.path.join(data_dir, f"{task}_train_meta_info_{version}.csv"), index=False)

    # val_df = pd.concat([val_df, val_df_1])
    val_df.reset_index(inplace=True,  drop=True)
    val_df.to_csv(os.path.join(data_dir, f"{task}_val_meta_info_{version}.csv"), index=False)

    # test_df = pd.concat([test_df, test_df_1])
    test_df.reset_index(inplace=True,  drop=True)
    test_df.to_csv(os.path.join(data_dir, f"{task}_test_meta_info_{version}.csv"), index=False)
