from typing import Literal
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
np.random.seed(12345)

test_code = [str(i).zfill(4) for i in range(900, 1013)]
val_code = [str(i).zfill(4) for i in range(800, 900)]

def split_data(meta_path: str, task = "segmentation"):
    df = pd.read_csv(meta_path)

    # Ensure patient_id is zero-padded in the DataFrame
    df["patient_id"] = df["patient_id"].apply(lambda x: str(x).zfill(4))

    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    if task == "segcls":
        # only for case
        df_groupby = df.groupby("patient_id")
        df_patients = [df_patient for _, df_patient in df_groupby]
        np.random.shuffle(df_patients)

        for df_patient in tqdm(df_patients):
            groups = df_patient.groupby("original_image")
            new_df = pd.DataFrame(columns=df_patient.columns)

            for _, group in groups:
                last_row = group.iloc[-1].copy()
                last_row.update({col: group[col].tolist() for col in group.columns[6:]})
                new_df = pd.concat([new_df, last_row.to_frame().T])

            if df_patient["patient_id"].iloc[0] in test_code:
                test_df = pd.concat([test_df, new_df])
            elif df_patient["patient_id"].iloc[0] in val_code:
                val_df = pd.concat([val_df, new_df])
            else:
                train_df = pd.concat([train_df, new_df])
        
    if task == "segmentation":
        # only for case
        df_groupby = df.groupby("patient_id")
        df_patients = [df_patient for _, df_patient in df_groupby]
        np.random.shuffle(df_patients)

        for df_patient in tqdm(df_patients):
            groups = df_patient.groupby("original_image")
            new_df = pd.DataFrame(columns=df_patient.columns)

            for _, group in groups:
                    group = pd.DataFrame([group.iloc[-1]])
                    new_df = pd.concat([new_df, group])

            if df_patient["patient_id"].iloc[0] in test_code:
                test_df = pd.concat([test_df, new_df])
            elif df_patient["patient_id"].iloc[0] in val_code:
                val_df = pd.concat([val_df, new_df])
            else:
                train_df = pd.concat([train_df, new_df])

    return train_df, val_df, test_df

if __name__ == "__main__":
    data_dir = "data/LIDC"
    
    task = "segmentation"

    meta_name = "meta_info_bbox_2.5.csv"

    train_df, val_df, test_df = split_data(meta_path=os.path.join(data_dir, meta_name),  task = task) 

    train_df.to_csv(os.path.join(data_dir, "{task}_train_meta_info_lidc.csv".format(task=task)), index=False)

    val_df.to_csv(os.path.join(data_dir, "{task}_val_meta_info_lidc.csv".format(task=task)), index=False)

    test_df.to_csv(os.path.join(data_dir, "{task}_test_meta_info_lidc.csv".format(task=task)), index=False)