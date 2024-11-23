import os
import pandas as pd

def merge_df(folder_path):

    cnt  = 0
    merged_df = []
    for file in os.listdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path, file))

        if len(df) < 100:
            print(file, len(df))
            continue

        merged_df.append(df)

        if cnt > 200:
            break

        cnt += 1

    merged_df = pd.concat(merged_df)

    folder_parts = folder_path.split('/')
    file_path = '/'.join(folder_parts[:-2]) + f"/{folder_parts[-2]}.csv"
    merged_df.to_csv(file_path, index=False)

if __name__ == "__main__":
    merge_df(folder_path="data/kc_cancer_v3/segcls_inference_3.1/nhom_benh/csv")
    print("-" * 100)
    # merge_df(folder_path="data/kc_cancer_v3/segcls_inference_3.1/nhom_chung")

