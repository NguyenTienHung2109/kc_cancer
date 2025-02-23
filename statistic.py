import argparse
import pandas as pd

def find_combination(meta_path: str):
    df = pd.read_csv(meta_path)

    if "v3.2" in meta_path:
        df = df[df['is_lung_loc'] == 0]

    dam_do_columns = ["Nhóm 1 - Đậm độ - 1.1 Đặc", "Nhóm 1 - Đậm độ - 1.2 Bán đặc", "Nhóm 1 - Đậm độ - 1.3 Kính mờ",]
    voi_hoa_columns = ["Nhóm 2 - Đậm độ vôi - 2.1 Không có vôi", "Nhóm 2 - Đậm độ vôi - 2.2 Vôi trung tâm", "Nhóm 2 - Đậm độ vôi - 2.3 Vôi dạng lá", "Nhóm 2 - Đậm độ vôi - 2.4 Vôi lan toả", "Nhóm 2 - Đậm độ vôi - 2.5 Vôi dạng bắp", "Nhóm 2 - Đậm độ vôi - 2.6 Vôi lấm tấm", "Nhóm 2 - Đậm độ vôi - 2.7 Vôi lệch tâm"]
    chua_mo_columns = ["Nhóm 3 - Đậm độ mỡ - 3.1 Không chứa mỡ", "Nhóm 3 - Đậm độ mỡ - 3.2 Có chứa mỡ"]
    duong_vien_columns = ["Nhóm 4 - Bờ và Đường viền - 4.1 Tròn đều", "Nhóm 4 - Bờ và Đường viền - 4.2 Đa thuỳ", "Nhóm 4 - Bờ và Đường viền - 4.3 Bờ không đều", "Nhóm 4 - Bờ và Đường viền - 4.4 Tua gai"]
    tao_hang_columns = ["Nhóm 5 - Tạo hang - 5.1 Không có", "Nhóm 5 - Tạo hang - 5.2 Hang lành tính", "Nhóm 5 - Tạo hang - 5.3 Hang ác tính"]

    def print_combinations(grouped_df, columns, group_name):
        print(f"{group_name} Combination")
        grouped_df = df.groupby(columns)
        for _, group in grouped_df:
            print(f"{group.iloc[0][columns].tolist()} - Count: {len(group)}")

    print("=" * 50)
    print_combinations(df, dam_do_columns, "Đậm độ")
    print("=" * 50)
    print_combinations(df, voi_hoa_columns, "Vôi hóa")
    print("=" * 50)
    print_combinations(df, chua_mo_columns, "Chứa mỡ")
    print("=" * 50)
    print_combinations(df, duong_vien_columns, "Đường viền")
    print("=" * 50)
    print_combinations(df, tao_hang_columns, "Tạo hang")

    dam_do_target_values = [[1, 1, 0],
                            [0, 0, 0]]
    voi_hoa_target_values = [[1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]
    chua_mo_target_values = [[0, 0]]
    duong_vien_target_values = [[0, 0, 0, 0], 
                                [0, 0, 1, 1], 
                                [0, 1, 0, 1], 
                                [0, 1, 1, 0],
                                [1, 1, 0, 0]]
    tao_hang_target_values = [[0, 0, 0]]

    condition_dam_do = df[dam_do_columns].apply(lambda row: row.tolist() in dam_do_target_values, axis=1)
    condition_voi_hoa = df[voi_hoa_columns].apply(lambda row: row.tolist() in voi_hoa_target_values, axis=1)
    condition_chua_mo = df[chua_mo_columns].apply(lambda row: row.tolist() in chua_mo_target_values, axis=1)
    condition_duong_vien = df[duong_vien_columns].apply(lambda row: row.tolist() in duong_vien_target_values, axis=1)
    condition_tao_hang = df[tao_hang_columns].apply(lambda row: row.tolist() in tao_hang_target_values, axis=1)

    matching_rows = df[condition_dam_do | condition_voi_hoa | condition_chua_mo | condition_duong_vien | condition_tao_hang]

    matching_rows.to_csv("error.csv", index=False)

def check_data(meta_path: str):
    no_nodule_case = []
    df = pd.read_csv(meta_path)
    df_groupby = df.groupby("code")
    for _, df_code in df_groupby:
        if (df_code["is_clean"] != 1).sum() == 0:
            no_nodule_case.append(df_code.iloc[0]["bnid"])

    print("-" * 100)

    df_pred = pd.read_csv("nhom_benh.csv")
    df_groupby = df_pred.groupby("bnid")
    for _, df_code in df_groupby:
        if (df_code["is_clean"] != 1).sum() <= 4:
            if df_code.iloc[0]["bnid"] not in no_nodule_case:
                print(df_code.iloc[0]["bnid"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data based on version")
    parser.add_argument("--version", "-v", default="v3.1", type=str, help="Specify the version")
    args = parser.parse_args()

    version = args.version
    print(f"Data {version}")

    find_combination(meta_path=f"data/kc_cancer/nhom_benh/meta_info.csv")
    # check_data(meta_path=f"data/kc_cancer_{version}/nhom_benh/meta_info.csv")
