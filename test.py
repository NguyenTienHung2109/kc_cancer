file_path = "data/kc_cancer_4.4/nhom_chung/Image/KC_CT_0001/KC_CT_0001_IMAGE_SLICE_0001.npy"
import numpy as np

# Load dữ liệu
try:
    data = np.load(file_path)
    print(f"Kích thước của file: {data.shape}")
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")