# read /work/hpc/iai/hungnt/kc_cancer/data/LIDC/Meta/meta_info.csv

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Read meta_info.csv
meta_info = pd.read_csv('/work/hpc/iai/hungnt/kc_cancer/data/LIDC/Meta/meta_info.csv')

# count how many class in malignancy column and how many value in each class
malignancy = meta_info['malignancy']
malignancy = malignancy.dropna()
malignancy = malignancy.astype(int)
malignancy = malignancy.sort_values()
malignancy = malignancy.value_counts()
print(malignancy)

# count how many True, False in is_cancer column
is_cancer = meta_info['is_cancer']
is_cancer = is_cancer.dropna()
is_cancer = is_cancer.sort_values()
is_cancer = is_cancer.value_counts()
print(is_cancer)

# count how many True, False in is_clean column
is_clean = meta_info['is_clean']
is_clean = is_clean.dropna()
is_clean = is_clean.sort_values()
is_clean = is_clean.value_counts()
print(is_clean)
