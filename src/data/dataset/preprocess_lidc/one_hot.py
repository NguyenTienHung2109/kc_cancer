import pandas as pd

# Define the file path
file_path = '/work/hpc/iai/hungnt/kc_cancer/data/LIDC/meta_info.csv'

# Read the CSV file
df = pd.read_csv(file_path)
cols = list(df.columns)
cols[5], cols[7] = cols[7], cols[5]
df = df[cols]

# Perform one-hot encoding for 'is_cancer' and 'is_clean'
encoded_data = pd.get_dummies(df, columns=['is_cancer', 'malignancy'])
# change column True, False to 1, 0
encoded_data['is_cancer_True'] = encoded_data['is_cancer_True'].astype(int)
encoded_data['is_cancer_False'] = encoded_data['is_cancer_False'].astype(int)
encoded_data['is_cancer_Ambiguous'] = encoded_data['is_cancer_Ambiguous'].astype(int)
encoded_data['malignancy_0'] = encoded_data['malignancy_0'].astype(int)
encoded_data['malignancy_1'] = encoded_data['malignancy_1'].astype(int)
encoded_data['malignancy_2'] = encoded_data['malignancy_2'].astype(int)
encoded_data['malignancy_3'] = encoded_data['malignancy_3'].astype(int)
encoded_data['malignancy_4'] = encoded_data['malignancy_4'].astype(int)
encoded_data['malignancy_5'] = encoded_data['malignancy_5'].astype(int)
encoded_data['is_clean'] = encoded_data['is_clean'].astype(int)

# Display the first few rows of the modified DataFrame
print(encoded_data.head())

# Optionally, save the one-hot encoded DataFrame to a new CSV file
output_path = '/work/hpc/iai/hungnt/kc_cancer/data/LIDC/meta_info_one_hot.csv'
encoded_data.to_csv(output_path, index=False)
