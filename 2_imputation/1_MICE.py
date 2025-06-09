import os
import pathlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData
np.random.seed(42)
import random
random.seed(42)

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

rename_mapping = {
    'LL (%)': 'T1',
    'PI (%)': 'T2',
    'LI': 'T3',
    '(qt-svo)/s¢vo': 'T4',
    '(qt-u2)/s¢vo': 'T5',
    '(u2-u0)/s¢vo': 'T6',
    'Bq': 'T7',
}

variables_latex = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$']

original_data = pd.read_excel(folder.parent / '0_based_multinormal_model' / 'original_data.xlsx')
original_data = original_data[list(rename_mapping.keys())].copy()

def arcsigmoid(x):
    return np.log(x / (1 - x))

dataset = pd.DataFrame()
for i, variable_name in enumerate(original_data.columns):

    data_r = original_data[variable_name].copy()
    mask = np.isnan(data_r)

    if variable_name != 'su(mob)/s¢v0' and variable_name != 'Bq':
        data_r[~mask] = np.arcsinh(data_r[~mask])
    elif variable_name == 'Bq':
        data_r[~mask] = arcsigmoid(data_r[~mask] / 1.2)
    else:
        data_r[~mask] = np.log(data_r[~mask])
    dataset[variable_name] = data_r
dataset.rename(columns=rename_mapping, inplace=True)
non_nan_counts = dataset.notna().sum(axis=1)
count_of_counts = non_nan_counts.value_counts().sort_index()
print(non_nan_counts)
print(count_of_counts)

def sequential_fill(df):
    filled_datasets = []
    n_iterations = 10  # 迭代次数

    for n_non_null in range(1, 5):
        df_copy = df.dropna(thresh=n_non_null).copy(deep=True)  # 深拷贝，确保不修改原始数据
        print(df_copy.shape)

        mice_data = MICEData(df_copy)
        for i in range(n_iterations):
            mice_data.update_all()  # 迭代填补

        filled_datasets.append(mice_data.next_sample().copy(deep=True))  # 结果也做深拷贝

    return filled_datasets


filled_datasets = sequential_fill(dataset)

with pd.ExcelWriter('mice_data.xlsx') as writer:
    for i, filled_df in enumerate(filled_datasets):
        filled_df.to_excel(writer, sheet_name=f'Sheet_{i + 1}', index=False)
