import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
print(original_data.columns)
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

def miss_forest_impute(df, max_iter=10, random_state=42, tol=1e-4):

    df_imputed = df.copy()
    missing_mask = df.isnull()

    # 初始均值填补（保证后续训练数据完整）
    for col in df.columns:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    # 缓存原始缺失位置信息
    missing_cols = missing_mask.any()
    missing_cols = missing_cols[missing_cols].index.tolist()

    for iteration in range(max_iter):
        prev_imputed = df_imputed.copy()
        max_change = 0

        # 按缺失量升序处理（从易到难）
        processing_order = missing_mask.sum().sort_values().index

        for col in processing_order:
            if col not in missing_cols:
                continue

            # 获取当前列的已知/未知样本
            known_mask = ~missing_mask[col]
            unknown_mask = missing_mask[col]

            # 准备训练数据
            X_train = df_imputed.loc[known_mask, df.columns != col]
            y_train = df_imputed.loc[known_mask, col]
            X_test = df_imputed.loc[unknown_mask, df.columns != col]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # 训练随机森林（启用并行计算）
            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=100,
                n_jobs=-1,
                random_state=random_state
            )
            rf.fit(X_train, y_train)

            # 预测并更新数据
            imputed_values = rf.predict(X_test)
            current_change = np.abs(
                df_imputed.loc[unknown_mask, col] - imputed_values
            ).max()
            max_change = max(max_change, current_change)

            df_imputed.loc[unknown_mask, col] = imputed_values

        # 收敛判断（仅考虑缺失位置的变化）
        print(f"Iteration {iteration + 1}, Max change: {max_change:.6f}")
        if max_change < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

    return df_imputed

def sequential_fill(df, min_non_null=1, max_non_null=4):

    filled_datasets = []

    for n_non_null in range(min_non_null, max_non_null + 1):
        print(f"\nProcessing subset with >= {n_non_null} non-null values")

        # 子集筛选
        subset_mask = df.notna().sum(axis=1) >= n_non_null
        df_subset = df.loc[subset_mask].copy()

        # 执行填补
        try:
            mf_data = miss_forest_impute(df_subset)
            filled_datasets.append(mf_data)
        except Exception as e:
            print(f"Error processing subset {n_non_null}: {str(e)}")
            continue

    return filled_datasets

filled_datasets = sequential_fill(dataset)

with pd.ExcelWriter('mf_data.xlsx') as writer:
    for i, filled_df in enumerate(filled_datasets):
        filled_df.to_excel(writer, sheet_name=f'Sheet_{i + 1}', index=False)