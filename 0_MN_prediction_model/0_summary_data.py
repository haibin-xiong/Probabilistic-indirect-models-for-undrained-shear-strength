import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_origin = pd.read_excel('CLAY_10_7490_TC304.xlsx')
data_origin.iloc[1:, 2:] = data_origin.iloc[1:, 2:].map(lambda x: x if isinstance(x, (int, float)) else np.nan)
print(data_origin.shape)
data_origin['(qt-svo)/s¢vo'] = (data_origin['qt (kN/m2)'] - data_origin['svo (kN/m2)']) / data_origin['s¢vo (kN/m2)']
data_origin['(qt-u2)/s¢vo'] = (data_origin['qt (kN/m2)'] - data_origin['u2 (kN/m2)']) / data_origin['s¢vo (kN/m2)']
data_origin['(u2-u0)/s¢vo'] = (data_origin['u2 (kN/m2)'] - data_origin['u0 (kN/m2)']) / data_origin['s¢vo (kN/m2)']

variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']

non_nan_counts = data_origin.notna().sum()
print(non_nan_counts)

summary_data = []

for i in range(44):
    var = data_origin.columns[i]
    data_origin[var] = pd.to_numeric(data_origin[var], errors='coerce')
    non_nan_count = data_origin[var].notna().sum()

    mean_value = data_origin[var].mean(skipna=True)
    std_dev = data_origin[var].std(skipna=True)
    cov = std_dev / mean_value if mean_value != 0 else None
    min_value = data_origin[var].min(skipna=True)
    max_value = data_origin[var].max(skipna=True)

    summary_data.append({
        'Variable': var,
        'Mean': mean_value,
        'COV': cov,
        'Min': min_value,
        'Max': max_value,
        'Points': non_nan_count
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_excel("summary_data.xlsx", index=False)
data_all_df = pd.DataFrame(data_origin)
data_all_cleaned = data_all_df.dropna(subset=variables, how='all')
data_all_cleaned.to_excel("original_data.xlsx", index=False)
print(data_all_cleaned.shape[0])