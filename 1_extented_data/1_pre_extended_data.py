import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
import os
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
origin_data = pd.read_excel(folder.parent/'0_based_multinormal_model'/'original_data.xlsx')
all_su_pre = pd.read_excel(folder.parent/'0_based_multinormal_model'/'su_pre.xlsx')
labels = all_su_pre.iloc[:, -1]
labels_index = all_su_pre[(labels > 0) & (labels < 7)].index
input_vars = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
sigma_list = np.load("sigma_list.npy", allow_pickle=True)
miu_list = np.load("miu_list.npy", allow_pickle=True)
miu_sigma = pd.read_excel(folder.parent/'0_based_multinormal_model'/'miu_sigma_t.xlsx')
Rmatrix = np.loadtxt(folder.parent/'0_based_multinormal_model'/'Rmatrix.txt')
def arcsigmoid(x):
    return np.log(x / (1 - x))
dataset = pd.DataFrame()
for i, variable_name in enumerate(variables):

    data_r = origin_data[variable_name].copy()
    mask = np.isnan(data_r)

    if variable_name != 'su(mob)/s¢v0' and variable_name != 'Bq':
        data_r[~mask] = np.arcsinh(data_r[~mask])
    elif variable_name == 'Bq':
        data_r[~mask] = arcsigmoid(data_r[~mask] / 1.2)
    else:
        data_r[~mask] = np.log(data_r[~mask])
    dataset[variable_name] = data_r

mius = miu_sigma['miu']
sigmas = miu_sigma['sigma']
covariance = np.diag(sigmas.to_numpy()) @ Rmatrix @ np.diag(sigmas.to_numpy())
covariance = covariance[:-1, :-1]
data_all = np.c_[dataset['LL (%)'], dataset['PI (%)'], dataset['LI'],
                 dataset['(qt-svo)/s¢vo'], dataset['(qt-u2)/s¢vo'],
                 dataset['(u2-u0)/s¢vo'], dataset['Bq']]
labels_7 = all_su_pre[labels == 7].index
data = data_all[labels_7]
import numpy as np
import pandas as pd

# Assuming `data_all`, `all_su_pre`, `labels`, `labels_7`, and `miu_list` are defined

ext_data_list = []  # Store extended datasets
ori_data_list = []  # Store original datasets

for i in range(4):
    labels_index = all_su_pre[(labels > i) & (labels < 7)].index
    data_selected = data_all[labels_index]
    extended_data = np.zeros((data_selected.shape[0], 7))

    labels_ori = all_su_pre[labels > i].index
    ori_data_t = data_all[labels_ori]

    print(ori_data_t.shape)

    ext_data_t = np.full_like(ori_data_t, np.nan)  # Initialize with NaNs

    # Fill ext_data_t where labels match labels_7
    mask_7 = np.isin(labels_ori, labels_7)
    ext_data_t[mask_7] = data_all[labels_ori[mask_7]]

    # Fill extended_data with `miu_list` values
    for j, miu in enumerate(miu_list[labels_index]):
        mask = np.isnan(data_selected[j, :])
        non_nan_indices = np.where(~mask)[0]
        nan_indices = np.where(mask)[0]

        extended_data[j, nan_indices] = miu
        extended_data[j, non_nan_indices] = data_selected[j, non_nan_indices]

    # Fill ext_data_t where labels do not match labels_7
    mask_not_7 = ~np.isin(labels_ori, labels_7)
    ext_data_t[mask_not_7] = extended_data

    # Convert to DataFrame and store
    df_ext = pd.DataFrame(ext_data_t, index=labels_ori)
    df_ori = pd.DataFrame(ori_data_t, index=labels_ori)

    ext_data_list.append(df_ext)
    ori_data_list.append(df_ori)

# Save extended data to ext_data_t.xlsx
with pd.ExcelWriter("ext_data_t.xlsx") as writer:
    for i, df_ext in enumerate(ext_data_list):
        df_ext.to_excel(writer, sheet_name=f"Sheet_{i + 1}", index=False)

# Save original data to ori_data_t.xlsx
with pd.ExcelWriter("ori_data_t.xlsx") as writer:
    for i, df_ori in enumerate(ori_data_list):
        df_ori.to_excel(writer, sheet_name=f"Sheet_{i + 1}", index=False)

labels_tar = all_su_pre[labels > 3].index
target = origin_data['su(mob)/s¢v0'][labels_tar]
target = target.fillna(np.nan)
df_target = pd.DataFrame(target)
df_target['dummy'] = 1
df_target.to_excel('target.xlsx', index=False)