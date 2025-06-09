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

input_vars = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']

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
print(covariance.shape)
data_all = np.c_[dataset['LL (%)'], dataset['PI (%)'], dataset['LI'],
                 dataset['(qt-svo)/s¢vo'], dataset['(qt-u2)/s¢vo'],
                 dataset['(u2-u0)/s¢vo'], dataset['Bq']]

sigma_list = []
miu_list = []

for i, row in enumerate(data_all):
    mask = np.isnan(row)
    non_nan_indices = np.where(~mask)[0]
    nan_indices = np.where(mask)[0]

    data = row[non_nan_indices]
    input_covariance = covariance[np.ix_(non_nan_indices, non_nan_indices)]
    output_covariance = covariance[np.ix_(nan_indices, nan_indices)]
    co_covariance = covariance[np.ix_(nan_indices, non_nan_indices)]
    covariance_beta = co_covariance @ np.linalg.pinv(input_covariance)

    sigma_i = output_covariance - covariance_beta @ co_covariance.T
    miu_i = mius.to_numpy()[nan_indices] + covariance_beta @ (data - mius.to_numpy()[non_nan_indices])

    sigma_list.append(sigma_i)
    miu_list.append(miu_i)

    print(f"Row {i}: sigma shape = {sigma_i.shape}, miu shape = {miu_i.shape}")

np.save("sigma_list.npy", np.array(sigma_list, dtype=object))
np.save("miu_list.npy", np.array(miu_list, dtype=object))
