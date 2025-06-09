import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib

# Define the folder path
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

samples = pd.read_excel('original_data.xlsx')
# print(samples.shape)
input_vars = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']

miu_sigma = pd.read_excel('miu_sigma_t.xlsx')
Rmatrix = np.loadtxt('Rmatrix.txt')
def arcsigmoid(x):
    return np.log(x / (1 - x))
dataset = pd.DataFrame()
for i, variable_name in enumerate(variables):

    data_r = samples[variable_name].copy()
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
# print(covariance)
data_all = np.c_[dataset['LL (%)'], dataset['PI (%)'], dataset['LI'],
                 dataset['(qt-svo)/s¢vo'], dataset['(qt-u2)/s¢vo'],
                 dataset['(u2-u0)/s¢vo'], dataset['Bq']]

miu = np.zeros((data_all.shape[0], 1))
sigma = np.zeros((data_all.shape[0], 1))
num_non_nan_set = np.zeros((data_all.shape[0], 1))

for i, row in enumerate(data_all):
    if np.all(np.isnan(row)):
        miu[i] = np.nan
        sigma[i] = np.nan
        continue
    non_nan_indices = np.where(~np.isnan(row))[0]
    num_non_nan = len(non_nan_indices)
    # print(i, non_nan_indices)
    # print(num_non_nan)
    data = row[non_nan_indices]
    # print(data)
    selected_indices = np.unique(np.append(non_nan_indices, 7))
    # print(selected_indices)
    covariance_selected = covariance[np.ix_(selected_indices, selected_indices)]
    # print(covariance_selected)
    covariance_selected_beta = covariance_selected[-1, :-1][np.newaxis, :] @ np.linalg.inv(covariance_selected[:-1, :-1])
    sigma[i] = covariance_selected[-1, -1] - covariance_selected_beta @ covariance_selected[:-1, -1]
    # print(covariance_selected_beta.shape)
    # print(mius.to_numpy()[non_nan_indices].T.shape)
    # print(sigma[i])
    miu[i] = mius.to_numpy()[-1] + covariance_selected_beta @ (data.T - mius.to_numpy()[non_nan_indices].T)
    # print(mius.to_numpy()[-1])
    num_non_nan_set[i] = num_non_nan
mvn = tfd.Normal(
    loc=miu,
    scale=tf.sqrt(sigma)
)

predicted = mvn.sample(1000)
su_pre = np.percentile(np.exp(predicted), [2.5, 50, 97.5], axis=0).T
# print(su_pre.shape)
# save
su_pre = np.squeeze(su_pre, axis=0)
su_pre = np.hstack((su_pre, num_non_nan_set))
# print(su_pre.shape)
all_su_pre = pd.DataFrame(su_pre)
all_su_pre.to_excel('su_pre.xlsx', index=False)