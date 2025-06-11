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

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
miu_sigma_t = pd.read_excel('miu_sigmas_t.xlsx')
ori_Rmatrix = np.loadtxt(folder.parent/'0_based_multinormal_model'/'Rmatrix.txt')
MN_Rmatrix = np.loadtxt('MN_Rmatrix.txt')
ori_data_t = pd.read_excel('ori_data_t.xlsx', sheet_name="Sheet_1")
MN_data_t = pd.read_excel('ext_data_t.xlsx', sheet_name="Sheet_1")

ori_miu = miu_sigma_t['miu_ori']
ori_sigma = miu_sigma_t['sigma_ori']
MN_miu = miu_sigma_t['miu_ext']
MN_sigma = miu_sigma_t['sigma_ext']

ori_covariance = np.diag(ori_sigma.to_numpy()) @ ori_Rmatrix @ np.diag(ori_sigma.to_numpy())
MN_covariance = np.diag(MN_sigma.to_numpy()) @ MN_Rmatrix @ np.diag(MN_sigma.to_numpy())

ori_mius = np.zeros((ori_data_t.shape[0], 1))
ori_sigmas = np.zeros((ori_data_t.shape[0], 1))
num_non_nan_set = np.zeros((ori_data_t.shape[0], 1))
MN_mius = np.zeros((ori_data_t.shape[0], 1))
MN_sigmas = np.zeros((ori_data_t.shape[0], 1))

for i, row in enumerate(ori_data_t.values):
    if np.all(np.isnan(row)):
        ori_mius[i] = np.nan
        ori_sigmas[i] = np.nan
        continue
    non_nan_indices = np.where(~np.isnan(row))[0]
    num_non_nan = len(non_nan_indices)
    data = row[non_nan_indices]
    selected_indices = np.unique(np.append(non_nan_indices, 7))

    covariance_selected = ori_covariance[np.ix_(selected_indices, selected_indices)]
    covariance_selected_beta = covariance_selected[-1, :-1][np.newaxis, :] @ np.linalg.inv(covariance_selected[:-1, :-1])
    ori_sigmas[i] = covariance_selected[-1, -1] - covariance_selected_beta @ covariance_selected[:-1, -1]
    ori_mius[i] = ori_miu.to_numpy()[-1] + covariance_selected_beta @ (data.T - ori_miu.to_numpy()[non_nan_indices].T)
    num_non_nan_set[i] = num_non_nan
for i, row in enumerate(MN_data_t.values):
    non_nan_indices = np.where(~np.isnan(row))[0]
    data = row
    selected_indices = np.unique(np.append(non_nan_indices, 7))

    covariance_selected = ori_covariance[np.ix_(selected_indices, selected_indices)]
    covariance_selected_beta = covariance_selected[-1, :-1][np.newaxis, :] @ np.linalg.inv(covariance_selected[:-1, :-1])
    MN_sigmas[i] = covariance_selected[-1, -1] - covariance_selected_beta @ covariance_selected[:-1, -1]
    MN_mius[i] = MN_miu.to_numpy()[-1] + covariance_selected_beta @ (data.T - MN_miu.to_numpy()[non_nan_indices].T)

ori_mvn = tfd.Normal(
    loc=ori_mius,
    scale=tf.sqrt(ori_sigmas)
)
ori_predicted = ori_mvn.sample(1000)
su_pre_ori = np.percentile(np.exp(ori_predicted), [2.5, 50, 97.5], axis=0).T
su_pre_ori = np.squeeze(su_pre_ori, axis=0)
su_pre_ori = np.hstack((su_pre_ori, num_non_nan_set))
su_pre_ori= pd.DataFrame(su_pre_ori)

MN_mvn = tfd.Normal(
    loc=MN_mius,
    scale=tf.sqrt(MN_sigmas)
)
MN_predicted = MN_mvn.sample(1000)
su_pre_MN = np.percentile(np.exp(MN_predicted), [2.5, 50, 97.5], axis=0).T
su_pre_MN = np.squeeze(su_pre_MN, axis=0)
su_pre_MN = pd.DataFrame(su_pre_MN)

with pd.ExcelWriter("su_pre_results.xlsx") as writer:

    su_pre_ori.to_excel(writer, sheet_name="su_pre_ori", index=False)
    su_pre_MN.to_excel(writer, sheet_name="su_pre_MN", index=False)