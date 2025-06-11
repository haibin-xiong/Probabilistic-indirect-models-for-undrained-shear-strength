import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']
su = ['su(mob)/s¢v0']
all_su_pre = pd.read_excel(folder_path.parent / '0_based_multinormal_model' / 'su_pre.xlsx')
su_1 = all_su_pre[all_su_pre.iloc[:, -1] > 0]
su_2 = all_su_pre[all_su_pre.iloc[:, -1] > 1]
su_3 = all_su_pre[all_su_pre.iloc[:, -1] > 2]
su_4 = all_su_pre[all_su_pre.iloc[:, -1] > 3]

mice_data_1 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_1')
mice_data_1.columns = features
mice_data_2 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_2')
mice_data_2.columns = features
mice_data_3 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_3')
mice_data_3.columns = features
mice_data_4 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_4')
mice_data_4.columns = features

mf_data_1 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_1')
mf_data_1.columns = features
mf_data_2 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_2')
mf_data_2.columns = features
mf_data_3 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_3')
mf_data_3.columns = features
mf_data_4 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_4')
mf_data_4.columns = features

mice_data_1 = pd.concat([mice_data_1.reset_index(drop=True), su_1.reset_index(drop=True)], axis=1)
mf_data_1 = pd.concat([mf_data_1.reset_index(drop=True), su_1.reset_index(drop=True)], axis=1)
mice_data_2 = pd.concat([mice_data_2.reset_index(drop=True), su_2.reset_index(drop=True)], axis=1)
mf_data_2 = pd.concat([mf_data_2.reset_index(drop=True), su_2.reset_index(drop=True)], axis=1)
mice_data_3 = pd.concat([mice_data_3.reset_index(drop=True), su_3.reset_index(drop=True)], axis=1)
mf_data_3 = pd.concat([mf_data_3.reset_index(drop=True), su_3.reset_index(drop=True)], axis=1)
mice_data_4 = pd.concat([mice_data_4.reset_index(drop=True), su_4.reset_index(drop=True)], axis=1)
mf_data_4 = pd.concat([mf_data_4.reset_index(drop=True), su_4.reset_index(drop=True)], axis=1)
non_nan_count = mice_data_4.iloc[:, -1].notna().sum()
# print(non_nan_count)
# print(mice_data_1.shape)
# print(mice_data_2.shape)
# print(mice_data_3.shape)
# print(mice_data_4.shape)

def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

datasets = {
    'mice_1': mice_data_1, 'mf_1': mf_data_1,
    'mice_2': mice_data_2, 'mf_2': mf_data_2,
    'mice_3': mice_data_3, 'mf_3': mf_data_3,
    'mice_4': mice_data_4, 'mf_4': mf_data_4
}

for name, dataset in datasets.items():
    Nvariables = dataset.shape[1]
    Rmatrix_mean = np.zeros((Nvariables, Nvariables))
    Rmatrix_samples = np.empty((Nvariables, Nvariables, 10000))

    for i in range(Nvariables):
        for j in range(i + 1, Nvariables):
            data_selected = dataset.values[:, [i, j]]
            index = (~np.isnan(data_selected)).all(axis=1)
            data_selected_clean = data_selected[index]
            print(name,i,j,data_selected_clean[:, 0].shape)
            print(name,i,j,data_selected_clean[:, 1].shape)
            Rmatrix_mean[i, j] = pearson_corr(data_selected_clean[:, 0], data_selected_clean[:, 1])
            res = scipy.stats.bootstrap((data_selected_clean[:, 0], data_selected_clean[:, 1]),
                                         pearson_corr, vectorized=False, paired=True, n_resamples = 10000)
            Rmatrix_samples[i, j] = res.bootstrap_distribution

    R_temp = np.ones((Nvariables, Nvariables))
    R_sum = np.zeros((Nvariables, Nvariables))
    N_samples = 0
    N_trys = 0

    while N_samples < 10000:
        index = np.random.randint(Rmatrix_samples.shape[2], size=(Nvariables, Nvariables), dtype=int)
        for i in range(Nvariables):
            for j in range(i + 1, Nvariables):
                R_temp[i, j] = Rmatrix_samples[i, j, index[i, j]]
                R_temp[j, i] = R_temp[i, j]
        if np.all(np.linalg.eig(R_temp)[0] > 0.0):
            R_sum += R_temp
            N_samples += 1
        N_trys += 1

    result = R_sum / N_samples

    np.savetxt(f"{name}_Rmatrix.txt", result)