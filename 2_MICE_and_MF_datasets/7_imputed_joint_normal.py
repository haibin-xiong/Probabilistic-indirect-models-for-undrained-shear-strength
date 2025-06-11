import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
ori_Rmatrix = np.loadtxt(folder_path.parent/'0_based_multinormal_model'/'Rmatrix.txt')
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

import pandas as pd
import numpy as np
from scipy.stats import norm
# 读取数据
mice_data = {}
mf_data = {}
for i in range(1, 5):
    mice_data[i] = pd.read_excel('mice_data.xlsx', sheet_name=f'Sheet_{i}')
    mf_data[i] = pd.read_excel('mf_data.xlsx', sheet_name=f'Sheet_{i}')

# 设定列名
features = mice_data[1].columns
for i in range(1, 5):
    mice_data[i].columns = features
    mf_data[i].columns = features

z = 0.7
miu_sigmas = []

# 初始化 dataset 存储
dataset_mice = {i: pd.DataFrame() for i in range(1, 5)}
dataset_mf = {i: pd.DataFrame() for i in range(1, 5)}

# 计算 miu 和 sigma
for var in features:

    for i in range(1, 5):

        data_r_t_mice = mice_data[i][var]
        data_r_t_mf = mf_data[i][var]

        # MICE 数据集
        y_posz_mice = np.percentile(data_r_t_mice, 100 * norm.cdf(z))
        y_negz_mice = np.percentile(data_r_t_mice, 100 * norm.cdf(-z))
        miu_mice = np.percentile(data_r_t_mice, 100 * norm.cdf(0))
        sigma_mice = 0.5 * (y_posz_mice - y_negz_mice)

        # MF 数据集
        y_posz_mf = np.percentile(data_r_t_mf, 100 * norm.cdf(z))
        y_negz_mf = np.percentile(data_r_t_mf, 100 * norm.cdf(-z))
        miu_mf = np.percentile(data_r_t_mf, 100 * norm.cdf(0))
        sigma_mf = 0.5 * (y_posz_mf - y_negz_mf)

        # 生成数据集
        dataset_mice[i][var] = data_r_t_mice
        dataset_mf[i][var] = data_r_t_mf

        miu_sigmas.append({
            'Sheet': i,
            'Variable': var,
            'miu_mice': miu_mice,
            'sigma_mice': sigma_mice,
            'miu_mf': miu_mf,
            'sigma_mf': sigma_mf
        })

for i in range(1, 5):

    miu_sigmas.append({
        'Sheet': i,
        'Variable': 'su(mob)/s¢v0',
        'miu_mice': -1.17292014948552,
        'sigma_mice': 0.524814710034928,
        'miu_mf': -1.17292014948552,
        'sigma_mf': 0.524814710034928
    })
# 转换 miu_sigmas 为 DataFrame
miu_sigmas_df = pd.DataFrame(miu_sigmas)

mice_Rmatrix = {}
mf_Rmatrix = {}

for i in range(1, 5):
    mice_Rmatrix[i] = np.loadtxt(f'mice_{i}_Rmatrix.txt')
    mf_Rmatrix[i] = np.loadtxt(f'mf_{i}_Rmatrix.txt')

datasets = {}

for i in range(1, 5):
    datasets[f'dataset_mice_{i}'] = (
        dataset_mice[i],  # MICE 数据
        mice_Rmatrix[i],  # MICE Rmatrix
        miu_sigmas_df[miu_sigmas_df['Sheet'] == i]['miu_mice'].values,  # MICE μ
        miu_sigmas_df[miu_sigmas_df['Sheet'] == i]['sigma_mice'].values  # MICE σ
    )

    datasets[f'dataset_mf_{i}'] = (
        dataset_mf[i],  # MF 数据
        mf_Rmatrix[i],  # MF Rmatrix
        miu_sigmas_df[miu_sigmas_df['Sheet'] == i]['miu_mf'].values,  # MF μ
        miu_sigmas_df[miu_sigmas_df['Sheet'] == i]['sigma_mf'].values  # MF σ
    )
# print(datasets)

results = {}

for dataset_name, (dataset, Rmatrix, miu, sigma) in datasets.items():

    covariance = np.diag(sigma) @ ori_Rmatrix @ np.diag(sigma)

    all_su = covariance
    all_su_beta = all_su[7, :7] @ np.linalg.inv(all_su[:7, :7])
    all_su_sigma = all_su[7, 7] - all_su_beta @ all_su[:7, 7]

    miu_all = miu[7] + all_su_beta @ (dataset.T - miu[:7][..., np.newaxis])
    print(miu_all.shape)
    print(np.isnan(miu_all).sum())
    mvn_all = tfd.Normal(
        loc=miu_all,
        scale=tf.fill(miu_all.shape, tf.sqrt(all_su_sigma))
    )
    predicted_all_su = mvn_all.sample(1000)
    all_su_r = np.exp(predicted_all_su)
    all_su_pre = np.percentile(all_su_r, [2.5, 50, 97.5], axis=0).T

    # save
    pd.DataFrame(all_su_pre).to_excel(f'{dataset_name}_all_su_pre.xlsx', index=False)