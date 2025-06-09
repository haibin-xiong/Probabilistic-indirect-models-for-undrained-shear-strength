import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import os.path
import pathlib

# Define the folder path
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

def arcsigmoid(x):
    return np.log(x / (1 - x))
z = 0.7
samples = pd.read_excel('original_data.xlsx')
fitted_params = pd.read_excel('johnson_fit_results.xlsx')
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
dataset = pd.DataFrame()
miu_sigma_t = []
for i, row in fitted_params.iterrows():
    variable_name = row['Variable']

    data_r = samples[variable_name]
    index = ~np.isnan(data_r)
    data_r = data_r[index]
    if variable_name != 'su(mob)/s¢v0' and variable_name != 'Bq':
        data_r_t = np.arcsinh(data_r)
    elif variable_name == 'Bq':
        data_r_t = arcsigmoid(data_r / 1.2)
    else:
        data_r_t = np.log(data_r)

    y_posz = np.percentile(data_r_t, 100 * scipy.stats.norm.cdf(z))
    y_negz = np.percentile(data_r_t, 100 * scipy.stats.norm.cdf(-z))
    miu = np.percentile(data_r_t, 100 * scipy.stats.norm.cdf(0))
    sigma = 0.5 * (y_posz-y_negz)
    miu_sigma_t.append({
        'Variable': variable_name,
        'miu': miu,
        'sigma': sigma
    })
    data_r_t = (data_r_t - miu) / sigma
    dataset[variable_name] = data_r_t
miu_sigma_df = pd.DataFrame(miu_sigma_t)
miu_sigma_df.to_excel('miu_sigma_t.xlsx', index=False)
dataset = dataset.apply(pd.to_numeric, errors='coerce')

index = (~np.isnan(dataset.to_numpy())).all(axis=1)
print(index.shape)
dataset_clean = dataset[index]
print(dataset_clean.shape)

from pingouin import multivariate_normality
print(multivariate_normality(dataset_clean.values, alpha=.05))

def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

i = 0
j = 1

data_selected = dataset.values[:,[i,j]]
# print(data_selected)
index = (~np.isnan(data_selected)).all(axis=1)
data_selected_clean = data_selected[index]
# print(data_selected_clean.shape)
# np.savetxt('data.txt',data_selected_clean)
print(pearson_corr(data_selected_clean[:,0], data_selected_clean[:,1]))

res = scipy.stats.bootstrap((data_selected_clean[:,0], data_selected_clean[:,1]), pearson_corr, vectorized=False, paired=True)
print(res.confidence_interval)
print(res.confidence_interval.low)
print(res.bootstrap_distribution)
print(res.bootstrap_distribution.shape[0])

Nvariables = 8
Rmatrix_mean = np.zeros((Nvariables,Nvariables))
Rmatrix_samples = np.zeros((Nvariables,Nvariables,res.bootstrap_distribution.shape[0]))

for i in range(Nvariables):
    for j in range(i+1,Nvariables):
        data_selected = dataset.values[:,[i,j]]#to numpy and selected i and j column
        index = (~np.isnan(data_selected)).all(axis=1)
        data_selected_clean = data_selected[index]
        Rmatrix_mean[i,j] = pearson_corr(data_selected_clean[:,0], data_selected_clean[:,1])
        res = scipy.stats.bootstrap((data_selected_clean[:,0], data_selected_clean[:,1]), pearson_corr, vectorized=False, paired=True)
        Rmatrix_samples[i,j] = res.bootstrap_distribution

print(Rmatrix_samples.shape)

R_temp = np.ones((Nvariables,Nvariables))
R_sum = np.zeros((Nvariables,Nvariables))
N_samples = 0
N_trys = 0
while(N_samples <1000):
    index = np.random.randint(Rmatrix_samples.shape[2], size=(Nvariables,Nvariables), dtype=int)

    for i in range(Nvariables):
        for j in range(i+1,Nvariables):
            R_temp[i,j] = Rmatrix_samples[i,j,index[i,j]]
            R_temp[j,i] = R_temp[i,j]
    # print(R_temp)
    # print(np.linalg.eig(R_temp)[0])
    # print(np.all(np.linalg.eig(R_temp)[0] > 0.0))
    # if np.all(np.linalg.eig(R_temp).eigenvalues > 0.0):
    if np.all(np.linalg.eig(R_temp)[0] > 0.0):
        R_sum = R_sum + R_temp
        N_samples = N_samples + 1
    N_trys = N_trys + 1

print(N_samples / N_trys)
result = R_sum / N_samples
print(result)
np.savetxt('Rmatrix.txt',result)