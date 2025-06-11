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
ext_data_t = pd.read_excel('ext_data_t.xlsx')
target = pd.read_excel('su_r.xlsx')
target = target.iloc[:, :-1]
print(target.shape)
ext_data_t = pd.concat([ext_data_t, target], axis=1)
ext_data_t = ext_data_t.to_numpy()
print(ext_data_t.shape)
def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

res = scipy.stats.bootstrap((ext_data_t[:,0], ext_data_t[:,1]), pearson_corr, vectorized=False, paired=True)

Nvariables = 8
Rmatrix_mean = np.zeros((Nvariables,Nvariables))
Rmatrix_samples = np.zeros((Nvariables,Nvariables,res.bootstrap_distribution.shape[0]))

for i in range(Nvariables):
    for j in range(i+1,Nvariables):
        data_selected = ext_data_t[:,[i,j]]
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

    if np.all(np.linalg.eig(R_temp)[0] > 0.0):
        R_sum = R_sum + R_temp
        N_samples = N_samples + 1
    N_trys = N_trys + 1

print(N_samples / N_trys)
result = R_sum / N_samples
print(result)
np.savetxt('MN_Rmatrix.txt',result)