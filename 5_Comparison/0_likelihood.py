import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
tfd = tfp.distributions
tfb = tfp.bijectors
import tf_keras
import random
import os.path
import pathlib

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

pre_ori = pd.read_excel('pre_ori.xlsx').to_numpy()
pre_MN = pd.read_excel('pre_MN.xlsx').to_numpy()
pre_MICE = pd.read_excel('pre_MICE.xlsx').to_numpy()
pre_MF = pd.read_excel('pre_MF.xlsx').to_numpy()
target = pd.read_excel('ori_target.xlsx').to_numpy()

labels = np.unique(pre_MN[:, -1])

subsets_ori = {label: pre_ori[pre_ori[:, -1] == label] for label in labels}
subsets_MN = {label: pre_MN[pre_MN[:, -1] == label] for label in labels}
subsets_MICE = {label: pre_MICE[pre_MICE[:, -1] == label] for label in labels}
subsets_MF = {label: pre_MF[pre_MF[:, -1] == label] for label in labels}
subsets_target = {label: target[target[:, -1] == label] for label in labels}

datasets = {}
for label in labels:
    datasets[f'ori_{int(label)}'] = (subsets_ori[label], subsets_target[label])
    datasets[f'MN_{int(label)}'] = (subsets_MN[label], subsets_target[label])
    datasets[f'MICE_{int(label)}'] = (subsets_MICE[label], subsets_target[label])
    datasets[f'MF_{int(label)}'] = (subsets_MF[label], subsets_target[label])

for name, (data, target) in datasets.items():
    print(f"{name}: data shape = {data.shape}, target shape = {target.shape}")

def split_data(data, target):
    non_nan_index = ~np.isnan(target).any(axis=1)
    target_r = target[non_nan_index, :-1]
    data_t = data[non_nan_index]

    pre_train, pre_temp, r_train, r_temp = train_test_split(data_t, target_r, test_size=0.2, random_state=42)
    pre_val, pre_test, r_val, r_test = train_test_split(pre_temp, r_temp, test_size=0.5, random_state=42)
    return (data_t, target_r,
            pre_train, pre_val, pre_test,
            r_train, r_val, r_test)

def estimate_predictions(su_pre):
    log_su_pre = np.log(su_pre)

    miu = log_su_pre[:, 1]
    sigma = (log_su_pre[:, 2] - log_su_pre[:, 0]) / (2 * 1.96)

    predictions = tfd.Normal(
        loc=miu,
        scale=tf.sqrt(sigma)
    )

    return predictions

def predict(data, target):
    data_t, target_t, pre_train, pre_val, pre_test, r_train, r_val, r_test = split_data(data, target)

    predictions_train = estimate_predictions(pre_train)
    predictions_val = estimate_predictions(pre_val)
    predictions_test = estimate_predictions(pre_test)

    all_likelihood_train = predictions_train.prob(np.log(r_train).T[..., np.newaxis])
    all_likelihood_val = predictions_val.prob(np.log(r_val).T[..., np.newaxis])
    all_likelihood_test = predictions_test.prob(np.log(r_test).T[..., np.newaxis])

    mean_likelihood_train = -np.mean(np.log(np.mean(all_likelihood_train,axis = 0)))
    mean_likelihood_val = -np.mean(np.log(np.mean(all_likelihood_val,axis = 0)))
    mean_likelihood_test = -np.mean(np.log(np.mean(all_likelihood_test,axis = 0)))
    mean_likelihood_all = mean_likelihood_test*0.1 + mean_likelihood_val*0.1 + mean_likelihood_train*0.8
    # print(pre_test_r.shape)
    return {
        "train_r": r_train,
        "pre_train_r": pre_train[:,:-1],
        "mean_likelihood_train": mean_likelihood_train,
        "val_r": r_val,
        "pre_val_r": pre_val[:,:-1],
        "mean_likelihood_val": mean_likelihood_val,
        "test_r": r_test,
        "pre_test_r": pre_test[:,:-1],
        "mean_likelihood_test": mean_likelihood_test,
        "mean_likelihood_all": mean_likelihood_all
    }

likelihoods_dict = {}
predictions_mean_dict = {}
predictions_variance_dict = {}

results_predictions = []
results_likelihoods = []

for i, (name, (data, target)) in enumerate(datasets.items()):

    prediction_likelihood = predict(data, target)

    likelihoods = {
        "dataset": name,
        "mean_likelihood_train": prediction_likelihood["mean_likelihood_train"],
        "mean_likelihood_val": prediction_likelihood["mean_likelihood_val"],
        "mean_likelihood_test": prediction_likelihood["mean_likelihood_test"],
        "mean_likelihood_all": prediction_likelihood["mean_likelihood_all"],
    }

    pre_train_r = prediction_likelihood["pre_train_r"]
    pre_val_r = prediction_likelihood["pre_val_r"]
    pre_test_r = prediction_likelihood["pre_test_r"]
    df_train = pd.DataFrame(pre_train_r)
    df_train["train_r"] = prediction_likelihood["train_r"]
    df_val = pd.DataFrame(pre_val_r)
    df_val["val_r"] = prediction_likelihood["val_r"]
    df_test = pd.DataFrame(pre_test_r)
    df_test["test_r"] = prediction_likelihood["test_r"]

    with pd.ExcelWriter(f"{name}_predictions_MN.xlsx") as writer:
        df_train.to_excel(writer, sheet_name="pre_train_r", index=False)
        df_val.to_excel(writer, sheet_name="pre_val_r", index=False)
        df_test.to_excel(writer, sheet_name="pre_test_r", index=False)

    results_likelihoods.append(likelihoods)

df_results_likelihoods = pd.DataFrame(results_likelihoods)
df_results_likelihoods.to_excel( "mn_likelihood_results.xlsx", index=False)