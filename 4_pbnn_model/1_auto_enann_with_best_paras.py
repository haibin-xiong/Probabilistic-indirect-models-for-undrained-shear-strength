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

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

all_su_pre = pd.read_excel(folder.parent / '0_based_multinormal_model' /'su_pre.xlsx')
unique_labels = all_su_pre.iloc[:, -1].unique()
unique_labels = [label for label in unique_labels if 1 <= label <= 7]
all_su_pre = all_su_pre[all_su_pre.iloc[:, -1].isin(unique_labels)].reset_index(drop=True)

ext_data_t = pd.read_excel(folder.parent / '1_extented_data' /'ext_data_t.xlsx', sheet_name=f'Sheet_1')
mice_data = pd.read_excel(folder.parent / '2_imputation' /'mice_data.xlsx', sheet_name=f'Sheet_1')
mf_data = pd.read_excel(folder.parent / '2_imputation' /'mf_data.xlsx', sheet_name=f'Sheet_1')
target = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_1')

ext_data_t.columns = features
mice_data.columns = features
mf_data.columns = features

def compute_sample_weights(y_true, y_pred):
    error = tf.abs(y_true - y_pred.mean())
    weights = tf.exp(error)
    return weights / tf.reduce_sum(weights)

def split_data(data, target):
    target_t = target.drop(columns=['dummy']).copy()
    target_t = np.log(target_t).dropna()
    non_nan_index = target.dropna().index
    data_t = data.loc[non_nan_index]
    # print(data_t.shape, target_t.shape)
    X_train, X_temp, y_train, y_temp = train_test_split(data_t, target_t, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return (data_t.to_numpy(), target_t.to_numpy(),
            X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(),
            y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy())


best_params_df = pd.read_excel('best_params.xlsx')

def calculate_pre_with_uncertaity(predictions):
    print('m', predictions.mean().numpy().shape)
    print('var', predictions.variance().numpy().shape)
    normal = tfd.Normal(
        loc=predictions.mean().numpy(),
        scale=predictions.variance().numpy() ** 0.5
    )
    predicted = normal.sample(1000)
    pre = np.percentile(predicted, [2.5, 50, 97.5], axis=0).T

    return pre

def predict_with_best_params(data, target, best_params, num_rounds=10):
    data_t, target_t, X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target)

    units = best_params['units']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    activation_function = best_params['activation_function']
    batch_size = best_params['batch_size']
    sample_weights = np.ones(y_train.shape[0]) / y_train.shape[0]

    for round in range(num_rounds):  # 进行 Boosting 训练多轮
        print(f"Boosting Round {round + 1}")
        model = tf_keras.Sequential()
        model.add(tf_keras.Input(shape=(X_train.shape[1],)))
        model.add(tf_keras.layers.Dense(units, activation=activation_function))
        model.add(tf_keras.layers.LayerNormalization())
        model.add(tf_keras.layers.MultiHeadAttention(num_heads=4, key_dim=units))
        model.add(tf_keras.layers.Dropout(rate=dropout_rate))
        model.add(tf_keras.layers.Dense(2))
        model.add(tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))), )
        negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=2000,
                  validation_data=(X_val, y_val), sample_weight=sample_weights, verbose=0)

        predictions_train = model(X_train)

        sample_weights = compute_sample_weights(y_train, predictions_train)

    predictions_train = model(X_train)
    # print('predictions_train', predictions_train)
    predictions_val = model(X_val)
    predictions_test = model(X_test)

    all_likelihood_train = predictions_train.tensor_distribution.prob(y_train.T[..., np.newaxis])
    all_likelihood_val = predictions_val.tensor_distribution.prob(y_val.T[..., np.newaxis])
    all_likelihood_test = predictions_test.tensor_distribution.prob(y_test.T[..., np.newaxis])
    # print(all_likelihood_train.shape)
    # print(all_likelihood_val.shape)
    # print(all_likelihood_test.shape)

    mean_likelihood_train = np.mean(np.log(np.mean(all_likelihood_train,axis = 0)))
    mean_likelihood_val = np.mean(np.log(np.mean(all_likelihood_val,axis = 0)))
    mean_likelihood_test = np.mean(np.log(np.mean(all_likelihood_test,axis = 0)))
    # print(mean_likelihood_train.shape)
    # print(mean_likelihood_val.shape)
    # print(mean_likelihood_test.shape)
    mean_likelihood_all = (mean_likelihood_test + mean_likelihood_val + mean_likelihood_train)/3

    pre_train = calculate_pre_with_uncertaity(predictions_train)
    pre_val = calculate_pre_with_uncertaity(predictions_val)
    pre_test = calculate_pre_with_uncertaity(predictions_test)
    # print(pre_test.shape)
    pre_train = np.squeeze(pre_train)
    pre_val = np.squeeze(pre_val)
    pre_test = np.squeeze(pre_test)
    # print(pre_test.shape)
    pre_train_r = np.exp(pre_train)
    pre_val_r = np.exp(pre_val)
    pre_test_r = np.exp(pre_test)

    train_r, val_r, test_r = np.exp(y_train), np.exp(y_val), np.exp(y_test)
    # print(pre_test_r.shape)
    return {
        "train_r": train_r,
        "pre_train_r": pre_train_r,
        "mean_likelihood_train": mean_likelihood_train,
        "val_r": val_r,
        "pre_val_r": pre_val_r,
        "mean_likelihood_val": mean_likelihood_val,
        "test_r": test_r,
        "pre_test_r": pre_test_r,
        "mean_likelihood_test": mean_likelihood_test,
        "mean_likelihood_all": mean_likelihood_all
    }

likelihoods_dict = {}
predictions_mean_dict = {}
predictions_variance_dict = {}
datasets = {}
for label in unique_labels:
    mask = all_su_pre.iloc[:, -1] == label

    datasets[f'label_{label}_ext'] = (ext_data_t.loc[mask], target.loc[mask])
    datasets[f'label_{label}_mice'] = (mice_data.loc[mask], target.loc[mask])
    datasets[f'label_{label}_mf'] = (mf_data.loc[mask], target.loc[mask])
for name, (data, target) in datasets.items():
    print(f"{name}: data shape = {data.shape}, target shape = {target.shape}")

results_predictions = []
results_likelihoods = []

for i, (name, (data, target)) in enumerate(datasets.items()):
    best_params = best_params_df.iloc[i]
    prediction_likelihood = predict_with_best_params(data, target, best_params)

    likelihoods = {
        "dataset": name,
        "mean_likelihood_train": prediction_likelihood["mean_likelihood_train"],
        "mean_likelihood_val": prediction_likelihood["mean_likelihood_val"],
        "mean_likelihood_test": prediction_likelihood["mean_likelihood_test"],
        "mean_likelihood_all": prediction_likelihood["mean_likelihood_all"],
    }
    print("pre_train_r", prediction_likelihood["pre_train_r"].shape)

    pre_train_r = prediction_likelihood["pre_train_r"]
    pre_val_r = prediction_likelihood["pre_val_r"]
    pre_test_r = prediction_likelihood["pre_test_r"]
    df_train = pd.DataFrame(pre_train_r)
    df_train["train_r"] = prediction_likelihood["train_r"]
    df_val = pd.DataFrame(pre_val_r)
    df_val["val_r"] = prediction_likelihood["val_r"]
    df_test = pd.DataFrame(pre_test_r)
    df_test["test_r"] = prediction_likelihood["test_r"]

    with pd.ExcelWriter(f"{name}_predictions.xlsx") as writer:
        df_train.to_excel(writer, sheet_name="pre_train_r", index=False)
        df_val.to_excel(writer, sheet_name="pre_val_r", index=False)
        df_test.to_excel(writer, sheet_name="pre_test_r", index=False)

    results_likelihoods.append(likelihoods)

df_results_likelihoods = pd.DataFrame(results_likelihoods)
df_results_likelihoods.to_excel( "likelihood_results.xlsx", index=False)