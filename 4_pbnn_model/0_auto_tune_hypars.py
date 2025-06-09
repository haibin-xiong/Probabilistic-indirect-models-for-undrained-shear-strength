import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import tensorflow as tf
import tensorflow_probability as tfp
import random
import tf_keras
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
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
def objective(trial, data, target):
    data_t, target_t, X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target)
    units = trial.suggest_int('units', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    activation_function = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh", "swish"])  # 优化激活函数
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    model = tf_keras.Sequential()
    model.add(tf_keras.Input(shape=(X_train.shape[1],)))
    model.add(tf_keras.layers.Dense(units, activation=activation_function))
    model.add(tf_keras.layers.Dropout(rate=dropout_rate))
    model.add(tf_keras.layers.Dense(1 + 1))
    model.add(tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                             scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))), )
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=2000,
              validation_data=(X_val, y_val), verbose=0)
    score = model.evaluate(X_val, y_val, verbose=0)
    return score

datasets = {}
for label in unique_labels:
    mask = all_su_pre.iloc[:, -1] == label

    datasets[f'label_{label}_ext'] = (ext_data_t.loc[mask], target.loc[mask])
    datasets[f'label_{label}_mice'] = (mice_data.loc[mask], target.loc[mask])
    datasets[f'label_{label}_mf'] = (mf_data.loc[mask], target.loc[mask])
for name, (data, target) in datasets.items():
    print(f"{name}: data shape = {data.shape}, target shape = {target.shape}")

best_params_list = []
for name, (data, target) in datasets.items():
    print(name)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, target), n_trials=50)

    best_params_list.append({
        'dataset_names': name,
        'units': study.best_params['units'],
        'dropout_rate': study.best_params['dropout_rate'],
        'learning_rate': study.best_params['learning_rate'],
        'activation_function': study.best_params['activation'],
        'batch_size': study.best_params['batch_size'],
        'best_score': study.best_value
    })

df_best_params = pd.DataFrame(best_params_list)
df_best_params.to_excel('best_params.xlsx', index=False)