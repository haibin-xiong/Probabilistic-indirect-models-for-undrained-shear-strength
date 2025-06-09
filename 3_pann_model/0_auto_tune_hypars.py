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

ext_data_t = {}
mice_data = {}
mf_data = {}
target = {}
for i in range(1, 5):
    ext_data_t[i] = pd.read_excel(folder.parent / '1_extented_data' /'ext_data_t.xlsx', sheet_name=f'Sheet_{i}')
    mice_data[i] = pd.read_excel(folder.parent / '2_imputation' /'mice_data.xlsx', sheet_name=f'Sheet_{i}')
    mf_data[i] = pd.read_excel(folder.parent / '2_imputation' /'mf_data.xlsx', sheet_name=f'Sheet_{i}')
    target[i] = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_{i}')

for i in range(1, 5):
    ext_data_t[i].columns = features
    mice_data[i].columns = features
    mf_data[i].columns = features

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def split_data(data, target):
    target_t = target.drop(columns=['dummy']).copy()
    target_t = np.log(target_t).dropna()
    non_nan_index = target.dropna().index
    data_t = data.loc[non_nan_index]
    print(data_t.shape, target_t.shape)
    X_train, X_temp, y_train, y_temp = train_test_split(data_t, target_t, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
def objective(trial, data, target):
    X_train, X_val, _, y_train, y_val, _ = split_data(data, target)
    units = trial.suggest_int('units', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    activation_function = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh", "swish"])  # 优化激活函数
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    model = tf_keras.Sequential()
    model.add(tf_keras.Input(shape=(X_train.shape[1],)))
    model.add(tf_keras.layers.Dense(units,activation=activation_function))
    model.add(tf_keras.layers.Dropout(rate=dropout_rate))
    model.add(tf_keras.layers.Dense(1 + 1))
    model.add(tfp.layers.DistributionLambda(
          lambda t: tfd.Normal(loc=t[..., :1],
                               scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),)
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
    model.fit(x = X_train, y = y_train,
                        batch_size=batch_size,
                        epochs=2000,
                        validation_data=(X_val, y_val),
                        verbose=0)
    score = model.evaluate(X_val, y_val, verbose=0)
    return score

datasets = {}
for i in range(1, 5):
    datasets[f'ext_data_t_{i}'] = (ext_data_t[i], target[i])
    datasets[f'mice_data_{i}'] = (mice_data[i], target[i])
    datasets[f'mf_data_{i}'] = (mf_data[i], target[i])
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
df_best_params.to_excel('mha_pnn_best_params.xlsx', index=False)