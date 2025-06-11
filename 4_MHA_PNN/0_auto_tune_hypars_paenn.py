import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import optuna
import tensorflow as tf
import tensorflow_probability as tfp
import random
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

all_su_pre = pd.read_excel(folder.parent / '0_based_multinormal_model' /'su_pre.xlsx')
labels = all_su_pre.iloc[:, -1].to_numpy()
mask = (labels >= 1) & (labels <= 7)
y_labels = labels[mask]

ext_data_t = pd.read_excel(folder.parent / '1_extented_data' /'ext_data_t.xlsx', sheet_name=f'Sheet_1')
mice_data = pd.read_excel(folder.parent / '2_imputation' /'mice_data.xlsx', sheet_name=f'Sheet_1')
mf_data = pd.read_excel(folder.parent / '2_imputation' /'mf_data.xlsx', sheet_name=f'Sheet_1')
target = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_1')

ext_data_t.columns = features
mice_data.columns = features
mf_data.columns = features

target_t = target.drop(columns=['dummy']).copy()
target_t = np.log(target_t).dropna().squeeze()
non_nan_index = target.dropna().index
ext_data_t = ext_data_t.loc[non_nan_index]
mice_data_t = mice_data.loc[non_nan_index]
mf_data_t = mf_data.loc[non_nan_index]
y_labels = y_labels[non_nan_index]

def objective(trial, data_t, target_t, labels):

    num_heads = trial.suggest_int('num_heads', 2, 20)
    key_dim = trial.suggest_int('key_dim', 2, 20)
    units = trial.suggest_int('units', 20, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    activation_function = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

    nll_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_t, labels)):
        X_train, X_val = data_t.iloc[train_idx], data_t.iloc[test_idx]
        y_train, y_val = target_t.iloc[train_idx], target_t.iloc[test_idx]

        inputs = tf.keras.Input(shape=(X_train.shape[1],))
        x = tf.keras.layers.Dense(units, activation=activation_function)(inputs)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        x = tf.expand_dims(x, axis=1)
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x,
                                                                                                                     x)
        x = tf.keras.layers.Add()([x, attn_output])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
        x = tf.keras.layers.Dense(2)(x)
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=tf.math.softplus(t[..., 1:]) + 1e-6))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
        model.fit(X_train.to_numpy(), y_train.to_numpy(), validation_data=(X_val, y_val), batch_size=64, epochs=2000,
                  verbose=0)
        score = model.evaluate(X_val.to_numpy(), y_val.to_numpy(), verbose=0)
        nll_list.append(score)

    return np.mean(nll_list)

datasets = {
    'ext': (ext_data_t, target_t, y_labels),
    'mice': (mice_data_t, target_t, y_labels),
    'mf': (mf_data_t, target_t, y_labels)
}
all_trials = []
best_params_list = []
for name, (data, target, labels) in datasets.items():
    print(name)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, data, target, labels), n_trials=10)

    best_params_list.append({
        'dataset_names': name,
        'num_heads': study.best_params['num_heads'],
        'key_dim': study.best_params['key_dim'],
        'units': study.best_params['units'],
        'dropout_rate': study.best_params['dropout_rate'],
        'learning_rate': study.best_params['learning_rate'],
        'activation_function': study.best_params['activation'],
        'best_score': study.best_value
    })
    for trial in study.trials:
        trial_result = {
            'dataset_name': name,
            'score': trial.value
        }
        all_trials.append(trial_result)

df_best_params = pd.DataFrame(best_params_list)
df_best_params.to_excel('mha_pnn_best_params.xlsx', index=False)
df_all_trials = pd.DataFrame(all_trials)
df_all_trials.to_excel('mha_pnn_all_trials.xlsx', index=False)