import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow_probability as tfp
import random
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

all_su_pre = pd.read_excel(folder.parent / '0_MN_prediction_model' /'su_pre.xlsx')
labels = all_su_pre.iloc[:, -1].to_numpy()
mask = (labels >= 1) & (labels <= 7)
y_labels = labels[mask]

ori_data_t = pd.read_excel(folder.parent / '1_MN_dataset' /'ori_data_t.xlsx', sheet_name=f'Sheet_1')
ext_data_t = pd.read_excel(folder.parent / '1_MN_dataset' /'ext_data_t.xlsx', sheet_name=f'Sheet_1')
mice_data = pd.read_excel(folder.parent / '2_MICE_and_MF_datasets' /'mice_data.xlsx', sheet_name=f'Sheet_1')
mf_data = pd.read_excel(folder.parent / '2_MICE_and_MF_datasets' /'mf_data.xlsx', sheet_name=f'Sheet_1')
target = pd.read_excel(folder.parent / '1_MN_dataset' /'su_r.xlsx', sheet_name=f'Sheet_1')

ori_data_t.columns = features
ext_data_t.columns = features
mice_data.columns = features
mf_data.columns = features

target_t = target.drop(columns=['dummy']).copy()
target_t = np.log(target_t).dropna().squeeze()
non_nan_index = target.dropna().index
ori_data_t = ori_data_t.loc[non_nan_index]
ext_data_t = ext_data_t.loc[non_nan_index]
mice_data_t = mice_data.loc[non_nan_index]
mf_data_t = mf_data.loc[non_nan_index]
y_labels = y_labels[non_nan_index]

class XGBNLLRegressor:
    def __init__(self, params, num_boost_round):
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def _expand_inputs(self, X, y):

        X_exp = np.repeat(X.values, 2, axis=0)
        y_exp = np.repeat(y.values, 2, axis=0)
        return X_exp, y_exp

    def _nll_loss(self, preds, dtrain):
        labels = dtrain.get_label()
        preds = tf.convert_to_tensor(preds, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels[::2], dtype=tf.float32)

        mu = preds[::2]
        log_var = preds[1::2]
        var = tf.exp(log_var)

        grad_mu = (mu - labels) / var
        grad_log_var = 0.5 * (1 - ((mu - labels) ** 2) / var)

        hess_mu = 1.0 / var
        hess_log_var = 0.5 * ((mu - labels) ** 2) / var

        grad = tf.reshape(tf.stack([grad_mu, grad_log_var], axis=1), [-1])
        hess = tf.reshape(tf.stack([hess_mu, hess_log_var], axis=1), [-1])

        return grad.numpy(), hess.numpy()

    def fit(self, X_train, y_train, X_val, y_val):
        X_exp, y_exp = self._expand_inputs(X_train, y_train)
        dtrain = xgb.DMatrix(X_exp, label=y_exp)

        X_val_exp, y_val_exp = self._expand_inputs(X_val, y_val)
        dval = xgb.DMatrix(X_val_exp, label=y_val_exp)
        evals = [(dval, 'val')]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            obj=self._nll_loss,
            verbose_eval=False
        )

    def predict(self, X):
        X_exp = np.repeat(X, 2, axis=0)
        dtest = xgb.DMatrix(X_exp)
        preds = self.model.predict(dtest)
        mu = preds[::2]
        log_var = preds[1::2]
        sigma = np.sqrt(np.exp(log_var))
        return mu, sigma

    def evaluate(self, X, y_true):
        mu, sigma = self.predict(X)
        dist = tfd.Normal(loc=mu, scale=sigma)
        nll = -tf.reduce_mean(dist.log_prob(y_true))
        return {'NLL': nll}

def objective(trial, data_t, target_t, labels):

    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'auto',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 0.2),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
    }

    num_boost_round = 500

    nll_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBNLLRegressor(params=params, num_boost_round=num_boost_round)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_t, labels)):
        X_train, X_val = data_t.iloc[train_idx], data_t.iloc[test_idx]
        y_train, y_val = target_t.iloc[train_idx], target_t.iloc[test_idx]
        model.fit(X_train, y_train, X_val, y_val)
        score = model.evaluate(data_t, target_t)
        nll_list.append(score['NLL'].numpy())

    return np.mean(nll_list)

datasets = {
    'ori': (ori_data_t, target_t, y_labels),
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
        'learning_rate': study.best_params['learning_rate'],
        'max_depth': study.best_params['max_depth'],
        'min_child_weight': study.best_params['min_child_weight'],
        'gamma': study.best_params['gamma'],
        'colsample_bytree': study.best_params['colsample_bytree'],
        'subsample': study.best_params['subsample'],
        'best_score': study.best_value
    })

    for trial in study.trials:
        trial_result = {
            'dataset_name': name,
            'score': trial.value
        }
        all_trials.append(trial_result)

df_best_params = pd.DataFrame(best_params_list)
df_best_params.to_excel('xgb_best_params.xlsx', index=False)
df_all_trials = pd.DataFrame(all_trials)
df_all_trials.to_excel('xgb_all_trials.xlsx', index=False)
