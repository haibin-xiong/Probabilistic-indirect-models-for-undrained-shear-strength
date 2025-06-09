import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
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
labels = all_su_pre.iloc[:, -1].to_numpy()
mask = (labels >= 1) & (labels <= 7)
y_labels = labels[mask]

ori_data_t = pd.read_excel(folder.parent / '1_extented_data' /'ori_data_t.xlsx', sheet_name=f'Sheet_1')
ext_data_t = pd.read_excel(folder.parent / '1_extented_data' /'ext_data_t.xlsx', sheet_name=f'Sheet_1')
mice_data = pd.read_excel(folder.parent / '2_imputation' /'mice_data.xlsx', sheet_name=f'Sheet_1')
mf_data = pd.read_excel(folder.parent / '2_imputation' /'mf_data.xlsx', sheet_name=f'Sheet_1')
target = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_1')

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

best_params_df = pd.read_excel('xgb_best_params.xlsx')

def calculate_pre_with_uncertainty(mean_predictions, std_predictions):
    predicted_samples = np.random.normal(loc=mean_predictions, scale=std_predictions, size=(1000, len(mean_predictions)))
    pre = np.percentile(predicted_samples, [2.5, 50, 97.5], axis=0).T
    return pre

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

def predict_with_best_params(data_t, target_t, labels, best_params):

    params = {
        'tree_method': 'auto',
        'learning_rate': best_params['learning_rate'],
        'max_depth': best_params['max_depth'],
        'min_child_weight': best_params['min_child_weight'],
        'gamma': best_params['gamma'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample']
    }

    num_boost_round = 500

    nll_list = []
    data_r_list = []
    pre_r_list = []
    label_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBNLLRegressor(params=params, num_boost_round=num_boost_round)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_t, labels)):
        X_train, X_val = data_t.iloc[train_idx], data_t.iloc[test_idx]
        y_train, y_val = target_t.iloc[train_idx], target_t.iloc[test_idx]
        label_test = labels[test_idx]
        model.fit(X_train, y_train, X_val, y_val)
        mean_predictions, std_predictions = model.predict(X_val)
        score = model.evaluate(X_val, y_val)
        pre_test = calculate_pre_with_uncertainty(mean_predictions, std_predictions)
        pre_r_list.append(pre_test)
        nll_list.append(score)
        data_r_list.append(y_val)
        label_list.append(label_test)
    return {
        "data_r_list": data_r_list,
        "pre_r_list": pre_r_list,
        "nll_list": nll_list,
        "label_list": label_list
    }

def format_pre_r_list(pre_r_list, data_r_list, label_list):
    all_preds = []
    for i, (pre_r, data_r, label) in enumerate(zip(pre_r_list, data_r_list, label_list)):
        df = pd.DataFrame(np.exp(pre_r), columns=["lower", "median", "upper"])
        df["real value"] = np.exp(np.array(data_r).reshape(-1))
        df["label"] = np.array(label).reshape(-1)
        df["fold"] = i
        all_preds.append(df)
    return pd.concat(all_preds, ignore_index=True)

datasets = {
    'ori': (ori_data_t, target_t, y_labels),
    'ext': (ext_data_t, target_t, y_labels),
    'mice': (mice_data_t, target_t, y_labels),
    'mf': (mf_data_t, target_t, y_labels)
}

results_nll = []

for name, (data, target, labels) in datasets.items():
    print(f"Processing dataset: {name}")

    best_params = best_params_df[best_params_df["dataset_names"] == name].iloc[0].to_dict()
    print(best_params)
    prediction_likelihood = predict_with_best_params(data, target, labels, best_params)
    predictions_df = format_pre_r_list(prediction_likelihood["pre_r_list"], prediction_likelihood["data_r_list"], prediction_likelihood["label_list"])
    with pd.ExcelWriter(f"{name}_predictions_xgb.xlsx") as writer:
        for fold in sorted(predictions_df["fold"].unique()):
            fold_df = predictions_df[predictions_df["fold"] == fold]
            fold_df.to_excel(writer, sheet_name=f"fold_{fold+1}", index=False)

    nll_list = {
        "dataset": name,
        "nll_list": prediction_likelihood["nll_list"]
    }
    results_nll.append(nll_list)

df_results_likelihoods = pd.DataFrame(results_nll)
df_results_likelihoods.to_excel("nll_results_xgb.xlsx", index=False)