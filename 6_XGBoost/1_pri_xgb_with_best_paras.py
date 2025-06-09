import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
tfd = tfp.distributions
tfb = tfp.bijectors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

def predict_with_best_params(data_t, target_t, labels, best_params):

    params = {
        'tree_method': 'auto',
        'objective': 'reg:squarederror',
        'learning_rate': best_params['learning_rate'],
        'max_depth': best_params['max_depth'],
        'min_child_weight': best_params['min_child_weight'],
        'gamma': best_params['gamma'],
        'colsample_bytree': best_params['colsample_bytree'],
        'subsample': best_params['subsample']
    }

    num_boost_round = 300

    mae_list = []
    data_r_list = []
    pre_r_list = []
    label_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_t, labels)):
        X_train, X_test = data_t.iloc[train_idx], data_t.iloc[test_idx]
        y_train, y_test = target_t.iloc[train_idx], target_t.iloc[test_idx]
        label_test = labels[test_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
        pre_test = model.predict(xgb.DMatrix(X_test))
        mae = mean_absolute_error(y_test, pre_test)
        pre_r_list.append(pre_test)
        mae_list.append(mae)
        data_r_list.append(y_test)
        label_list.append(label_test)
    return {
        "data_r_list": data_r_list,
        "pre_r_list": pre_r_list,
        "mae_list": mae_list,
        "label_list": label_list
    }

def format_pre_r_list(pre_r_list, data_r_list, label_list):
    all_preds = []
    for i, (pre_r, data_r, label) in enumerate(zip(pre_r_list, data_r_list, label_list)):
        df = pd.DataFrame(np.exp(pre_r), columns=["pre"])
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
    with pd.ExcelWriter(f"{name}_predictions_pri_xgb.xlsx") as writer:
        for fold in sorted(predictions_df["fold"].unique()):
            fold_df = predictions_df[predictions_df["fold"] == fold]
            fold_df.to_excel(writer, sheet_name=f"fold_{fold+1}", index=False)

    nll_list = {
        "dataset": name,
        "mae_list": prediction_likelihood["mae_list"]
    }
    results_nll.append(nll_list)

df_results_likelihoods = pd.DataFrame(results_nll)
df_results_likelihoods.to_excel("mae_results_xgb.xlsx", index=False)