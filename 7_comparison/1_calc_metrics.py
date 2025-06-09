import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import os.path
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

def cal_R2(data):
    true = data[:, 3]
    pred = data[:, 1]
    ss_total = np.sum((true - np.mean(true)) ** 2)
    ss_residual = np.sum((true - pred) ** 2)
    return 1 - (ss_residual / ss_total)

def cal_RMSE(data):
    true = data[:, 3]
    pred = data[:, 1]
    return np.sqrt(np.mean((true - pred) ** 2))

def cal_MAPE(data):
    true = data[:, 3]
    pred = data[:, 1]
    return np.mean(np.abs((true - pred) / true))

def cal_wCI(data):
    pred_L = data[:, 0]
    pred_U = data[:, 2]
    return np.mean(pred_U - pred_L)

def cal_coverage_rate(data):
    pred_L = data[:, 0]
    pred_U = data[:, 2]
    true = data[:, 3]
    covered = (pred_L <= true) & (true <= pred_U)
    return np.mean(covered)

ori_xgb_dfs, ext_xgb_dfs, mice_xgb_dfs, mf_xgb_dfs = [], [], [], []
mn_MHA_dfs, mice_MHA_dfs, mf_MHA_dfs = [], [], []

for j in range(5):
    # MHA-PNN
    mn_df = pd.read_excel(folder.parent/'5_mha_pnn'/'ext_predictions_MHA.xlsx', sheet_name=f"fold_{j + 1}")
    mice_df = pd.read_excel(folder.parent/'5_mha_pnn'/'mice_predictions_MHA.xlsx', sheet_name=f"fold_{j + 1}")
    mf_df = pd.read_excel(folder.parent/'5_mha_pnn'/'mf_predictions_MHA.xlsx', sheet_name=f"fold_{j + 1}")

    mn_MHA_dfs.append(mn_df[mn_df.iloc[:, -2] >= 4])
    mice_MHA_dfs.append(mice_df[mice_df.iloc[:, -2] >= 4])
    mf_MHA_dfs.append(mf_df[mf_df.iloc[:, -2] >= 4])

    # XGBoost
    ori_df = pd.read_excel(folder.parent/'6_XGBoost'/'ori_predictions_xgb.xlsx', sheet_name=f"fold_{j + 1}")
    ext_df = pd.read_excel(folder.parent/'6_XGBoost'/'ext_predictions_xgb.xlsx', sheet_name=f"fold_{j + 1}")
    mice_xgb_df = pd.read_excel(folder.parent/'6_XGBoost'/'mice_predictions_xgb.xlsx', sheet_name=f"fold_{j + 1}")
    mf_xgb_df = pd.read_excel(folder.parent/'6_XGBoost'/'mf_predictions_xgb.xlsx', sheet_name=f"fold_{j + 1}")

    ori_xgb_dfs.append(ori_df[ori_df.iloc[:, -2] >= 4])
    ext_xgb_dfs.append(ext_df[ext_df.iloc[:, -2] >= 4])
    mice_xgb_dfs.append(mice_xgb_df[mice_xgb_df.iloc[:, -2] >= 4])
    mf_xgb_dfs.append(mf_xgb_df[mf_xgb_df.iloc[:, -2] >= 4])

ori_mn_dfs = pd.read_excel('pre_ori.xlsx')
ori_mn_dfs = ori_mn_dfs[ori_mn_dfs.iloc[:, -1] >= 4]

def compute_metrics_overall(folds):
    metrics = []
    for df in folds:
        df = df.dropna()
        data = df.to_numpy()
        mape = cal_MAPE(data)
        rmse = cal_RMSE(data)
        r2 = cal_R2(data)
        wci = cal_wCI(data)
        cr = cal_coverage_rate(data)
        metrics.append([mape, rmse, r2, wci, cr])
    return np.mean(np.array(metrics), axis=0)

def compute_metrics_overall_single(df):
    df = df.dropna()
    data = df.to_numpy()
    mape = cal_MAPE(data)
    rmse = cal_RMSE(data)
    r2 = cal_R2(data)
    wci = cal_wCI(data)
    cr = cal_coverage_rate(data)
    return np.array([mape, rmse, r2, wci, cr])

all_results_overall = {
    'ori_xgb': compute_metrics_overall(ori_xgb_dfs),
    'ext_xgb': compute_metrics_overall(ext_xgb_dfs),
    'mice_xgb': compute_metrics_overall(mice_xgb_dfs),
    'mf_xgb': compute_metrics_overall(mf_xgb_dfs),
    'ori_mn': compute_metrics_overall_single(ori_mn_dfs),
    'mn_MHA': compute_metrics_overall(mn_MHA_dfs),
    'mice_MHA': compute_metrics_overall(mice_MHA_dfs),
    'mf_MHA': compute_metrics_overall(mf_MHA_dfs)
}

rows = []
for method, metrics in all_results_overall.items():
    rows.append([method] + list(metrics))

metrics_df = pd.DataFrame(rows, columns=['Method', 'MAPE', 'RMSE', 'R2', 'wCI', 'CR'])
metrics_df.to_excel("metrics_overall.xlsx", index=False)
