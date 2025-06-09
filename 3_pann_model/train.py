import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pathlib
from train_and_test import split_data

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
miu_sigma = pd.read_excel(folder.parent/'0_based_multinormal_model'/'miu_sigma_t.xlsx')
Rmatrix = np.loadtxt(folder.parent/'0_based_multinormal_model'/'Rmatrix.txt')
ori_data_t = pd.read_excel(folder.parent/'1_extented_data'/'ori_data_t.xlsx')
target = pd.read_excel(folder.parent/'1_extented_data'/'target.xlsx')
target.drop(columns=['dummy'], inplace=True)
data = pd.concat([ori_data_t, target], axis=1)

variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
data.columns = variables
train_input, test_input, train_target, test_target = split_data(data, variables)

mask = np.isfinite(train_target)

train_input = train_input[mask]
train_target = train_target[mask]

xgb_model = xgb.XGBRegressor(missing=np.nan, n_estimators=100, max_depth=5)
xgb_model.fit(train_input, train_target)

xgb_pred = xgb_model.predict(test_input)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(train_input, train_target)

rf_pred = rf_model.predict(test_input)

import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图样式
sns.set(style="whitegrid")

# 创建一个 1 行 2 列的子图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 绘制训练集的对比图
axes[0].scatter(train_target, xgb_model.predict(train_input), color='blue', label='XGBoost (Train)', alpha=0.6)
axes[0].scatter(train_target, rf_model.predict(train_input), color='green', label='Random Forest (Train)', alpha=0.6)
axes[0].plot([train_target.min(), train_target.max()], [train_target.min(), train_target.max()], 'k--', lw=2)  # 完美预测线
axes[0].set_xlabel('True Values')
axes[0].set_ylabel('Predictions')
axes[0].set_title('Training Set Predictions')
axes[0].legend()

# 绘制测试集的对比图
axes[1].scatter(test_target, xgb_pred, color='red', label='XGBoost (Test)', alpha=0.6)
axes[1].scatter(test_target, rf_pred, color='orange', label='Random Forest (Test)', alpha=0.6)
axes[1].plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=2)  # 完美预测线
axes[1].set_xlabel('True Values')
axes[1].set_ylabel('Predictions')
axes[1].set_title('Test Set Predictions')
axes[1].legend()

# 调整布局
plt.tight_layout()

# 显示图形
output_path = os.path.join(folder.parent, 'results', 'xgb_and_rf.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()