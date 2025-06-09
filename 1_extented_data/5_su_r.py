import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib
folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']
su = ['su(mob)/s¢v0']
all_su_pre = pd.read_excel(folder_path.parent / '0_based_multinormal_model' / 'su_pre.xlsx')
su_r = pd.read_excel(folder_path.parent / '0_based_multinormal_model' / 'original_data.xlsx')[su]
index_1 = all_su_pre[all_su_pre.iloc[:, -1]>0].index
su_1 = su_r.loc[index_1]
index_2 = all_su_pre[all_su_pre.iloc[:, -1]>1].index
su_2 = su_r.loc[index_2]
index_3 = all_su_pre[all_su_pre.iloc[:, -1]>2].index
su_3 = su_r.loc[index_3]
index_4 = all_su_pre[all_su_pre.iloc[:, -1]>3].index
su_4 = su_r.loc[index_4]
index_5 = all_su_pre[(all_su_pre.iloc[:, -1] > 0) & (all_su_pre.iloc[:, -1] < 7)].index
su_5 = su_r.loc[index_5]
# 为 su_1、su_2、su_3 和 su_4 添加 'dummy' 列
su_1 = su_1.fillna(np.nan)
su_2 = su_2.fillna(np.nan)
su_3 = su_3.fillna(np.nan)
su_4 = su_4.fillna(np.nan)
su_5 = su_5.fillna(np.nan)
# 将 su_1、su_2、su_3 和 su_4 转换为 DataFrame 并添加 'dummy' 列
df_su_1 = pd.DataFrame(su_1)
df_su_1['dummy'] = 1

df_su_2 = pd.DataFrame(su_2)
df_su_2['dummy'] = 1

df_su_3 = pd.DataFrame(su_3)
df_su_3['dummy'] = 1

df_su_4 = pd.DataFrame(su_4)
df_su_4['dummy'] = 1

df_su_5 = pd.DataFrame(su_5)
df_su_5['dummy'] = 1

with pd.ExcelWriter("su_r.xlsx") as writer:
    df_su_1.to_excel(writer, sheet_name="Sheet_1", index=False)
    df_su_2.to_excel(writer, sheet_name="Sheet_2", index=False)
    df_su_3.to_excel(writer, sheet_name="Sheet_3", index=False)
    df_su_4.to_excel(writer, sheet_name="Sheet_4", index=False)
    df_su_5.to_excel(writer, sheet_name="Sheet_5", index=False)