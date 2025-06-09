import pandas as pd
import numpy as np
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
variables_latex = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$']
original_data = pd.read_excel(folder.parent/'0_based_multinormal_model'/'original_data.xlsx')
df = original_data[variables]
missing_bool = df.isnull().astype(int)
missing_corr = missing_bool.corr()

plt.figure(figsize=(14, 14))
ax = sns.heatmap(missing_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"shrink": 0.8})
ax.set_xticklabels(variables_latex, rotation=0)
ax.set_yticklabels(variables_latex)
plt.gca().set_aspect('equal', adjustable='box')
output_path = os.path.join(folder.parent, 'results', 'missing_co.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# import pandas as pd
# from sklearn.impute import KNNImputer
#
# # 假设 df 是你的 DataFrame
# # 初始化 KNNImputer，设定邻居数为 5（可根据实际情况调整）
# knn_imputer = KNNImputer(n_neighbors=5)
#
# # 进行填补操作，返回的是 NumPy 数组，再转换成 DataFrame
# df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
#
# # 查看填补后的缺失情况
# print(df_knn_imputed.isnull().sum())
#
# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer  # 启用实验性功能
# from sklearn.impute import IterativeImputer
#
# # 假设 df 是你的 DataFrame
# # 初始化 IterativeImputer，可以设置 random_state 保证结果可重复
# mice_imputer = IterativeImputer(random_state=0)
#
# # 进行插补操作，返回的结果转换为 DataFrame
# df_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
#
# # 查看插补后的缺失情况
# print(df_mice_imputed.isnull().sum())
#
# # 计算填补前的相关系数矩阵
# corr_original = df.corr()
#
# # 计算 KNN 填补后的相关系数矩阵
# corr_knn = df_knn_imputed.corr()
#
# # 计算 MICE 填补后的相关系数矩阵
# corr_mice = df_mice_imputed.corr()
#
# # 计算填补前后相关性的变化
# diff_knn = (corr_knn - corr_original).abs().mean().mean()
# diff_mice = (corr_mice - corr_original).abs().mean().mean()
#
# print(f"KNN 填补的平均相关性变化: {diff_knn:.4f}")
# print(f"MICE 填补的平均相关性变化: {diff_mice:.4f}")
#
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 创建 2 行 4 列的子图
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# axes = axes.flatten()  # 将子图数组展平成一维，方便索引
#
# # 遍历变量并绘制 KDE 曲线
# for i, variable in enumerate(variables):
#     ax = axes[i]  # 获取当前子图
#
#     # 原始数据
#     sns.kdeplot(df[i], color="red", ax=ax, label="Original")
#
#     # KNN 填补
#     sns.kdeplot(df_knn_imputed[i], color="blue", ax=ax, label="KNN")
#
#     # MICE 填补
#     sns.kdeplot(df_mice_imputed[i], color="green", ax=ax, label="MICE")
#
#     # 设置标题和图例
#     ax.set_title(variable)
#     ax.legend()
#
# # 调整子图布局，避免重叠
# plt.tight_layout()
#
# # 保存
# output_path = os.path.join(folder.parent, 'results', 'ml_comparison_grid.png')
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
#
# # 显示
# plt.show()
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats
# import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# tfb = tfp.bijectors
# import os.path
# import pathlib
#
# folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
# miu_sigma = pd.read_excel(folder.parent/'0_based_multinormal_model'/'miu_sigma_t.xlsx')
# Rmatrix = np.loadtxt(folder.parent/'0_based_multinormal_model'/'Rmatrix.txt')
# ori_data_t = pd.read_excel(folder.parent/'1_extented_data'/'ori_data_t.xlsx')
#
# mius = miu_sigma['miu']
# sigmas = miu_sigma['sigma']
# covariance = np.diag(sigmas.to_numpy()) @ Rmatrix @ np.diag(sigmas.to_numpy())
# covariance_beta = covariance[-1, :-1][np.newaxis, :] @ np.linalg.inv(covariance[:-1, :-1])
# sigma_ext = covariance[-1, -1] - covariance_beta @ covariance[:-1, -1]
# miu_df_mice_imputed = mius.to_numpy()[-1] + covariance_beta @ (df_mice_imputed.T - mius.to_numpy()[:-1][...,np.newaxis])
# miu_df_knn_imputed = mius.to_numpy()[-1] + covariance_beta @ (df_knn_imputed.T - mius.to_numpy()[:-1][...,np.newaxis])
#
# miu_ori = np.zeros((ori_data_t.shape[0], 1))
# sigma_ori = np.zeros((ori_data_t.shape[0], 1))
# for i, row in enumerate(ori_data_t.values):
#     non_nan_indices = np.where(~np.isnan(row))[0]
#     num_non_nan = len(non_nan_indices)
#     data = row[non_nan_indices]
#
#     selected_indices = np.unique(np.append(non_nan_indices, 7))
#     covariance_selected = covariance[np.ix_(selected_indices, selected_indices)]
#     covariance_selected_beta = covariance_selected[-1, :-1][np.newaxis, :] @ np.linalg.inv(covariance_selected[:-1, :-1])
#     sigma_ori[i] = covariance_selected[-1, -1] - covariance_selected_beta @ covariance_selected[:-1, -1]
#     miu_ori[i] = mius.to_numpy()[-1] + covariance_selected_beta @ (data.T - mius.to_numpy()[non_nan_indices])
#
# su_pre_ori = []
# su_pre_mice = []
# su_pre_knn = []
# for miu, sigma in ((miu_ori, sigma_ori), (miu_df_mice_imputed, sigma_ext), (miu_df_knn_imputed, sigma_ext)):
#     mvn = tfd.Normal(
#         loc=miu,
#         scale=sigma
#     )
#     predicted = mvn.sample(1000)
#     su_pre = np.percentile(np.exp(predicted), [2.5, 50, 97.5], axis=0).T
#
#     if miu is miu_ori:
#         su_pre_ori.append(su_pre)
#     elif miu is miu_df_mice_imputed:
#         su_pre_mice.append(su_pre)
#     else:
#         su_pre_knn.append(su_pre)
#
# su_pre_ori = np.array(su_pre_ori)
# su_pre_mice = np.array(su_pre_mice)
# su_pre_knn = np.array(su_pre_knn)
# df_su_pre_ori = pd.DataFrame(su_pre_ori.reshape(-1, su_pre_ori.shape[-1]), columns=["2.5%", "50%", "97.5%"])
# df_su_pre_mice = pd.DataFrame(su_pre_mice.reshape(-1, su_pre_mice.shape[-1]), columns=["2.5%", "50%", "97.5%"])
# df_su_pre_knn = pd.DataFrame(su_pre_knn.reshape(-1, su_pre_knn.shape[-1]), columns=["2.5%", "50%", "97.5%"])
#
# with pd.ExcelWriter("su_pre_results.xlsx") as writer:
#
#     df_su_pre_ori.to_excel(writer, sheet_name="su_pre_ori", index=False)
#     df_su_pre_mice.to_excel(writer, sheet_name="su_pre_mice", index=False)
#     df_su_pre_knn.to_excel(writer, sheet_name="su_pre_knn", index=False)