import pandas as pd
import numpy as np
import os
import pathlib
from scipy import stats

# 设置文件路径
folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$']

# 加载数据
ori_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_1')
ext_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ext_data_t.xlsx', sheet_name='Sheet_1')
mice_data = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_1')
mf_data = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_1')

# 确保数据形状一致
assert ori_data.shape == ext_data.shape == mice_data.shape == mf_data.shape

# 设置列名
ori_data.columns = features
ext_data.columns = features
mice_data.columns = features
mf_data.columns = features

# 初始化结果存储
ks_results = []

# 对每个特征执行KS检验
for feature in features:
    # 原始数据(去除NaN)
    original = ori_data[feature].dropna()

    # 比较三种填补方法与原始数据的分布
    for method_name, method_data in zip(['MN', 'MICE', 'MF'], [ext_data, mice_data, mf_data]):
        # 获取填补数据
        imputed = method_data[feature]

        # 执行KS检验
        d_stat, p_value = stats.ks_2samp(original, imputed)

        # 存储结果
        ks_results.append({
            'Feature': feature,
            'Method': method_name,
            'KS_D': d_stat,
            'p_value': p_value,
            'Same_Distribution': p_value > 0.05  # 判断是否同分布
        })

# 转换为DataFrame
results_df = pd.DataFrame(ks_results)

# 输出结果到Excel
output_path = os.path.join(folder_path.parent, 'results', 'KS_Test_Results.xlsx')
results_df.to_excel(output_path, index=False)

# 打印汇总统计
print("\nKS检验结果汇总:")
print(f"总检验次数: {len(results_df)}")
print(f"分布一致的比例: {results_df['Same_Distribution'].mean():.2%}")

# 按方法和特征分组显示统计量
print("\n按方法和特征分组的KS统计量:")
grouped_stats = results_df.groupby(['Method', 'Feature']).agg({
    'KS_D': 'mean',
    'p_value': 'mean'
}).unstack(level=0)
print(grouped_stats.round(4))

# 保存分组统计
grouped_stats.to_excel(os.path.join(folder_path.parent, 'results', 'KS_Grouped_Stats.xlsx'))

print(f"\n结果已保存至: {output_path}")