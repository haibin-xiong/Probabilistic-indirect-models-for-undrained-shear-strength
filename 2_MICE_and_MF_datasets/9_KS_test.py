import pandas as pd
from scipy.stats import ks_2samp
import os
import pathlib
folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$']
# 读取数据
ori_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_1')
ext_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ext_data_t.xlsx', sheet_name='Sheet_1')
mice_data = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_1')
mf_data = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_1')
ori_data.columns = features
ext_data.columns = features
mice_data.columns = features
mf_data.columns = features
# KS检验函数
def ks_test(reference, target, label):
    print(f"\n==== KS检验结果：{label} vs 原始数据 ====")
    for col in reference.columns:
        if col in target.columns:
            d, p = ks_2samp(reference[col].dropna(), target[col].dropna())
            print(f"{col:<15} | D = {d:.4f}, p = {p:.4f}")
        else:
            print(f"{col:<15} | 不存在于 {label} 中")

# 打印结果
ks_test(ori_data, ext_data, '扩展数据 ext_data')
ks_test(ori_data, mice_data, 'MICE填充数据 mice_data')
ks_test(ori_data, mf_data, 'MF填充数据 mf_data')
