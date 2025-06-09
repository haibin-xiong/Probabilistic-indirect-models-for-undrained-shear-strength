import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import scienceplots

# 设置绘图样式
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

# === 1. 加载 Excel 数据 ===
df = pd.read_excel("metrics_overall.xlsx")

# === 2. 设置 Method 为索引 ===
df.set_index('Method', inplace=True)

# === 3. 指标归一化（统一方向：越大越好）===
df_norm = df.copy()

# 对于越小越好的指标，先反转
for col in ['MAPE', 'RMSE', 'wCI']:
    df_norm[col] = -df_norm[col]

# 归一化，并设置最小值以避免图形从中心穿出
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
print(df_norm)
# === 4. 绘制雷达图 ===
labels = df_norm.columns.tolist()
num_vars = len(labels)

# 角度划分
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 配色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# 画图
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

for i, (method, row) in enumerate(df_norm.iterrows()):
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=method, linewidth=2, color=colors[i % len(colors)])
    ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=14)

# 设置极坐标的 y 轴范围与网格样式
ax.set_ylim(-0.05, 1)
ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)

# 隐藏 y 轴刻度
ax.set_yticklabels([])

# 设置图例
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), fontsize=10)

# 提升整体美观性
plt.tight_layout()
plt.savefig("radar_model_comparison.png", dpi=300)
# plt.show()
