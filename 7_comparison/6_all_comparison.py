import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scienceplots

# 设置绘图样式
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

# 加载数据
df = pd.read_excel("metrics_overall.xlsx")

# 模型名称（与 df 中的行顺序一致）
model_names = [
    "XGB model using Original datasets",
    "XGB model using MN datasets",
    "XGB model using MICE datasets",
    "XGB model using MF datasets",
    "MN model using Original datasets",
    "MHA-PNN model using MN datasets",
    "MHA-PNN model using MICE datasets",
    "MHA-PNN model using MF datasets"
]

# 指标列
metrics = ['MAPE', 'RMSE', 'R2', 'wCI', 'CR']
metrics_names = [r'$\mathit{MAPE}$', r'$\mathit{RMSE}$', r'$\mathit{R^2}$', r'$\mathit{w}_{\mathrm{CI}}$', r'$\mathit{CR}$']

# 转置：指标为列，模型为列名
df_plot = df.set_index('Method').T  # 假设你的Excel中“Method”是列名之一
df_plot.columns = model_names  # 更新列名为清晰模型名称

# 绘图参数
bar_width = 0.07
bar_dis = 0.03
total_models = len(model_names)
index = np.arange(len(metrics))

# 配色和纹理
colormap = plt.cm.tab20
hatch_patterns = ['x', 'o', '*', '.', '/', '\\', '-', '|']
colors = [colormap(i) for i in range(0, 2 * total_models, 2)]

# 图像尺寸
fig, ax = plt.subplots(figsize=(20, 9))

# 绘制柱状图
for i, model in enumerate(model_names):
    values = df_plot[model].values
    bar_positions = index + i * (bar_width + bar_dis)
    ax.bar(bar_positions, values, bar_width,
           alpha=0.9, color=colors[i], edgecolor='black', linewidth=3,
           hatch=hatch_patterns[i % len(hatch_patterns)], label=model)

# 设置 Y 轴
ax.set_ylabel("Metric Values", fontsize=28, fontweight="bold")
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.2)

# 设置 X 轴
xtick_positions = index + (bar_width + bar_dis) * total_models / 2 - (bar_width + bar_dis) / 2
ax.set_xticks(xtick_positions)
ax.set_xticklabels(metrics_names, fontsize=28)

# 设置图例
ax.legend(frameon=True, shadow=False, borderpad=0.5, borderaxespad=0.5,
          fontsize=16, loc="upper center", ncol=2, facecolor='white', edgecolor='black')

# 保存图像
output_dir = pathlib.Path(__file__).parent.parent / 'results'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'bar_comparison.png'

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
