import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 28})
plt.rcParams['axes.linewidth'] = 3

import os
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

# 读取数据
origin_data = pd.read_excel('original_data.xlsx')
all_su_pre = pd.read_excel('su_pre.xlsx')

# 获取标签范围 1 到 7
unique_labels = all_su_pre.iloc[:, -1].unique()
unique_labels = [label for label in unique_labels if 1 <= label <= 7]
Inputs = ["Single available input", "Two available inputs", "Three available inputs", "Four available inputs", "Five available inputs", "Six available inputs", "Seven available inputs"]
Labels = ["Label = 1", "Label = 2", "Label = 3", "Label = 4", "Label = 5", "Label = 6", "Label = 7"]

# 按标签大小排序
unique_labels.sort()

# 选择更容易区分的颜色映射（tab10）
colormap = plt.cm.tab10
colors = [colormap(i / (len(unique_labels) - 1)) for i in range(len(unique_labels))]

# 创建 4x2 子图（8 个子图，最后一个只显示图例）
fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 4 行 2 列布局
axes = axes.flatten()

# 计算统计特征
def compute_statistics(values, name):
    print(f"Statistics for {name}:")
    print(f"Count: {len(values)}")
    print(f"Mean: {np.mean(values)}")
    print(f"Median: {np.median(values)}")
    print(f"Standard Deviation: {np.std(values)}")
    print(f"Minimum: {np.min(values)}")
    print(f"Maximum: {np.max(values)}")
    print(f"25th Percentile: {np.percentile(values, 25)}")
    print(f"75th Percentile: {np.percentile(values, 75)}")
    print("\n")

# 参考线
x_line = np.linspace(0.02, 18, 100)

# 遍历每个 Inputs 对应的标签进行绘图
for idx, label in enumerate(unique_labels):
    mask = all_su_pre.iloc[:, -1] == label
    selected_data = all_su_pre.loc[mask]

    x_values = origin_data.loc[selected_data.index, 'su(mob)/s¢v0']
    y_values = selected_data.iloc[:, 1]

    ax = axes[idx]

    ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=2,
            label='Predictions = Real Values' if idx == 0 else '', zorder=0)

    ax.scatter(x_values, y_values, s=100, alpha=0.9, color=colors[idx],
               edgecolors='black', linewidth=2, label=Labels[label - 1], zorder=2)

    # 误差条
    ax.vlines(x_values, selected_data.iloc[:, 0], selected_data.iloc[:, 2],
              color=colors[idx], alpha=0.5, linewidth=2, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.03, 17])
    ax.set_ylim([0.03, 17])
    ax.tick_params(axis='both', which='major', length=10, width=2)
    ax.tick_params(axis='both', which='minor', length=8, width=1.5)
    ax.set_aspect('equal', adjustable='box')

    ax.text(0.05, 8.5, f"({chr(97 + idx)}) {Labels[label - 1]}", fontsize=28, fontweight='bold', color='black')
    print(idx)
    x_values_clean = x_values[~np.isnan(x_values)]
    compute_statistics(x_values_clean, "x_values")
# 处理最后一个子图（仅显示图例）
axes[-1].axis('off')  # 移除坐标轴
legend_handles = [
    plt.Line2D([0], [0], linestyle=(0, (10, 5)), linewidth=2, color='k', label='Pred. = Real')
] + [plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                             markerfacecolor=colors[i], markeredgecolor='black', label=Labels[i])
                  for i in range(len(unique_labels))]
axes[-1].legend(handles=legend_handles, frameon=True, shadow=True, borderpad=1, borderaxespad=1, fontsize=28,
                loc='center')

# 设置全局标签
fig.text(0.5, 0, 'Real Values', ha='center', fontsize=32)
fig.text(0, 0.5, 'Predictions', va='center', rotation='vertical', fontsize=32)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
# 保存图像
output_path = os.path.join(folder.parent, 'results', 'joint_model_comparison_split.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()