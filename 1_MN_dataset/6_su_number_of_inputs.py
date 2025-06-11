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
target = pd.read_excel('su_r.xlsx')
target = target.iloc[:, :-1]

ori_data = pd.read_excel('su_pre_results.xlsx', sheet_name='su_pre_ori')
MN_data = pd.read_excel('su_pre_results.xlsx', sheet_name='su_pre_MN')

# 获取标签范围 1 到 7
unique_labels = ori_data.iloc[:, -1].unique()
unique_labels = [label for label in unique_labels if 1 <= label <= 7]
Inputs = ["Single Input", "Two Inputs", "Three Inputs", "Four Inputs", "Five Inputs", "Six Inputs", "Seven Inputs"]
Labels = ["Label = 1", "Label = 2", "Label = 3", "Label = 4", "Label = 5", "Label = 6", "Label = 7"]

unique_labels.sort()

colormap = plt.cm.tab10

fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 4 行 2 列布局
axes = axes.flatten()

x_line = np.linspace(0.02, 18, 100)

for idx, label in enumerate(unique_labels):
    mask = ori_data.iloc[:, -1] == label
    x_values = target.loc[mask]
    ori_y = ori_data.loc[mask]
    MN_y = MN_data.loc[mask]

    ax = axes[idx]

    ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=2,
            label='Predictions = Real Values' if idx == 0 else '', zorder=0)

    ax.scatter(x_values, ori_y.iloc[:, 1], s=100, alpha=0.9, color=colormap(0), marker='s',
               edgecolors='black', linewidth=2, label='Ori.' if idx == 0 else '', zorder=2,)
    ax.vlines(x_values, ori_y.iloc[:, 0], ori_y.iloc[:, 2],
              color=colormap(0), alpha=0.5, linewidth=2, zorder=1)

    ax.scatter(x_values, MN_y.iloc[:, 1], s=80, alpha=0.9, color=colormap(3), marker='o',
               edgecolors='black', linewidth=2, label='MN' if idx == 0 else '', zorder=2,)
    ax.vlines(x_values, MN_y.iloc[:, 0], MN_y.iloc[:, 2],
              color=colormap(3), alpha=0.5, linewidth=2, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.03, 17])
    ax.set_ylim([0.03, 17])
    ax.tick_params(axis='both', which='major', length=10, width=2)
    ax.tick_params(axis='both', which='minor', length=8, width=1.5)
    ax.set_aspect('equal', adjustable='box')

    ax.text(0.05, 8.5, f"({chr(97 + idx)}) {Labels[label - 1]}", fontsize=28, fontweight='bold', color='black')
labels = ['Original database', 'MN database']
markers = ['s', 'o']
axes[-1].axis('off')
legend_handles = [
    plt.Line2D([0], [0], linestyle=(0, (10, 5)), linewidth=2, color='k', label='Pred. = Real')
] + [plt.Line2D([0], [0], marker=markers[i], color='w', markersize=10,
                             markerfacecolor=colormap(i*3), markeredgecolor='black', label=labels[i])
                  for i in range(2)]
axes[-1].legend(handles=legend_handles, frameon=True, shadow=True, borderpad=1, borderaxespad=1, fontsize=28,
                loc='center')

# 设置全局标签
fig.text(0.5, 0, 'Real Values', ha='center', fontsize=32)
fig.text(0, 0.5, 'Predictions', va='center', rotation='vertical', fontsize=32)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
# 保存图像
output_path = os.path.join(folder.parent, 'results', 'Ori_and_MN.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()