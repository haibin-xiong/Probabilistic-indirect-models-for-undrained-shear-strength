import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

import os
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

Input = ["Label = 1", "Label = 2", "Label = 3", "Label = 4", "Label = 5", "Label = 6", "Label = 7"]

colormap = plt.cm.tab10  # Using a larger colormap to ensure distinct colors
ori_dfs, ext_dfs, mice_dfs, mf_dfs = [], [], [], []

for j in range(5):
    ori_dfs.append(pd.read_excel('ori_predictions_pri_xgb.xlsx', sheet_name=f"fold_{j+1}"))
    ext_dfs.append(pd.read_excel('ext_predictions_pri_xgb.xlsx', sheet_name=f"fold_{j+1}"))
    mice_dfs.append(pd.read_excel('mice_predictions_pri_xgb.xlsx', sheet_name=f"fold_{j+1}"))
    mf_dfs.append(pd.read_excel('mf_predictions_pri_xgb.xlsx', sheet_name=f"fold_{j+1}"))
# Create subplots with 7 rows and 3 columns (one for each group)
fig, axes = plt.subplots(7, 5, figsize=(20, 28))  # Adjusted figsize to make it taller
axes = axes.flatten()

x_line = np.linspace(0.02, 18, 100)
subplot_labels = [rf'$(\mathit{{{l}}}_{{{i}}}):\ \mathrm{{label}} = {i}$' for l in 'abcde' for i in range(1, 8)]

for i in range(7):
    for j in range(5):
        ax = axes[i * 5 + j]
        ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=3,
                  label='Predictions = Real Values' if i == 0 else '', zorder=0)
        ori_i_j = ori_dfs[j][ori_dfs[j]["label"] == i+1]
        ext_i_j = ext_dfs[j][ext_dfs[j]["label"] == i+1]
        mice_i_j = mice_dfs[j][mice_dfs[j]["label"] == i+1]
        mf_i_j = mf_dfs[j][mf_dfs[j]["label"] == i+1]

        ax.scatter(ori_i_j.iloc[:, 1], ori_i_j.iloc[:, 0], color=colormap(0), s=150, marker='s',
                   edgecolors='black', linewidth=2, label='Ori.' if i == 0 and j == 0 else '', zorder=2)
        # ax.scatter(ext_i_j.iloc[:, 1], ext_i_j.iloc[:, 0], color=colormap(2), s=150, marker='o',
        #            edgecolors='black', linewidth=2, label='MN' if i == 0 and j == 0 else '', zorder=2)
        ax.scatter(mice_i_j.iloc[:, 1], mice_i_j.iloc[:, 0], color=colormap(4), s=150, marker='<',
                   edgecolors='black', linewidth=2, label='MICE' if i == 0 and j == 0 else '', zorder=2)
        # ax.scatter(mf_i_j.iloc[:, 1], mf_i_j.iloc[:, 0], color=colormap(6), s=150, marker='>',
        #            edgecolors='black', linewidth=2, label='MF' if i == 0 and j == 0 else '', zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([0.03, 17])
        ax.set_ylim([0.03, 17])
        ax.tick_params(axis='both', which='major', length=10, width=2)
        ax.tick_params(axis='both', which='minor', length=8, width=1.5)
        ax.set_aspect('equal', adjustable='box')

for col in range(5):
    for row in range(7):
        axes_idx = row * 5 + col
        label_idx = row + col * 7
        axes[axes_idx].text(
            0.05, 0.95, subplot_labels[label_idx], transform=axes[axes_idx].transAxes,
            fontsize=24, fontweight='bold', va='top', ha='left'
        )

# Set global labels
fig.text(0.5, 0, 'Real Values', ha='center', fontsize=32)
fig.text(0, 0.5, 'Predictions', va='center', rotation='vertical', fontsize=32)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# Save the figure
output_path = os.path.join(folder.parent, 'results', 'pri_xgb_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')