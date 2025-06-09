import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import os.path
import pathlib

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

target = {}
for i in range(1, 5):
    target[i] = pd.read_excel(folder.parent / '1_extented_data' /'su_r.xlsx', sheet_name=f'Sheet_{i}')

datasets = {}
for i in range(1, 5):
    datasets[f'ext_data_t_{i}'] = (f'ext_data_t_{i}_predictions.xlsx', target[i])
    datasets[f'mice_data_{i}'] = (f'mice_data_{i}_predictions.xlsx', target[i])
    datasets[f'mf_data_{i}'] = (f'mf_data_{i}_predictions.xlsx', target[i])

def split_data(target):
    target_r = target.drop(columns=['dummy']).copy()
    target_r = target_r.dropna()
    y_train, y_temp = train_test_split(target_r, test_size=0.2, random_state=42)
    y_val, y_test = train_test_split(y_temp, test_size=0.5, random_state=42)
    return y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

fig, axes = plt.subplots(4, 3, figsize=(18, 24))
axes = axes.flatten()

colormap = plt.cm.viridis

for i, (name, (dataset_pred, target)) in enumerate(datasets.items()):

    ax = axes[i]

    pre_train_r = pd.read_excel(dataset_pred, sheet_name="pre_train_r")
    pre_val_r = pd.read_excel(dataset_pred, sheet_name="pre_val_r")
    pre_test_r = pd.read_excel(dataset_pred, sheet_name="pre_test_r")
    target_train_r, target_val_r, target_test_r = split_data(target)

    # ax.scatter(target_train_r, pre_train_r.iloc[:, 1], color=colormap(7), s=80, marker='s', facecolors='none',
    #            linewidth=2, label='Train')
    # ax.vlines(target_train_r, pre_train_r.iloc[:, 0], pre_train_r.iloc[:, 2], color=colormap(7), alpha=0.3, linewidth=2)

    ax.scatter(target_val_r, pre_val_r.iloc[:, 1], color=colormap(0.7), s=80, marker='s', facecolors='none',
               linewidth=2, label='Val')
    ax.vlines(target_val_r, pre_val_r.iloc[:, 0], pre_val_r.iloc[:, 2], color=colormap(0.7), alpha=0.3, linewidth=2)

    ax.scatter(target_test_r, pre_test_r.iloc[:, 1], color=colormap(0), s=80, marker='x',
               linewidth=2, label='Test')
    ax.vlines(target_test_r, pre_test_r.iloc[:, 0], pre_test_r.iloc[:, 2], color=colormap(0), alpha=0.3, linewidth=2)

    x_line = np.linspace(0.02, 18, 100)
    ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim([0.03, 17])
    ax.set_ylim([0.03, 17])

    ax.set_xlabel('Real Values', fontsize=14)
    ax.set_ylabel('Predictions', fontsize=14)

    ax.set_aspect('equal', adjustable='box')

    ax.text(0.05, 0.95, f"({i+1}) {name}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=2))

    ax.legend(frameon=True, shadow=True, borderpad=1, borderaxespad=1, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
output_image_path = os.path.join(folder.parent, 'results', 'comparison_based_on_ann.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()