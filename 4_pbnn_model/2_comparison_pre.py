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

datasets = {}
for i in range(1, 8):
    datasets[f'ext_data_t_{i}'] = f'label_{i}_ext_predictions.xlsx'
    datasets[f'mice_data_{i}'] = f'label_{i}_mice_predictions.xlsx'
    datasets[f'mf_data_{i}'] = f'label_{i}_mf_predictions.xlsx'

Inputs = ["Single Input", "Two Inputs", "Three Inputs", "Four Inputs", "Five Inputs", "Six Inputs", "Seven Inputs"]

colormap = plt.cm.tab10  # Using a larger colormap to ensure distinct colors

# Create subplots with 3 rows (train, val, test) and 7 columns (one for each group)
fig, axes = plt.subplots(3, 7, figsize=(28, 12))
axes = axes.flatten()

x_line = np.linspace(0.02, 18, 100)
subplot_labels = [rf'$(\mathit{{{l}}}_{{{i}}})$' for l in 'abc' for i in range(1, 8)]
for i in range(7):

    # Define subplot for the three categories (train, val, test)
    train_ax = axes[i]  # First row: training data
    val_ax = axes[i + 7]  # Second row: validation data
    test_ax = axes[i + 14]  # Third row: test data

    train_ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=3,
            label='Predictions = Real Values' if i == 0 else '', zorder=0)
    val_ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=3,
            label='Predictions = Real Values' if i == 0 else '', zorder=0)
    test_ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=3,
            label='Predictions = Real Values' if i == 0 else '', zorder=0)

    # Load data for EXT, MICE, MF for train, validation, and test
    ext_train_r = pd.read_excel(f'label_{i+1}_ext_predictions.xlsx', sheet_name="pre_train_r")
    ext_val_r = pd.read_excel(f'label_{i+1}_ext_predictions.xlsx', sheet_name="pre_val_r")
    ext_test_r = pd.read_excel(f'label_{i+1}_ext_predictions.xlsx', sheet_name="pre_test_r")

    mice_train_r = pd.read_excel(f'label_{i + 1}_mice_predictions.xlsx', sheet_name="pre_train_r")
    mice_val_r = pd.read_excel(f'label_{i + 1}_mice_predictions.xlsx', sheet_name="pre_val_r")
    mice_test_r = pd.read_excel(f'label_{i + 1}_mice_predictions.xlsx', sheet_name="pre_test_r")

    mf_train_r = pd.read_excel(f'label_{i + 1}_mf_predictions.xlsx', sheet_name="pre_train_r")
    mf_val_r = pd.read_excel(f'label_{i + 1}_mf_predictions.xlsx', sheet_name="pre_val_r")
    mf_test_r = pd.read_excel(f'label_{i + 1}_mf_predictions.xlsx', sheet_name="pre_test_r")

    # Plot for training set (using distinct colors)
    train_ax.vlines(ext_train_r.iloc[:, 3], ext_train_r.iloc[:, 0], ext_train_r.iloc[:, 2], color=colormap(0),
                    alpha=0.3, linewidth=2, zorder=1)
    train_ax.vlines(mice_train_r.iloc[:, 3], mice_train_r.iloc[:, 0], mice_train_r.iloc[:, 2], color=colormap(3),
                    alpha=0.3, linewidth=2, zorder=1)
    train_ax.vlines(mf_train_r.iloc[:, 3], mf_train_r.iloc[:, 0], mf_train_r.iloc[:, 2], color=colormap(4),
                    alpha=0.3, linewidth=2, zorder=1)

    train_ax.scatter(ext_train_r.iloc[:, 3], ext_train_r.iloc[:, 1], color=colormap(0), s=150, marker='s',
                     edgecolors='black', linewidth=2, label='EXT', zorder=2)
    train_ax.scatter(mice_train_r.iloc[:, 3], mice_train_r.iloc[:, 1], color=colormap(3), s=150, marker='o',
                     edgecolors='black', linewidth=2, label='MICE', zorder=2)
    train_ax.scatter(mf_train_r.iloc[:, 3], mf_train_r.iloc[:, 1], color=colormap(4), s=150, marker='<',
                     edgecolors='black', linewidth=2, label='MF', zorder=2)

    # Plot for validation set (using distinct colors)
    val_ax.vlines(ext_val_r.iloc[:, 3], ext_val_r.iloc[:, 0], ext_val_r.iloc[:, 2], color=colormap(0), alpha=0.3,
                  linewidth=2, zorder=1)
    val_ax.scatter(ext_val_r.iloc[:, 3], ext_val_r.iloc[:, 1], color=colormap(0), s=150, marker='s', edgecolors='black',
                   linewidth=2, label='EXT', zorder=2)

    val_ax.vlines(mice_val_r.iloc[:, 3], mice_val_r.iloc[:, 0], mice_val_r.iloc[:, 2], color=colormap(3), alpha=0.3,
                  linewidth=2, zorder=1)
    val_ax.scatter(mice_val_r.iloc[:, 3], mice_val_r.iloc[:, 1], color=colormap(3), s=150, marker='o',
                   edgecolors='black',
                   linewidth=2, label='MICE', zorder=2)

    val_ax.vlines(mf_val_r.iloc[:, 3], mf_val_r.iloc[:, 0], mf_val_r.iloc[:, 2], color=colormap(4), alpha=0.3,
                  linewidth=2, zorder=1)
    val_ax.scatter(mf_val_r.iloc[:, 3], mf_val_r.iloc[:, 1], color=colormap(4), s=150, marker='<', edgecolors='black',
                   linewidth=2, label='MF', zorder=2)

    # Plot for test set (using distinct colors)
    test_ax.vlines(ext_test_r.iloc[:, 3], ext_test_r.iloc[:, 0], ext_test_r.iloc[:, 2], color=colormap(0), alpha=0.3,
                   linewidth=2, zorder=1)
    test_ax.scatter(ext_test_r.iloc[:, 3], ext_test_r.iloc[:, 1], color=colormap(0), s=150, marker='s',
                    edgecolors='black',
                    linewidth=2, label='EXT', zorder=2)

    test_ax.vlines(mice_test_r.iloc[:, 3], mice_test_r.iloc[:, 0], mice_test_r.iloc[:, 2], color=colormap(3), alpha=0.3,
                   linewidth=2, zorder=1)
    test_ax.scatter(mice_test_r.iloc[:, 3], mice_test_r.iloc[:, 1], color=colormap(3), s=150, marker='o',
                    edgecolors='black',
                    linewidth=2, label='MICE', zorder=2)

    test_ax.vlines(mf_test_r.iloc[:, 3], mf_test_r.iloc[:, 0], mf_test_r.iloc[:, 2], color=colormap(4), alpha=0.3,
                   linewidth=2, zorder=1)
    test_ax.scatter(mf_test_r.iloc[:, 3], mf_test_r.iloc[:, 1], color=colormap(4), s=150, marker='<',
                    edgecolors='black',
                    linewidth=2, label='MF', zorder=2)

    # Set logarithmic scales for x and y axes
    for ax_ in [train_ax, val_ax, test_ax]:
        ax_.set_xscale("log")
        ax_.set_yscale("log")
        ax_.set_xlim([0.03, 17])
        ax_.set_ylim([0.03, 17])
        ax_.tick_params(axis='both', which='major', length=10, width=2)
        ax_.tick_params(axis='both', which='minor', length=8, width=1.5)
        ax_.set_aspect('equal', adjustable='box')

    for idx, ax in enumerate(axes):
        ax.text(
            0.05, 0.95, subplot_labels[idx], transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='left'
        )
# Set global labels
fig.text(0.5, 0, 'Real Values', ha='center', fontsize=28)
fig.text(0, 0.5, 'Predictions', va='center', rotation='vertical', fontsize=28)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# Save the figure
output_path = os.path.join(folder.parent, 'results', 'bann_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()