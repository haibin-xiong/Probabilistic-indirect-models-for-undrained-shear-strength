import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import os
import pathlib

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
su_r = {}
for i in range(1, 5):
    su_r[i] = pd.read_excel(folder.parent/'1_extented_data'/'su_r.xlsx', sheet_name=f'Sheet_{i}')
    su_r[i].drop(columns=['dummy'], inplace=True)

# Set up the figure (2x2 grid for 4 plots)
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Dataset names and file paths
datasets = [
    ('dataset_mice_1_all_su_pre.xlsx', 'dataset_mf_1_all_su_pre.xlsx', su_r[1], axs[0, 0]),
    ('dataset_mice_2_all_su_pre.xlsx', 'dataset_mf_2_all_su_pre.xlsx', su_r[2], axs[0, 1]),
    ('dataset_mice_3_all_su_pre.xlsx', 'dataset_mf_3_all_su_pre.xlsx', su_r[3], axs[1, 0]),
    ('dataset_mice_4_all_su_pre.xlsx', 'dataset_mf_4_all_su_pre.xlsx', su_r[4], axs[1, 1]),
]
subplot_labels = ['a', 'b', 'c', 'd']
subplot_texts = ['Original Database',
                 'Subset 1',
                 'Subset 2',
                 'Subset 3']

# Loop over each dataset and plot
for i, (dataset_mice, dataset_mf, target, ax) in enumerate(datasets):
    # Load data
    mice_data = pd.read_excel(dataset_mice)
    mf_data = pd.read_excel(dataset_mf)

    # Scatter plot for the original data
    colormap = plt.cm.tab10
    ax.scatter(target, mice_data[mice_data.columns[1]],
               color=colormap(0), marker='s', s=100, alpha=0.9, linewidth=2, edgecolors='black',label='MICE', zorder=2)
    ax.vlines(target, mice_data[mice_data.columns[0]],
              mice_data[mice_data.columns[2]], color=colormap(0), alpha=0.5,linewidth=2, zorder=1)

    # Scatter plot for the extended data
    ax.scatter(target, mf_data[mf_data.columns[1]],
               color=colormap(3), marker='o', s=80, alpha=0.9, linewidth=2, edgecolors='black', label='MF', zorder=2)
    ax.vlines(target, mf_data[mf_data.columns[0]],
              mf_data[mf_data.columns[2]], color=colormap(3), alpha=0.3, linewidth=2, zorder=1)

    # Reference line
    x_line = np.linspace(0.02, 18, 100)
    ax.plot(x_line, x_line, color='k', linestyle=(0, (10, 5)), linewidth=2, zorder=0)

    # Set logarithmic scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set limits
    ax.set_xlim([0.03, 17])
    ax.set_ylim([0.03, 17])

    # Aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Label the axes
    ax.set_xlabel('Real Values', fontsize=28)
    ax.set_ylabel('Predictions', fontsize=28)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=10, width=2)
    ax.tick_params(axis='both', which='minor', length=8, width=1.5)

    ax.text(0.05, 8.5,
            f"({subplot_labels[i]}) {subplot_texts[i]}", fontsize=24, fontweight='bold', color='black')
    # Legend
    ax.legend(frameon=True, shadow=False, borderpad=0.5, borderaxespad=0.5, fontsize=24,
              ncol=2, loc="lower right", columnspacing=0.5,facecolor='white', edgecolor='black')

# Adjust layout for better spacing between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=-0.3, hspace=0.2)
# Save the output
output_path = os.path.join(folder.parent, 'results', 'imputed_comparison_based_on_nm.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()