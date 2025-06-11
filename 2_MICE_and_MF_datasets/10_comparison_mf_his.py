import pandas as pd
import numpy as np
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 56})
plt.rcParams['axes.linewidth'] = 3

folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

original_data_1 = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_1')
original_data_1.columns = features
original_data_2 = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_2')
original_data_2.columns = features
original_data_3 = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_3')
original_data_3.columns = features
original_data_4 = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_4')
original_data_4.columns = features

mf_data_1 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_1')
mf_data_1.columns = features
mf_data_2 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_2')
mf_data_2.columns = features
mf_data_3 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_3')
mf_data_3.columns = features
mf_data_4 = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_4')
mf_data_4.columns = features

latex_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$']

fig, axs = plt.subplots(4, 7, figsize=(54, 21))
axs = axs.flatten()

for i in range(4):
    for idx, feature in enumerate(features):
        ax = axs[idx + i * 7]
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        original_data_i = globals().get(f"original_data_{i+1}")
        mice_data_i = globals().get(f"mf_data_{i+1}")

        bins = 20

        sns.histplot(
            original_data_i[feature], bins=bins, kde=False,
            fill=False, ax=ax,
            color='black', edgecolor='black', linewidth=5,
            stat='density', alpha=0.8
        )

        hist, bin_edges = np.histogram(mice_data_i[feature].dropna(), bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, hist, color='red', linewidth=3, marker='o', markersize=12,
                markerfacecolor='red', markeredgewidth=0.5, linestyle='-', alpha=0.9)

        if i == 3:
            ax.set_xlabel(latex_labels[idx], fontweight='bold', labelpad=15, fontsize=56)
        else:
            ax.set_xlabel('')

        ax.set_ylabel('', fontsize=56)
        ax.tick_params(axis='both', which='major', pad=15, width=3, length=8)
        ax.grid(True, linestyle='--', alpha=0.8)

subplot_labels = [rf'$(\mathit{{{l}}}_{{{i}}})$' for l in 'abcd' for i in range(1, 8)]

for idx, ax in enumerate(axs):
    ax.text(
        0.85, 0.7, subplot_labels[idx], transform=ax.transAxes,
        fontsize=56, fontweight='bold', va='bottom', ha='center'
    )

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
output_image_path = os.path.join(folder_path.parent, 'results', 'Comparison_His_MF.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()