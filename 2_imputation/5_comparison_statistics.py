import pandas as pd
import numpy as np
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib.patches as mpatches

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 48})
plt.rcParams['axes.linewidth'] = 5

folder_path = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
features = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$']

# Read data
ori_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ori_data_t.xlsx', sheet_name='Sheet_1')
ext_data = pd.read_excel(folder_path.parent / '1_extented_data' / 'ext_data_t.xlsx', sheet_name='Sheet_1')
mice_data = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_1')
mf_data = pd.read_excel('mf_data.xlsx', sheet_name='Sheet_1')

assert ori_data.shape == ext_data.shape == mice_data.shape == mf_data.shape

ori_data.columns = features
ext_data.columns = features
mice_data.columns = features
mf_data.columns = features

ori_data_flat = ori_data.melt(var_name='Feature', value_name='Data')
ori_data_flat['Group'] = 'Ori.'

ext_data_flat = ext_data.melt(var_name='Feature', value_name='Data')
ext_data_flat['Group'] = 'MN'

mice_data_flat = mice_data.melt(var_name='Feature', value_name='Data')
mice_data_flat['Group'] = 'MICE'

mf_data_flat = mf_data.melt(var_name='Feature', value_name='Data')
mf_data_flat['Group'] = 'MF'

combined_data = pd.concat([ori_data_flat, ext_data_flat, mice_data_flat, mf_data_flat], ignore_index=True)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(36, 18))
axes = axes.flatten()
# Define the colormap (use tab10)
colormap = plt.cm.tab10
labels = ['Ori.', 'MN', 'MICE', 'MF']
# Loop through each feature to create violin plots
for i, feature in enumerate(features):
    ax = axes[i]
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # Extract colors from the colormap
    palette = [colormap(j) for j in range(4)]  # 4 colors for 'ori', 'ext', 'mice', 'mf'
    sns.violinplot(x='Group', y='Data', data=combined_data[combined_data['Feature'] == feature],
                   ax=ax, palette=palette,bw=0.2, inner_kws=dict(box_width=15, whis_width=3, color='black'))
    ax.set_xlabel('')
    ax.set_ylabel('')
    if i > 2:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(labels, fontsize=48, fontweight='bold')
    else:
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([])
    ax.text(-0.22, 0.5, feature, ha='center', va='center', rotation=90, transform=ax.transAxes, fontsize=48)
    ax.tick_params(axis='both', which='major', pad=15, width=3, length=8)
    ax.grid(True, linestyle='--', alpha=0.8)

axes[-1].axis('off')
legend_handles = [mpatches.Rectangle((0, 0), 2, 2, color=colormap(i), ec='black') for i in range(4)]
axes[-1].legend(handles=legend_handles, labels=labels, frameon=True, shadow=True, borderpad=1, borderaxespad=1, fontsize=48,
                loc='center')
# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.1)
output_image_path = os.path.join(folder_path.parent, 'results', 'Comparison_Statistics.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()