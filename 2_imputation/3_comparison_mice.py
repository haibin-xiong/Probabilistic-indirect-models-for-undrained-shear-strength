import pandas as pd
import numpy as np
import os
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator

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

mice_data_1 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_1')
mice_data_1.columns = features
mice_data_2 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_2')
mice_data_2.columns = features
mice_data_3 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_3')
mice_data_3.columns = features
mice_data_4 = pd.read_excel('mice_data.xlsx', sheet_name='Sheet_4')
mice_data_4.columns = features

latex_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$']

colormap = plt.cm.tab10

color_palette_box = [
    (*colormap(0)[:3], 0.2),  # First color
    (*colormap(4)[:3], 0.2)  # Second color
]

color_palette_scatter = [colormap(0), colormap(4)]

fig, axs = plt.subplots(4, 7, figsize=(54, 21))
axs = axs.flatten()

def jitter(arr, strength=0.2):
    return arr + np.random.uniform(-strength, strength, size=arr.shape)

for i in range(4):
    for idx, feature in enumerate(features):
        ax = axs[idx + i * 7]  # Flatten the axis index correctly
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        # Dynamically access the variables
        original_data_i = globals().get(f"original_data_{i+1}")
        mice_data_i = globals().get(f"mice_data_{i+1}")

        plot_data = pd.DataFrame({
            'Origin': original_data_i[feature],
            'MICE': mice_data_i[feature]
        })

        for col_idx, (name, data) in enumerate([('Ori', original_data_i),
                                                ('MICE', mice_data_i)]):
            sns.boxplot(
                data=plot_data, palette=color_palette_box, ax=ax, width=0.8,
                showfliers=False, patch_artist=True,
                boxprops=dict(edgecolor='black', linewidth=5),
                medianprops=dict(color='black', linewidth=5),
                whiskerprops=dict(color='black', linewidth=5, linestyle='-'),
                capprops=dict(color='black', linewidth=5, linestyle='-'),
                whis=1.5
            )

        for col_idx, (name, data) in enumerate([('Ori', original_data_i),
                                                ('MICE', mice_data_i)]):

            ax.scatter(
                x=jitter(np.full_like(data[feature], col_idx)),
                y=data[feature],
                facecolor=color_palette_scatter[col_idx],
                s=50,
                zorder=1
            )

        if i == 3:
            ax.set_xlabel(latex_labels[idx], fontweight='bold', labelpad=15)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Ori.', 'MICE'], fontsize=56, fontweight='bold')
        else:
            ax.set_xlabel('')
            ax.set_xticks([0, 1])
            ax.set_xticklabels([])

        ax.tick_params(axis='both', which='major', pad=15, width=3, length=8)
        ax.grid(True, linestyle='--', alpha=0.8)

subplot_labels = [rf'$(\mathit{{{l}}}_{{{i}}})$' for l in 'abcd' for i in range(1, 8)]

for idx, ax in enumerate(axs):
    ax.text(
        0.5, 0.05, subplot_labels[idx], transform=ax.transAxes,
        fontsize=56, fontweight='bold', va='bottom', ha='center'
    )

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.1)
output_image_path = os.path.join(folder_path.parent, 'results', 'Comparison_Box_MICE.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()