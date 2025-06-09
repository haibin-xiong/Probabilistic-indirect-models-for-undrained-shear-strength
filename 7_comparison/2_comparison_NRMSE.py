import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 32})
plt.rcParams['axes.linewidth'] = 3

import os
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

metrics_train = pd.read_excel('metrics_summary.xlsx', sheet_name='train')
metrics_val = pd.read_excel('metrics_summary.xlsx', sheet_name='val')
metrics_test = pd.read_excel('metrics_summary.xlsx', sheet_name='test')

metrics = ["NRMSE", "wCI", "R2", "CR"]
models = [
    "MN model using Original datasets -",
    "MHA-PNN model using MN datasets -",
    "MHA-PNN model using MICE datasets -",
    "MHA-PNN model using MF datasets -"
]
datasets = ["Label = 1", "Label = 2", "Label = 3", "Label = 4", "Label = 5", "Label = 6", "Label = 7"]

train_NRMSE = [metrics_train["NRMSE"].values[i * 7:(i + 1) * 7] for i in range(4)]
val_NRMSE = [metrics_val["NRMSE"].values[i * 7:(i + 1) * 7] for i in range(4)]
test_NRMSE = [metrics_test["NRMSE"].values[i * 7:(i + 1) * 7] for i in range(4)]

bar_width = 0.15
bar_dis = 0.05
index = np.arange(len(datasets))

fig, ax = plt.subplots(figsize=(28, 12))

# Define colors for the stacked bars (Train, Validation, Test)
colormap = plt.cm.tab20
hatch_patterns = ['x', 'o', '*', '.']
# Plot the stacked bars for each model
for i, (train, val, test) in enumerate(zip(train_NRMSE, val_NRMSE, test_NRMSE)):

    ax.bar(index + i * (bar_width+bar_dis), train, bar_width, label=f'{models[i]} Train', alpha=0.7,
           color=colormap(0), edgecolor='black', linewidth=3, hatch=hatch_patterns[i])
    ax.bar(index + i * (bar_width+bar_dis), val, bar_width, bottom=train,alpha=0.7,
           label=f'{models[i]} Validation', color=colormap(4), edgecolor='black', linewidth=3, hatch=hatch_patterns[i])
    ax.bar(index + i * (bar_width+bar_dis), test, bar_width, bottom=train+val,alpha=0.7,
           label=f'{models[i]} Test', color=colormap(6), edgecolor='black', linewidth=3, hatch=hatch_patterns[i])

# Adding labels and title
ax.set_ylabel(r'$\mathit{NRMSE}$')
# ymin, ymax = ax.get_ylim()
# ax.set_ylim(ymin, ymax * 1.5)
ax.set_xticks(index + (bar_width+bar_dis) * 1.5)
ax.set_xticklabels(datasets)

# Adding legend
# ax.legend(frameon=True, shadow=False, borderpad=0.5, borderaxespad=0.5, fontsize=32,
#               ncol=2, loc="upper right", columnspacing=0.5,facecolor='white', edgecolor='black')

# Show the plot
plt.tight_layout()
output_path = os.path.join(folder.parent, 'results', 'NRMSE_bar_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
