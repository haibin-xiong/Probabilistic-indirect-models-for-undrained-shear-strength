import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.linewidth'] = 3

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
model_names = [
    "ori_xgb",
    "ext_xgb",
    "mice_xgb",
    "mf_xgb",
    "ori_mn",
    "mn_MHA",
    "mice_MHA",
    "mf_MHA"
]

model_display_names = [
    "XGB (Ori.)",
    "XGB (MN)",
    "XGB (MICE)",
    "XGB (MF)",
    r'$\mathrm{MN}^{\mathrm{p}}$ (Ori.)',
    "MHA-PNN (MN)",
    "MHA-PNN (MICE)",
    "MHA-PNN (MF)"
]

metrics = ['MAPE', 'R2', 'RMSE', 'CR', 'wCI']
indicators = [
    r'$\mathit{MAPE}^{\mathrm{N}}$',
    r'$R^{2,\mathrm{N}}$',
    r'$\mathit{RMSE}^{\mathrm{N}}$',
    r'$\mathit{CR}^{\mathrm{N}}$',
    r'$\mathit{w}_{\mathrm{CI}}^{\mathrm{N}}$'
]

df = pd.read_excel("metrics_overall.xlsx")
df.set_index('Method', inplace=True)

df_norm = df.copy()
for col in ['MAPE', 'RMSE', 'wCI']:
    df_norm[col] = 1/df_norm[col]
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
print(df_norm)

num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def draw_pentagon_grid(ax, num_layers=5):
    """Draw a pentagon-shaped grid with specified layers and standard tick labels."""
    for layer in range(1, num_layers + 1):
        radius = layer / num_layers  # Normalize radius for each layer
        points = np.column_stack([angles, [radius] * len(angles)])

        if layer == num_layers:
            # Last layer (outer layer), solid line and thicker
            pentagon = plt.Polygon(points, edgecolor='black', fill=None, linewidth=3)
        else:
            # Inner layers, dashed line
            pentagon = plt.Polygon(points, edgecolor='gray', fill=None, linestyle='--', linewidth=1.5)

        ax.add_patch(pentagon)  # Add each layer as a pentagon

        # Add standard tick labels
        ax.text(np.pi, radius*np.cos(36/180*np.pi)-0.01, f'{radius:.1f}', ha='center', va='bottom', fontsize=16)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

draw_pentagon_grid(ax)

for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 1], color='gray', linestyle='--', linewidth=1.5)
colors = [
    '#1f77b4',  # XGB Ori.
    '#3399cc',  # XGB MN
    '#66b2cc',  # XGB MICE
    '#99cce5',  # XGB MF
    '#b2182b',  # MN Ori.
    '#d6604d',  # MHA MN
    '#f4a582',  # MHA MICE
    '#fddbc7'   # MHA MF
]
markers = ['o', 's', '^', 'D', 'v', 'P','X','*']
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (3, 5, 1, 5))]

for i, model in enumerate(model_names):

    values = df_norm.loc[model, metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, color=colors[i], linestyle=line_styles[i],
            linewidth=3, marker=markers[i], markersize=8, label=model_display_names[i],
            markerfacecolor='none',
            markeredgewidth=3
            )
    # ax.fill(angles, values, color=colors[i], alpha=0.2)

ax.set_ylim(-0.05, 1.2)
ax.set_xticklabels([])
for angle, label in zip(angles[:-1], indicators):
    ax.text(angle, 1.13, label, size=18,
            horizontalalignment='center',
            verticalalignment='center')

ax.spines['polar'].set_visible(False)
ax.grid(False)
ax.set_yticks([])
legend = fig.legend(loc='lower center',
                    bbox_to_anchor=(0.51, -0.05),
                    ncol=2,
                    fancybox=True,
                    framealpha=1,
                    shadow=True,
                    borderpad=0.5,
                    fontsize=18,
                    edgecolor='black',
                    frameon=True)
output_path = os.path.join(folder.parent, 'results', 'model_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
