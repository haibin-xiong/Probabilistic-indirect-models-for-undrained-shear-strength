import pandas as pd
import numpy as np
import os
import pathlib
import missingno as msno
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
original_data = pd.read_excel('original_data.xlsx')
original_data = original_data[variables]
num_samples = original_data.shape[0]
rename_mapping = {
    'LL (%)': r'$\mathit{LL} \, (\%)$',
    'PI (%)': r'$\mathit{PI} \, (\%)$',
    'LI': r'$\mathit{LI}$',
    '(qt-svo)/s¢vo': r'$ (q_{\mathrm{t}} - \sigma_{\mathrm{vo}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    '(qt-u2)/s¢vo': r'$ (q_{\mathrm{t}} - u_{\mathrm{2}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    '(u2-u0)/s¢vo': r'$ (u_{\mathrm{2}} - u_{\mathrm{0}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    'Bq': r'$ B_{\mathrm{q}} $',
    'su(mob)/s¢v0': r'$ s_{\mathrm{u}} \,(\mathrm{mob}) / \sigma^{\prime}_{\mathrm{v}} $'
}
original_data.rename(columns=rename_mapping, inplace=True)
ax = msno.matrix(original_data, figsize=(12, 8))
ax.xaxis.tick_bottom()
ax.set_xticklabels(original_data.columns, rotation=-60, ha='center')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.set_ylabel('Sample Index')
yticks = [1, 1000, 2000, 3000, 4000, 5000, 6264]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
output_path = os.path.join(folder.parent, 'results', 'missing_map.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()