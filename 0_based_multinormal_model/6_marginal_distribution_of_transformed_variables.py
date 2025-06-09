import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])
import os.path
import pathlib
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
origin_data = pd.read_excel('original_data.xlsx')
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']

def arcsigmoid(x):
    return np.log(x / (1 - x))

sns.set_style("white")
z = 0.7
miu_sigmas = []
dataset_origin = pd.DataFrame()

for i, var in enumerate(variables):

    data_r_origin = origin_data[var]
    index_origin = ~np.isnan(data_r_origin)
    data_r_origin = data_r_origin[index_origin]

    if var != 'Bq' and var != 'su(mob)/s¢v0':
        data_r_t_origin = np.arcsinh(data_r_origin)
    elif var == 'Bq':
        data_r_t_origin = arcsigmoid(data_r_origin / 1.2)
    else:
        data_r_t_origin = np.log(data_r_origin)

    y_posz_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(z))
    y_negz_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(-z))
    miu_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(0))
    sigma_ori = 0.5 * (y_posz_ori - y_negz_ori)
    miu_sigmas.append({
        'Variable': var,
        'miu_ori': miu_ori,
        'sigma_ori': sigma_ori
    })
    dataset_origin[var] = data_r_t_origin

miu_sigmas = pd.DataFrame(miu_sigmas)

Nrow = 8
variable_limits = np.zeros((Nrow, 2))

for i in range(Nrow):
    variable_limits[i, 0] = miu_sigmas['miu_ori'][i] - 4 * miu_sigmas['sigma_ori'][i]
    variable_limits[i, 1] = miu_sigmas['miu_ori'][i] + 4 * miu_sigmas['sigma_ori'][i]

variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
variables_latex = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$']
fig, axes = plt.subplots(Nrow, Nrow, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
sns.set(style="white")

for i in range(len(dataset_origin.columns)):
    data = dataset_origin.iloc[:, i]
    data = data[np.isfinite(data)]
    ax = axes[i, i]
    x = np.linspace(variable_limits[i, 0], variable_limits[i, 1], 100)
    ax.plot(x, norm.pdf(x, miu_sigmas['miu_ori'][i], miu_sigmas['sigma_ori'][i]), '--k', linewidth=2)
    bin = int((data.max() - data.min()) * 5)
    ax.hist(data, bins=bin, density=True, color='skyblue', edgecolor='k')

    ax.set_xlim(variable_limits[i, :])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=3))
    ax.set_xlabel(variables_latex[i], fontsize=16, fontweight='bold')
    ax.set_ylabel(variables_latex[i], fontsize=16, fontweight='bold')
# up ori
for i in range(Nrow):
    for j in range(i + 1, Nrow):
        ax = axes[i, j]
        data_i = dataset_origin.iloc[:, i].dropna()
        data_j = dataset_origin.iloc[:, j].dropna()
        common_index = data_i.index.intersection(data_j.index)
        data_i = data_i[common_index]
        data_j = data_j[common_index]

        ax.scatter(data_j, data_i, s=5, color='k')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=3))
        ax.set_xlim(variable_limits[j, :])
        ax.set_ylim(variable_limits[i, :])

for i in range(Nrow):
    for j in range(Nrow):
        ax = axes[i, j]
        if i == j:
            ax.set_xlim(variable_limits[i, :])
        else:
            ax.set_xlim(variable_limits[j, :])
            ax.set_ylim(variable_limits[i, :])

for i in range(Nrow):
    for j in range(Nrow):
        ax = axes[i, j]
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

for i in range(1, Nrow):
    for j in range(i):
        axes[i, j].set_visible(False)

output_path = os.path.join(folder.parent, 'results', 'marginal_distribution_of_transformed_variables.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
