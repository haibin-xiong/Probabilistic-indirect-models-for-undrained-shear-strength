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
original_data = pd.read_excel('ori_data_t.xlsx')
# print(original_data)
extended_data = pd.read_excel('ext_data_t.xlsx')
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq']

def arcsigmoid(x):
    return np.log(x / (1 - x))

sns.set_style("white")
z = 0.7
miu_sigmas = []

dataset_origin = pd.DataFrame()
dataset_extended = pd.DataFrame()
for i, var in enumerate(variables):

    data_r_origin = original_data.iloc[:, i]
    data_r_extended = extended_data.iloc[:, i]

    index_origin = ~np.isnan(data_r_origin)
    index_extended = ~np.isnan(data_r_extended)

    data_r_t_origin = data_r_origin[index_origin]
    data_r_t_extended = data_r_extended[index_extended]

    y_posz_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(z))
    y_negz_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(-z))
    miu_ori = np.percentile(data_r_t_origin, 100 * norm.cdf(0))
    sigma_ori = 0.5 * (y_posz_ori - y_negz_ori)

    y_posz_ext = np.percentile(data_r_t_extended, 100 * norm.cdf(z))
    y_negz_ext = np.percentile(data_r_t_extended, 100 * norm.cdf(-z))
    miu_ext = np.percentile(data_r_t_extended, 100 * norm.cdf(0))
    sigma_ext = 0.5 * (y_posz_ext - y_negz_ext)

    miu_sigmas.append({
        'Variable': var,
        'miu_ori': miu_ori,
        'sigma_ori': sigma_ori,
        'miu_ext': miu_ext,
        'sigma_ext': sigma_ext
    })

    dataset_origin[var] = data_r_t_origin
    dataset_extended[var] = data_r_t_extended

miu_sigmas = pd.DataFrame(miu_sigmas)
miu_sigmas.to_excel('miu_sigmas_t.xlsx', index=False)

Nrow = 7
variable_limits = np.zeros((Nrow, 2))

for i in range(Nrow):
    variable_limits[i, 0] = miu_sigmas['miu_ori'][i] - 4 * miu_sigmas['sigma_ori'][i]
    variable_limits[i, 1] = miu_sigmas['miu_ext'][i] + 4 * miu_sigmas['sigma_ext'][i]

variables_latex = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$']
fig, axes = plt.subplots(Nrow, Nrow, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
sns.set(style="white")

for i in range(len(dataset_origin.columns)):
    data = dataset_extended.iloc[:, i]
    data = data[np.isfinite(data)]
    ax = axes[i, i]
    x = np.linspace(variable_limits[i, 0], variable_limits[i, 1], 100)
    ax.plot(x, norm.pdf(x, miu_sigmas['miu_ext'][i], miu_sigmas['sigma_ext'][i]), '--k', linewidth=2)
    bin = int((data.max() - data.min()) * 5)
    ax.hist(data, bins=bin, density=True, color='skyblue', edgecolor='k')

    ax.set_xlim(variable_limits[i, :])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=3))

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
# up ext
for i in range(1, Nrow):
    for j in range(i):
        ax = axes[i, j]
        data_i = dataset_extended.iloc[:, i].dropna()
        data_j = dataset_extended.iloc[:, j].dropna()
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

        if j == 0:
            ax.set_ylabel(variables_latex[i], fontsize=16, fontweight='bold')
            ax.yaxis.set_label_coords(-0.5, 0.5)
        else:
            ax.set_ylabel("")

        if i == Nrow - 1:
            ax.set_xlabel(variables_latex[j], fontsize=16, fontweight='bold')
        else:
            ax.set_xlabel("")

for i in range(Nrow):
    for j in range(Nrow):
        ax = axes[i, j]
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

output_path = os.path.join(folder.parent, 'results', 'Comparison_marginal_distribution_of_transformed_variables.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()