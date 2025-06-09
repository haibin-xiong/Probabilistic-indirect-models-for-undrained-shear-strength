import scipy.stats
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 36})
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import os.path
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))

def johnson_su_pdf(y,ax,bx,ay,by):
    yn = (y-by)/ay
    return (ax/ay)*scipy.stats.norm.pdf(bx + ax*(np.arcsinh(yn)))/(np.sqrt(1+yn**2))

def johnson_sb_pdf(y,ax,bx,ay,by):
    yn = (y-by)/ay
    return (ax/ay)*scipy.stats.norm.pdf(bx + ax*(np.log(yn) - np.log(1-yn)))/(yn*(1-yn))

def johnson_sl_pdf(y,ax,bx,ay,by):
    yn = (y-by)/ay
    return (ax/ay)*scipy.stats.norm.pdf(bx + ax*np.log(ay)+ax*np.log(yn))/(yn)

def calculate_mnp(data, z):
    y_pos3z = np.percentile(data, 100 * scipy.stats.norm.cdf(3 * z))
    y_posz = np.percentile(data, 100 * scipy.stats.norm.cdf(z))
    y_negz = np.percentile(data, 100 * scipy.stats.norm.cdf(-z))
    y_neg3z = np.percentile(data, 100 * scipy.stats.norm.cdf(-3 * z))

    m = y_pos3z - y_posz
    n = y_negz - y_neg3z
    p = y_posz - y_negz
    return m, n, p

def calculate_d(m, n, p):
    return m * n / (p ** 2)

def calculate_parameters(m, n, p, d, y_posz, y_negz, z):
    if d > 1:  # SU
        ax = 2 * z / np.arccosh(0.5 * ((m / p) + (n / p)))
        bx = ax * np.arcsinh(((n / p) - (m / p)) / (2 * np.sqrt(d - 1)))
        ay = 2 * p * np.sqrt(d - 1) / (((m / p) + (n / p) - 2) * np.sqrt((m / p) + (n / p) + 2))
        by = (y_posz + y_negz) / 2 + p * ((n / p) - (m / p)) / (2 * ((m / p) + (n / p) - 2))
        dist_type = 'SU'
    elif d < 1:  # SB
        ax = z / np.arccosh(0.5 * np.sqrt((1 + (p / m)) * (1 + (p / n))))
        bx = ax * np.arcsinh(((p / n) - (p / m)) * np.sqrt((1 + (p / m)) * (1 + (p / n)) - 4) / 2 / (1/d - 1))
        ay = p * np.sqrt(((1 + (p / m)) * (1 + (p / n)) - 2) ** 2 - 4) / (1/d - 1)
        by = (y_posz + y_negz) / 2 - ay / 2 + p * ((p / n) - (p / m)) / 2 / (1/d - 1)
        dist_type = 'SB'
    else:  # SL
        ax = 2 * z / np.log(m / p)
        bx = ax * np.log(((m / p) - 1) / (p * np.sqrt(m / p)))
        ay = (y_posz + y_negz) / 2 - p * ((m / p) + 1) / ((m / p) - 1) / 2
        by = 0
        dist_type = 'SL'
    return ax, bx, ay, by, dist_type

import os.path
import pathlib
folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
samples = pd.read_excel('original_data.xlsx')
variables = ['LL (%)', 'PI (%)', 'LI', '(qt-svo)/s¢vo', '(qt-u2)/s¢vo', '(u2-u0)/s¢vo', 'Bq', 'su(mob)/s¢v0']
variables_latex = [
    r'$\mathit{LL} \, (\%)$',
    r'$\mathit{PI} \, (\%)$',
    r'$\mathit{LI}$',
    r'$ (q_{\mathrm{t}} - \sigma_{\mathrm{vo}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    r'$ (q_{\mathrm{t}} - u_{\mathrm{2}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    r'$ (u_{\mathrm{2}} - u_{\mathrm{0}}) / \sigma^{\prime}_{\mathrm{vo}} $',
    r'$ B_{\mathrm{q}} $',
    r'$ s_{\mathrm{u}} \,(\mathrm{mob}) / \sigma^{\prime}_{\mathrm{v}} $'
]
variables_latex1 = [
    "LL", "PI", "LI", "Qt1", "Qt2", "U2", "Bq", "Su"
]
xlim = [[0, 200],[0, 120],[-1, 4],[0, 30],[0, 15],[0, 15],[0, 1.2],[0, 1.5]]
fit_results = []
figures = []

fig, axs = plt.subplots(2, 4, figsize=(36, 12))
axs = axs.flatten()
z = 0.7
miu_sigma = []
subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
with pd.ExcelWriter("pdf_data.xlsx", engine="xlsxwriter") as writer:
    for i, var in enumerate(variables):
        data = samples[var].dropna()

        m, n, p = calculate_mnp(data, z)

        d = calculate_d(m, n, p)

        y_posz = np.percentile(data, 100 * scipy.stats.norm.cdf(z))
        y_negz = np.percentile(data, 100 * scipy.stats.norm.cdf(-z))
        mius = np.percentile(data, 100 * scipy.stats.norm.cdf(0))
        sigma = 0.5 * (y_posz-y_negz)
        a_x, b_x, a_y, b_y, dist_type = calculate_parameters(m, n, p, d, y_posz, y_negz, z)
        fit_results.append({
            'Variable': var,
            'Type': dist_type,
            'ax': a_x,
            'bx': b_x,
            'ay': a_y,
            'by': b_y
        })
        miu_sigma.append({
            'Variable': var,
            'miu': mius,
            'sigma': sigma
        })

        x = np.linspace(data.min(), data.max(), 1000)
        bin = int((data.max()-data.min()) / (xlim[i][1]-xlim[i][0]) * 50)
        if dist_type == 'SU':
            pdf = johnson_su_pdf(x, a_x, b_x, a_y, b_y)
        elif dist_type == 'SB':
            pdf = johnson_sb_pdf(x, a_x, b_x, a_y, b_y)
        else:  # SL
            pdf = johnson_sl_pdf(x, a_x, b_x, a_y, b_y)

        ax = axs[i]
        data_r = samples[var]
        ax.hist(data_r, bins=bin, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        ax.plot(x, pdf, color='black', lw=5, linestyle='--')
        ax.tick_params(axis='both')
        ax.spines['top'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_tick_params(pad=12)
        ax.set_xlabel(variables_latex[i], labelpad=10)
        ax.text(0.95, 0.91,
                f"({subplot_labels[i]}) {dist_type}\n$\mu={mius:.2f}$,\n$\sigma={sigma:.2f}$",
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=2))

        ax.set_xlim(xlim[i][0], xlim[i][1])
        ax.legend()
        df = pd.DataFrame({"x": x, "pdf": pdf})
        df.to_excel(writer, sheet_name=variables_latex1[i], index=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.3)
output_path = os.path.join(folder.parent, 'results', 'Johnson_fitted.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

results_df = pd.DataFrame(fit_results)
results_df.to_excel('johnson_fit_results.xlsx', index=False)

miu_sigma_df = pd.DataFrame(miu_sigma)
miu_sigma_df.to_excel('miu_sigma.xlsx', index=False)