import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import pathlib

folder = pathlib.Path(os.path.dirname(os.path.abspath('__file__')))
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'font.size': 24})
plt.rcParams['axes.linewidth'] = 3

file_path = 'mha_pnn_all_trials20.xlsx'
df = pd.read_excel(file_path)

epochs = df['epoch']
loss_mn = df['MN']
loss_mice = df['MICE']
loss_mf = df['MF']

# 限制loss不增大
for i in range(1, len(loss_mn)):
    if loss_mn[i] > loss_mn[i - 1]:
        loss_mn[i] = loss_mn[i - 1]
    if loss_mice[i] > loss_mice[i - 1]:
        loss_mice[i] = loss_mice[i - 1]
    if loss_mf[i] > loss_mf[i - 1]:
        loss_mf[i] = loss_mf[i - 1]

plt.figure(figsize=(8, 5))

plt.plot(epochs, loss_mn, label='MN', color='blue', marker='o', linestyle='--')
plt.plot(epochs, loss_mice, label='MICE', color='green', marker='s', linestyle='--')
plt.plot(epochs, loss_mf, label='MF', color='red', marker='^', linestyle='--')

plt.xlabel('Trial Number', fontsize=18, fontweight='bold')
plt.ylabel('Best Loss', fontsize=18, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

y_min, y_max = plt.ylim()
plt.ylim(y_min, y_max * 1.1)
plt.xlim(-0.5, 20)
plt.legend(fontsize=18, frameon=True, edgecolor='black', framealpha=1,
           loc='upper right', ncol=3, borderpad=0.5)

plt.grid(True)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)

plt.tight_layout()

output_path = os.path.join(folder.parent, 'results', 'loss_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
