import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'sim_vs_gt_300conditions_force.csv')
OUT_DIR = SCRIPT_DIR
OUT_PATH = os.path.join(OUT_DIR, 'sim_vs_real_combined_force.png')

MAGS = [1, 3, 7]
AGENTS = list(range(10, 201, 10))
DURS = [1, 2, 3, 4, 5]

sim_M = {m: np.full((len(DURS), len(AGENTS)), np.nan) for m in MAGS}
gt_M = {m: np.full((len(DURS), len(AGENTS)), np.nan) for m in MAGS}

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        m, d, a = int(row['activated_magnets']), int(row['duration_s']), int(row['agent_count'])
        di, ai = DURS.index(d), AGENTS.index(a)
        sim_M[m][di, ai] = float(row['sim_final_left_fraction'])
        gt_M[m][di, ai] = float(row['gt_final_left_fraction'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

BLUE_CMAP = LinearSegmentedColormap.from_list(
    'blue_theme', ['#eaf3fb', '#4a90d9', '#0a1f3d'], N=256)

fig, axes = plt.subplots(2, 3, figsize=(9.5, 8.8), facecolor='w')
row_data = [gt_M, sim_M]
row_label = ['Experiment', 'Simulation']

im = None
for r, (data_dict, rlabel) in enumerate(zip(row_data, row_label)):
    for c, m in enumerate(MAGS):
        ax = axes[r, c]
        im = ax.imshow(data_dict[m], origin='lower', aspect='auto', cmap=BLUE_CMAP,
                        vmin=0, vmax=1,
                        extent=[AGENTS[0], AGENTS[-1], DURS[0], DURS[-1]])
        ax.set_xticks([10, 100, 200])
        ax.set_yticks(DURS)
        if r == 0:
            sfx = '' if m == 1 else 's'
            ax.set_title(f'{m} activated magnet{sfx}', fontsize=15, pad=8)
        if r == 1:
            ax.set_xlabel('Agent count', fontsize=18)
        if c == 0:
            ax.set_ylabel('Duration (s)', fontsize=18)
        else:
            ax.set_yticklabels([])

fig.subplots_adjust(left=0.135, right=0.865, top=0.91, bottom=0.08,
                    wspace=0.18, hspace=0.14)

for r, rlabel in enumerate(row_label):
    pos0 = axes[r, 0].get_position()
    y_center = (pos0.y0 + pos0.y1) / 2
    fig.text(0.060, y_center, rlabel, rotation=90, ha='center', va='center',
              fontsize=16, fontweight='bold')

cax = fig.add_axes([0.89, 0.08, 0.025, 0.83])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Final left-behind fraction', fontsize=19)
cb.ax.tick_params(labelsize=15)

fig.savefig(OUT_PATH, dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.08)
print('saved', OUT_PATH)
