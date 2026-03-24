#!/usr/bin/env python3
"""
fig_encoding_comparison.py  —  Run from paper/figures/
2×2 heatmap: v0 vs v1 OFFLOAD likelihood, neutral vs moderate SKIP likelihood.
Shows why correct action semantics matter (+7.9 pp each).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

states = ['LOW', 'MED', 'HIGH']
obs    = ['GOOD', 'MARG', 'BAD']

# OFFLOAD v0: per-stream QoS — offloaded stream gets ~7 FPS → near-BAD for all states
A_off_v0 = np.array([[0.05, 0.10, 0.85]] * 3)

# OFFLOAD v1: system-level effect — offloading frees local GPU (from paper Eq. 3)
A_off_v1 = np.array([[0.30, 0.55, 0.15],
                     [0.50, 0.35, 0.15],
                     [0.60, 0.30, 0.10]])

# SKIP neutral: frame-skipping produces no signal → appears costless
A_skip_n = np.array([[0.50, 0.50, 0.00]] * 3)

# SKIP moderate: encodes coverage cost as degraded service
A_skip_m = np.array([[0.20, 0.45, 0.35]] * 3)

WRONG_CMAP = 'YlOrRd'
RIGHT_CMAP = 'YlGnBu'
RED   = '#c0392b'
GREEN = '#1a7a3a'

panels = [
    (0, 0, A_off_v0, 'OFFLOAD  v0  ✗',   'per-stream QoS → always near-BAD',    WRONG_CMAP, RED),
    (0, 2, A_off_v1, 'OFFLOAD  v1  ✓',   'system-level: GOOD↑ as load rises',   RIGHT_CMAP, GREEN),
    (1, 0, A_skip_n, 'SKIP  neutral  ✗',  'no BAD column → appears costless',    WRONG_CMAP, RED),
    (1, 2, A_skip_m, 'SKIP  moderate  ✓', '35% BAD → encodes coverage cost',     RIGHT_CMAP, GREEN),
]

fig = plt.figure(figsize=(10, 4.8))
gs  = GridSpec(2, 3, figure=fig, width_ratios=[1, 0.18, 1],
               wspace=0.55, hspace=0.72)

for row, col, data, title, subtitle, cmap, tcolor in panels:
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(3)); ax.set_xticklabels(obs, fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(states, fontsize=8.5)
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=tcolor, pad=5)
    ax.set_xlabel(subtitle, fontsize=7.5, color='#555555', style='italic')
    if col == 0:
        ax.set_ylabel('Load State', fontsize=8.5)
    for r in range(3):
        for c in range(3):
            v = data[r, c]
            tc = 'white' if v > 0.55 else '#1a1a1a'
            ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Arrow columns with +7.9 pp label
for row in [0, 1]:
    ax_a = fig.add_subplot(gs[row, 1])
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis('off')
    ax_a.annotate('', xy=(0.88, 0.5), xytext=(0.12, 0.5),
                  arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.8))
    ax_a.text(0.5, 0.73, '+7.9 pp', ha='center', fontsize=11,
              color=GREEN, fontweight='bold')
    ax_a.text(0.5, 0.27, 'SLO ↑', ha='center', fontsize=8.5,
              color=GREEN, style='italic')

plt.savefig('fig_encoding_comparison.pdf', bbox_inches='tight', dpi=150)
plt.savefig('fig_encoding_comparison.png', bbox_inches='tight', dpi=150)
print('Saved: fig_encoding_comparison.pdf  +  fig_encoding_comparison.png')
