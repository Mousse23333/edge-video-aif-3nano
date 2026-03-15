"""
Shared plotting configuration — colors, styles, labels, rcParams.
Publication-quality settings for two-column conference papers.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Try SciencePlots, fall back gracefully ───────────────────────────────────
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

# ── Figure dimensions (inches) ──────────────────────────────────────────────
SINGLE_COL_W = 3.33
DOUBLE_COL_W = 7.0
ASPECT = 0.72   # height = width * ASPECT

# ── rcParams for publication ────────────────────────────────────────────────
RCPARAMS = {
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.edgecolor': '#cccccc',
    'legend.fancybox': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
}


def apply_style():
    """Apply publication rcParams globally."""
    plt.rcParams.update(RCPARAMS)


# ── Colorblind-safe palette (Okabe-Ito / Wong) ─────────────────────────────
COLORS = {
    'noop':      '#000000',    # black
    'heuristic': '#56B4E9',    # sky blue
    'myopic':    '#009E73',    # bluish green
    'dqn':       '#D55E00',    # vermillion
    'aif':       '#CC79A7',    # reddish purple
}

# Mode colors
MODE_COLORS = {
    'FULL':    '#4477AA',   # blue
    'LITE':    '#66CCEE',   # cyan
    'SKIP':    '#BBBBBB',   # gray
    'OFFLOAD': '#EE6677',   # red/coral
}

# Ablation colors
ABLATION_CMAP = plt.cm.viridis

# ── Display labels ──────────────────────────────────────────────────────────
CTRL_LABELS = {
    'noop':      'No-Op',
    'heuristic': 'Heuristic',
    'myopic':    'Myopic Greedy',
    'dqn':       'DQN',
    'aif':       'AIF (v1)',
}

SCENARIO_LABELS = {
    'scenario_ramp_up':          'Ramp-Up',
    'scenario_burst':            'Burst',
    'scenario_steady_overload':  'Steady Overload',
    'scenario_oscillating':      'Oscillating',
}

SCENARIO_SHORT = {
    'scenario_ramp_up':          'Ramp',
    'scenario_burst':            'Burst',
    'scenario_steady_overload':  'Steady',
    'scenario_oscillating':      'Oscill.',
}

METRIC_LABELS = {
    'slo_satisfaction_rate': 'SLO Satisfaction Rate',
    'slo_violations':        'SLO Violations',
    'n_mode_switches':       'Mode Switches',
    'skip_ratio':            'Skip Ratio',
    'offload_ratio':         'Offload Ratio',
    'avg_fps':               'Avg FPS',
    'avg_latency_ms':        'Avg P95 Latency (ms)',
}

# ── Ordering ────────────────────────────────────────────────────────────────
CTRL_ORDER = ['noop', 'heuristic', 'myopic', 'dqn', 'aif']
SCENARIO_ORDER = [
    'scenario_ramp_up',
    'scenario_burst',
    'scenario_steady_overload',
    'scenario_oscillating',
]
MODE_ORDER = ['FULL', 'LITE', 'SKIP', 'OFFLOAD']


# ── Helpers ─────────────────────────────────────────────────────────────────

def savefig(fig, path, **kwargs):
    """Save figure as both PDF (vector) and PNG (preview)."""
    fig.savefig(path + '.pdf', format='pdf', **kwargs)
    fig.savefig(path + '.png', format='png', dpi=300, **kwargs)
    print(f"  Saved: {path}.pdf / .png")
    plt.close(fig)


def add_slo_line(ax, threshold=0.9, label='90% Target'):
    """Add horizontal dashed SLO reference line."""
    ax.axhline(y=threshold, color='#888888', linestyle='--',
               linewidth=0.8, zorder=0, label=label)
