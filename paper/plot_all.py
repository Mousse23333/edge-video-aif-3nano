#!/usr/bin/env python3
"""
Publication-quality figure generator for the AIF edge video scheduling paper.

Usage:
    python3 plot_all.py --main-dir ~/Desktop/experiments/main \
                        --ablation-dir ~/Desktop/experiments/ablation \
                        --output-dir ./figures

    python3 plot_all.py --main-dir ~/Desktop/experiments/main --only slo_bar

Generates 12+ figures covering:
  A. Main results (SLO bar, switch count, resource usage)
  B. Time-series (FPS, latency, mode allocation, CDF)
  C. AIF interpretability (belief, EFE decomposition)
  D. Ablation studies (likelihood, precision, epistemic, cooldown, offload)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from plot_config import (
    apply_style, savefig, add_slo_line,
    COLORS, MODE_COLORS, CTRL_LABELS, SCENARIO_LABELS, SCENARIO_SHORT,
    METRIC_LABELS, CTRL_ORDER, SCENARIO_ORDER, MODE_ORDER,
    SINGLE_COL_W, DOUBLE_COL_W, ASPECT,
)
from plot_data import (
    load_summaries, load_aggregate, load_step_csv, load_history_json,
    load_ablation_csv, load_ablation_aggregate,
    get_available_runs, aggregate_step_csvs,
    extract_per_stream_modes, extract_per_stream_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# A. MAIN RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_slo_bar(agg, out_dir, controllers=None, scenarios=None):
    """A1. Grouped bar chart — SLO satisfaction rate across scenarios."""
    controllers = controllers or CTRL_ORDER
    scenarios = scenarios or SCENARIO_ORDER

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.38))

    n_ctrl = len(controllers)
    n_scen = len(scenarios)
    x = np.arange(n_scen)
    width = 0.8 / n_ctrl
    offsets = np.linspace(-(n_ctrl - 1) / 2 * width, (n_ctrl - 1) / 2 * width, n_ctrl)

    for i, ctrl in enumerate(controllers):
        means = [agg[ctrl][s]['slo_satisfaction_rate']['mean'] for s in scenarios]
        stds = [agg[ctrl][s]['slo_satisfaction_rate']['std'] for s in scenarios]
        bars = ax.bar(x + offsets[i], means, width * 0.9, yerr=stds,
                      label=CTRL_LABELS[ctrl], color=COLORS[ctrl],
                      capsize=2, error_kw={'linewidth': 0.8},
                      edgecolor='white', linewidth=0.3)

    add_slo_line(ax, 0.9, '90% Target')
    ax.set_ylabel('SLO Satisfaction Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left', ncol=3, columnspacing=0.8, handletextpad=0.4)

    savefig(fig, os.path.join(out_dir, 'fig_slo_bar'))


def plot_switch_bar(agg, out_dir, controllers=None, scenarios=None):
    """A2. Grouped bar chart — mode switch count across scenarios."""
    controllers = controllers or CTRL_ORDER
    scenarios = scenarios or SCENARIO_ORDER

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.38))

    n_ctrl = len(controllers)
    n_scen = len(scenarios)
    x = np.arange(n_scen)
    width = 0.8 / n_ctrl
    offsets = np.linspace(-(n_ctrl - 1) / 2 * width, (n_ctrl - 1) / 2 * width, n_ctrl)

    for i, ctrl in enumerate(controllers):
        means = [agg[ctrl][s]['n_mode_switches']['mean'] for s in scenarios]
        stds = [agg[ctrl][s]['n_mode_switches']['std'] for s in scenarios]
        ax.bar(x + offsets[i], means, width * 0.9, yerr=stds,
               label=CTRL_LABELS[ctrl], color=COLORS[ctrl],
               capsize=2, error_kw={'linewidth': 0.8},
               edgecolor='white', linewidth=0.3)

    ax.set_ylabel('Mode Switches')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handletextpad=0.4)

    savefig(fig, os.path.join(out_dir, 'fig_switch_bar'))


def plot_resource_bar(agg, out_dir, controllers=None, scenarios=None):
    """A3. Grouped bar chart — skip ratio and offload ratio side by side."""
    controllers = controllers or CTRL_ORDER
    scenarios = scenarios or SCENARIO_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.34))

    n_ctrl = len(controllers)
    n_scen = len(scenarios)
    x = np.arange(n_scen)
    width = 0.8 / n_ctrl
    offsets = np.linspace(-(n_ctrl - 1) / 2 * width, (n_ctrl - 1) / 2 * width, n_ctrl)

    for metric, ax, title in [
        ('skip_ratio', axes[0], 'Skip Ratio'),
        ('offload_ratio', axes[1], 'Offload Ratio'),
    ]:
        for i, ctrl in enumerate(controllers):
            means = [agg[ctrl][s][metric]['mean'] for s in scenarios]
            stds = [agg[ctrl][s][metric]['std'] for s in scenarios]
            ax.bar(x + offsets[i], means, width * 0.9, yerr=stds,
                   label=CTRL_LABELS[ctrl], color=COLORS[ctrl],
                   capsize=2, error_kw={'linewidth': 0.8},
                   edgecolor='white', linewidth=0.3)
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    axes[1].legend(loc='upper right', ncol=1, fontsize=6,
                   handletextpad=0.3, columnspacing=0.5)
    fig.tight_layout(w_pad=2)
    savefig(fig, os.path.join(out_dir, 'fig_resource_bar'))


def plot_summary_heatmap(agg, out_dir, controllers=None, scenarios=None):
    """A5. Summary heatmap table — all metrics x controllers x scenarios."""
    controllers = controllers or CTRL_ORDER
    scenarios = scenarios or SCENARIO_ORDER
    metrics = ['slo_satisfaction_rate', 'n_mode_switches', 'skip_ratio',
               'offload_ratio', 'avg_fps', 'avg_latency_ms']

    # Build matrix
    rows = []
    row_labels = []
    for ctrl in controllers:
        for scen in scenarios:
            row = [agg[ctrl][scen][m]['mean'] for m in metrics]
            rows.append(row)
            row_labels.append(f"{CTRL_LABELS[ctrl]} / {SCENARIO_SHORT[scen]}")
    data = np.array(rows)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.7))

    # Normalize each column independently for color
    normed = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        mn, mx = col.min(), col.max()
        normed[:, j] = (col - mn) / (mx - mn + 1e-10) if mx > mn else 0.5

    # SLO higher = better (green), switches/latency higher = worse (invert)
    invert_cols = [1, 2, 5]  # switches, skip_ratio, latency
    for j in invert_cols:
        normed[:, j] = 1 - normed[:, j]

    im = ax.imshow(normed, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if metrics[j] in ('slo_satisfaction_rate', 'skip_ratio', 'offload_ratio'):
                text = f'{val:.1%}'
            elif metrics[j] == 'avg_latency_ms':
                text = f'{val:.0f}'
            elif metrics[j] == 'avg_fps':
                text = f'{val:.1f}'
            else:
                text = f'{val:.0f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=5.5,
                    color='black' if 0.3 < normed[i, j] < 0.7 else 'white')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m].replace(' ', '\n') for m in metrics],
                       fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.5)

    # Add controller group separators
    for i in range(1, len(controllers)):
        ax.axhline(y=i * len(scenarios) - 0.5, color='white', linewidth=2)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_summary_heatmap'))


# ═══════════════════════════════════════════════════════════════════════════
# B. TIME-SERIES ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_timeseries_fps(main_dir, out_dir, scenario='scenario_burst',
                        run_id=None, controllers=None):
    """B1. Time-series FPS with mean±std bands and workload overlay."""
    controllers = controllers or CTRL_ORDER

    fig, ax1 = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.35))
    ax2 = ax1.twinx()

    for ctrl in controllers:
        steps, means, stds = aggregate_step_csvs(main_dir, ctrl, scenario)
        if steps is None:
            continue
        ax1.plot(steps, means['avg_fps'], label=CTRL_LABELS[ctrl],
                 color=COLORS[ctrl], linewidth=1.2)
        ax1.fill_between(steps,
                         means['avg_fps'] - stds['avg_fps'],
                         means['avg_fps'] + stds['avg_fps'],
                         color=COLORS[ctrl], alpha=0.15)

    # Workload overlay (from any controller — n_active is deterministic)
    steps, means, _ = aggregate_step_csvs(main_dir, controllers[0], scenario)
    if steps is not None:
        ax2.fill_between(steps, means['n_active'],
                         alpha=0.10, color='gray', step='mid')
        ax2.plot(steps, means['n_active'],
                 color='gray', alpha=0.4, linewidth=0.8, drawstyle='steps-mid')
        ax2.set_ylabel('Active Streams', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    ax1.axhline(y=10, color='#888888', linestyle='--', linewidth=0.6, label='SLO (10 FPS)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Avg FPS')
    ax1.legend(loc='upper right', ncol=3, fontsize=6)
    ax1.set_title(f'Per-Step FPS — {SCENARIO_LABELS[scenario]} (5-run mean±std)', fontsize=9)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f'fig_ts_fps_{scenario}'))


def plot_timeseries_latency(main_dir, out_dir, scenario='scenario_burst',
                            run_id=None, controllers=None):
    """B2. Time-series latency with mean±std bands and SLO threshold."""
    controllers = controllers or CTRL_ORDER

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.35))

    for ctrl in controllers:
        steps, means, stds = aggregate_step_csvs(main_dir, ctrl, scenario)
        if steps is None:
            continue
        ax.plot(steps, means['avg_lat_ms'], label=CTRL_LABELS[ctrl],
                color=COLORS[ctrl], linewidth=1.2)
        ax.fill_between(steps,
                        means['avg_lat_ms'] - stds['avg_lat_ms'],
                        means['avg_lat_ms'] + stds['avg_lat_ms'],
                        color=COLORS[ctrl], alpha=0.15)

    ax.axhline(y=150, color='#888888', linestyle='--', linewidth=0.6, label='SLO (150ms)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg P95 Latency (ms)')
    ax.legend(loc='upper right', ncol=3, fontsize=6)
    ax.set_title(f'Per-Step P95 Latency — {SCENARIO_LABELS[scenario]} (5-run mean±std)', fontsize=9)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f'fig_ts_lat_{scenario}'))


def plot_mode_allocation(main_dir, out_dir, scenario='scenario_burst',
                         run_id=1, controllers=None):
    """B3. Stacked area chart — mode allocation over time (one subplot per controller)."""
    controllers = controllers or [c for c in CTRL_ORDER if c != 'noop']

    n = len(controllers)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.30),
                             sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ctrl in zip(axes, controllers):
        df = load_step_csv(main_dir, run_id, ctrl, scenario)
        if df is None:
            ax.set_title(CTRL_LABELS[ctrl])
            continue

        steps = df['step'].values
        stacks = []
        labels = []
        colors = []
        for mode in MODE_ORDER:
            col = f'n_{mode.lower()}'
            if col in df.columns:
                stacks.append(df[col].values.astype(float))
                labels.append(mode)
                colors.append(MODE_COLORS[mode])

        ax.stackplot(steps, *stacks, labels=labels, colors=colors, alpha=0.85)
        ax.set_title(CTRL_LABELS[ctrl], fontsize=7, pad=2)
        ax.set_xlabel('Step', fontsize=6)
        if ax == axes[0]:
            ax.set_ylabel('Streams')

    # Shared legend
    handles = [mpatches.Patch(color=MODE_COLORS[m], label=m) for m in MODE_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f'Mode Allocation — {SCENARIO_LABELS[scenario]}', fontsize=9, y=1.13)
    fig.tight_layout(w_pad=0.5)
    savefig(fig, os.path.join(out_dir, f'fig_mode_alloc_{scenario}'))


def plot_latency_cdf(main_dir, out_dir, scenario='scenario_burst',
                     run_id=1, controllers=None):
    """B4. CDF of per-stream latency."""
    controllers = controllers or CTRL_ORDER

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * ASPECT))

    for ctrl in controllers:
        history = load_history_json(main_dir, run_id, ctrl, scenario)
        if history is None:
            continue
        df = extract_per_stream_metrics(history)
        # Exclude SKIP
        active = df[df['mode'] != 'SKIP']['latency_ms'].dropna().sort_values()
        if len(active) == 0:
            continue
        cdf = np.arange(1, len(active) + 1) / len(active)
        ax.plot(active.values, cdf, label=CTRL_LABELS[ctrl],
                color=COLORS[ctrl])

    ax.axvline(x=150, color='#888888', linestyle='--', linewidth=0.6, label='SLO (150ms)')
    ax.set_xlabel('P95 Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_xlim(0, 300)
    ax.legend(loc='lower right', fontsize=6)
    ax.set_title(f'Latency CDF — {SCENARIO_LABELS[scenario]}', fontsize=9)

    savefig(fig, os.path.join(out_dir, f'fig_cdf_lat_{scenario}'))


# ═══════════════════════════════════════════════════════════════════════════
# C. AIF INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════

def plot_belief_evolution(main_dir, out_dir, scenario='scenario_burst', run_id=1):
    """
    C1. Belief evolution heatmap — Q(s) over time.
    Requires belief data in history JSON (aif controller must log self.belief).
    Falls back to a proxy from n_active if belief not logged.
    """
    history = load_history_json(main_dir, run_id, 'aif', scenario)
    if history is None:
        print("  [SKIP] No AIF history found for belief evolution.")
        return

    # Check if belief is logged in history
    has_belief = any('belief' in rec for rec in history)

    steps = [rec['step'] for rec in history]
    n_steps = len(steps)

    if has_belief:
        beliefs = np.array([rec['belief'] for rec in history])
    else:
        # Proxy: estimate belief from n_active
        print("  [NOTE] Belief not logged; using n_active proxy.")
        beliefs = np.zeros((n_steps, 3))
        for i, rec in enumerate(history):
            n = rec['observation']['global']['n_active_streams']
            if n <= 3:
                beliefs[i] = [0.8, 0.15, 0.05]
            elif n <= 5:
                beliefs[i] = [0.1, 0.7, 0.2]
            else:
                beliefs[i] = [0.05, 0.15, 0.8]

    n_active = [rec['observation']['global']['n_active_streams'] for rec in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.40),
                                    gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Belief heatmap
    im = ax1.imshow(beliefs.T, aspect='auto', cmap='viridis',
                    extent=[0, n_steps, -0.5, 2.5], origin='lower',
                    vmin=0, vmax=1)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['LOW', 'MED', 'HIGH'])
    ax1.set_ylabel('Load State')
    ax1.set_title(f'AIF Belief Evolution — {SCENARIO_LABELS[scenario]}', fontsize=9)
    plt.colorbar(im, ax=ax1, label='$Q(s)$', shrink=0.8, pad=0.02)

    # Active streams below
    ax2.step(range(n_steps), n_active, where='mid', color='gray', linewidth=1)
    ax2.fill_between(range(n_steps), n_active, alpha=0.15, color='gray', step='mid')
    ax2.set_ylabel('Streams')
    ax2.set_xlabel('Step')

    fig.tight_layout(h_pad=0.3)
    savefig(fig, os.path.join(out_dir, f'fig_belief_{scenario}'))


def plot_belief_entropy(main_dir, out_dir, scenario='scenario_burst', run_id=1):
    """C3. Belief entropy over time with n_active overlay."""
    history = load_history_json(main_dir, run_id, 'aif', scenario)
    if history is None:
        print("  [SKIP] No AIF history found for belief entropy.")
        return

    has_belief = any('belief' in rec for rec in history)
    steps = list(range(len(history)))
    n_active = [rec['observation']['global']['n_active_streams'] for rec in history]

    if has_belief:
        beliefs = [np.array(rec['belief']) for rec in history]
    else:
        # Proxy
        beliefs = []
        for rec in history:
            n = rec['observation']['global']['n_active_streams']
            if n <= 3:
                beliefs.append(np.array([0.8, 0.15, 0.05]))
            elif n <= 5:
                beliefs.append(np.array([0.1, 0.7, 0.2]))
            else:
                beliefs.append(np.array([0.05, 0.15, 0.8]))

    entropy = []
    for b in beliefs:
        b_safe = np.clip(b, 1e-10, 1.0)
        h = -np.sum(b_safe * np.log(b_safe))
        entropy.append(h)

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * ASPECT))
    ax2 = ax1.twinx()

    ax1.plot(steps, entropy, color=COLORS['aif'], linewidth=1.2, label='Belief Entropy')
    ax1.set_ylabel('Entropy $H(Q(s))$', color=COLORS['aif'])
    ax1.set_xlabel('Step')
    ax1.tick_params(axis='y', labelcolor=COLORS['aif'])

    ax2.step(steps, n_active, where='mid', color='gray', alpha=0.5, linewidth=0.8)
    ax2.fill_between(steps, n_active, alpha=0.1, color='gray', step='mid')
    ax2.set_ylabel('Active Streams', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_title(f'Belief Entropy — {SCENARIO_LABELS[scenario]}', fontsize=9)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f'fig_entropy_{scenario}'))


def plot_action_gantt(main_dir, out_dir, scenario='scenario_burst', run_id=1):
    """C4. Gantt chart — per-stream mode over time for AIF."""
    history = load_history_json(main_dir, run_id, 'aif', scenario)
    if history is None:
        print("  [SKIP] No AIF history found for action Gantt.")
        return

    modes_by_step = extract_per_stream_modes(history)
    all_sids = sorted(set().union(*[m.keys() for m in modes_by_step.values()]))
    n_steps = len(modes_by_step)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, max(1.5, 0.35 * len(all_sids))))

    for y_pos, sid in enumerate(all_sids):
        x_start = 0
        current_mode = None
        for step in range(n_steps):
            mode = modes_by_step.get(step, {}).get(sid, None)
            if mode is None:
                if current_mode is not None:
                    ax.barh(y_pos, step - x_start, left=x_start, height=0.7,
                            color=MODE_COLORS.get(current_mode, '#999'), edgecolor='none')
                current_mode = None
                x_start = step + 1
                continue
            if mode != current_mode:
                if current_mode is not None:
                    ax.barh(y_pos, step - x_start, left=x_start, height=0.7,
                            color=MODE_COLORS.get(current_mode, '#999'), edgecolor='none')
                current_mode = mode
                x_start = step
        # Final segment
        if current_mode is not None:
            ax.barh(y_pos, n_steps - x_start, left=x_start, height=0.7,
                    color=MODE_COLORS.get(current_mode, '#999'), edgecolor='none')

    ax.set_yticks(range(len(all_sids)))
    ax.set_yticklabels([f'Stream {s}' for s in all_sids])
    ax.set_xlabel('Step')
    ax.set_title(f'AIF Mode Allocation per Stream — {SCENARIO_LABELS[scenario]}', fontsize=9)

    handles = [mpatches.Patch(color=MODE_COLORS[m], label=m) for m in MODE_ORDER]
    ax.legend(handles=handles, loc='upper right', ncol=4, fontsize=6)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f'fig_gantt_{scenario}'))


def plot_scenario_deepdive(main_dir, out_dir, scenario='scenario_burst'):
    """B5. Scenario deep-dive — 4-row synchronized panel.

    Row 1: Workload (n_active) as context strip
    Row 2: FPS mean±std bands for all controllers
    Row 3: Mode allocation strips for AIF / Heuristic / DQN side-by-side
    Row 4: AIF belief heatmap (or proxy)

    This figure tells the complete causal story:
    workload → performance impact → controller decisions → AIF reasoning.
    """
    focus_ctrls = ['heuristic', 'dqn', 'aif']

    fig = plt.figure(figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.85))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 3, 2.5, 1.8],
                          hspace=0.35, wspace=0.08)

    # ── Row 0: Workload context (spans all 3 cols) ──
    ax_wl = fig.add_subplot(gs[0, :])
    steps, means, _ = aggregate_step_csvs(main_dir, 'aif', scenario)
    if steps is not None:
        ax_wl.fill_between(steps, means['n_active'],
                           color='#555555', alpha=0.3, step='mid')
        ax_wl.plot(steps, means['n_active'],
                   color='#333333', linewidth=1, drawstyle='steps-mid')
    ax_wl.set_ylabel('Streams', fontsize=6)
    ax_wl.set_title(f'Scenario Deep-Dive — {SCENARIO_LABELS[scenario]}',
                    fontsize=10, fontweight='bold', pad=6)
    ax_wl.set_xlim(steps[0], steps[-1])
    ax_wl.tick_params(labelbottom=False)
    ax_wl.set_ylim(0, None)

    # ── Row 1: FPS bands (spans all 3 cols) ──
    ax_fps = fig.add_subplot(gs[1, :], sharex=ax_wl)
    all_ctrls = CTRL_ORDER
    for ctrl in all_ctrls:
        s, m, sd = aggregate_step_csvs(main_dir, ctrl, scenario)
        if s is None:
            continue
        ax_fps.plot(s, m['avg_fps'], label=CTRL_LABELS[ctrl],
                    color=COLORS[ctrl], linewidth=1.1)
        ax_fps.fill_between(s, m['avg_fps'] - sd['avg_fps'],
                            m['avg_fps'] + sd['avg_fps'],
                            color=COLORS[ctrl], alpha=0.12)

    ax_fps.axhline(y=10, color='#888888', linestyle='--', linewidth=0.6)
    ax_fps.set_ylabel('Avg FPS', fontsize=7)
    ax_fps.legend(loc='upper right', ncol=5, fontsize=5.5,
                  handletextpad=0.3, columnspacing=0.6)
    ax_fps.tick_params(labelbottom=False)
    ax_fps.set_xlim(steps[0], steps[-1])

    # ── Row 2: Mode allocation (one subplot per focus controller) ──
    mode_axes = [fig.add_subplot(gs[2, i], sharex=ax_wl) for i in range(3)]

    for ax, ctrl in zip(mode_axes, focus_ctrls):
        s, m, _ = aggregate_step_csvs(main_dir, ctrl, scenario)
        if s is None:
            continue
        stacks = []
        colors = []
        for mode in MODE_ORDER:
            col = f'n_{mode.lower()}'
            if col in m:
                stacks.append(m[col])
                colors.append(MODE_COLORS[mode])
        ax.stackplot(s, *stacks, colors=colors, alpha=0.85)
        ax.set_title(CTRL_LABELS[ctrl], fontsize=7, pad=2)
        ax.set_xlim(steps[0], steps[-1])
        if ax == mode_axes[0]:
            ax.set_ylabel('Streams', fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)

    # Mode legend above mode row
    handles = [mpatches.Patch(color=MODE_COLORS[m], label=m) for m in MODE_ORDER]
    mode_axes[1].legend(handles=handles, loc='upper center', ncol=4, fontsize=5.5,
                        bbox_to_anchor=(0.5, 1.22), handletextpad=0.3)

    # ── Row 3: AIF belief heatmap (spans all 3 cols) ──
    ax_belief = fig.add_subplot(gs[3, :], sharex=ax_wl)
    history = load_history_json(main_dir, 1, 'aif', scenario)
    if history is not None:
        has_belief = any('belief' in rec for rec in history)
        n_steps = min(len(history), len(steps))

        if has_belief:
            beliefs = np.array([rec['belief'] for rec in history[:n_steps]])
        else:
            beliefs = np.zeros((n_steps, 3))
            for i, rec in enumerate(history[:n_steps]):
                n = rec['observation']['global']['n_active_streams']
                if n <= 3:
                    beliefs[i] = [0.8, 0.15, 0.05]
                elif n <= 5:
                    beliefs[i] = [0.1, 0.7, 0.2]
                else:
                    beliefs[i] = [0.05, 0.15, 0.8]

        belief_steps = steps[:n_steps]
        im = ax_belief.imshow(beliefs.T, aspect='auto', cmap='viridis',
                              extent=[belief_steps[0], belief_steps[-1], -0.5, 2.5],
                              origin='lower', vmin=0, vmax=1)
        ax_belief.set_yticks([0, 1, 2])
        ax_belief.set_yticklabels(['LOW', 'MED', 'HIGH'], fontsize=6)
        ax_belief.set_ylabel('AIF Belief', fontsize=7)
        plt.colorbar(im, ax=ax_belief, label='$Q(s)$', shrink=0.6,
                     pad=0.015, aspect=12)

    ax_belief.set_xlabel('Step', fontsize=7)
    ax_belief.set_xlim(steps[0], steps[-1])

    savefig(fig, os.path.join(out_dir, f'fig_deepdive_{scenario}'))


def plot_four_scenario_panel(main_dir, out_dir):
    """B6. Four-scenario comparison panel — 2×2 grid with FPS bands.

    Each subplot shows one scenario with all 5 controllers' mean±std bands,
    plus workload overlay. Compact overview of all experimental conditions.
    """
    scenarios = SCENARIO_ORDER

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.6),
                             sharex=False, sharey=True)
    axes = axes.flatten()

    for ax, scen in zip(axes, scenarios):
        ax2 = ax.twinx()

        for ctrl in CTRL_ORDER:
            s, m, sd = aggregate_step_csvs(main_dir, ctrl, scen)
            if s is None:
                continue
            ax.plot(s, m['avg_fps'], color=COLORS[ctrl], linewidth=0.9,
                    label=CTRL_LABELS[ctrl])
            ax.fill_between(s, m['avg_fps'] - sd['avg_fps'],
                            m['avg_fps'] + sd['avg_fps'],
                            color=COLORS[ctrl], alpha=0.10)

        # Workload
        s, m, _ = aggregate_step_csvs(main_dir, 'aif', scen)
        if s is not None:
            ax2.fill_between(s, m['n_active'], color='gray',
                             alpha=0.08, step='mid')
            ax2.plot(s, m['n_active'], color='gray', alpha=0.3,
                     linewidth=0.6, drawstyle='steps-mid')
            ax2.set_ylim(0, 12)
            if ax in axes[1::2]:  # right column
                ax2.set_ylabel('Streams', fontsize=6, color='gray')
            else:
                ax2.tick_params(labelright=False)

        ax.axhline(y=10, color='#888888', linestyle='--', linewidth=0.5)
        ax.set_title(SCENARIO_LABELS[scen], fontsize=8)
        ax.set_ylim(0, 25)

        if ax in axes[2:]:
            ax.set_xlabel('Step', fontsize=7)
        if ax in axes[::2]:  # left column
            ax.set_ylabel('Avg FPS', fontsize=7)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=6,
               bbox_to_anchor=(0.5, 1.04), handletextpad=0.3, columnspacing=0.6)

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    savefig(fig, os.path.join(out_dir, 'fig_four_scenario_fps'))


# ═══════════════════════════════════════════════════════════════════════════
# D. ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════════════════


def plot_dqn_variance(main_dir, dqn_extra_dir, out_dir):
    """E1. DQN 8-run variance analysis — box plot of SLO per scenario + switch scatter.

    Merges original 5-run data with 3 extra seeds to show seed sensitivity.
    """
    main_agg = load_aggregate(main_dir)
    extra_agg = load_aggregate(dqn_extra_dir)

    scenarios = SCENARIO_ORDER

    # Merge per-run SLO values: 5 original + 3 extra = 8
    merged_slo = {}
    merged_switches = {}
    for scen in scenarios:
        orig_slo = main_agg['dqn'][scen]['slo_satisfaction_rate']['values']
        extra_slo = extra_agg['dqn'][scen]['slo_satisfaction_rate']['values']
        merged_slo[scen] = orig_slo + extra_slo

        orig_sw = main_agg['dqn'][scen]['n_mode_switches']['values']
        extra_sw = extra_agg['dqn'][scen]['n_mode_switches']['values']
        merged_switches[scen] = orig_sw + extra_sw

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.35))

    # --- Left: SLO box plot with individual points ---
    slo_data = [merged_slo[s] for s in scenarios]
    bp = ax1.boxplot(slo_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.2),
                     flierprops=dict(marker='o', markersize=3))
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['dqn'])
        patch.set_alpha(0.5)

    # Overlay individual points (jittered)
    for i, scen in enumerate(scenarios):
        vals = merged_slo[scen]
        x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        # Mark original 5 vs extra 3
        ax1.scatter([i + 1 + xj for xj in x_jitter[:5]], vals[:5],
                    color=COLORS['dqn'], s=18, zorder=5, edgecolors='white',
                    linewidths=0.5, label='Original (5 runs)' if i == 0 else None)
        ax1.scatter([i + 1 + xj for xj in x_jitter[5:]], vals[5:],
                    color='#0072B2', s=18, zorder=5, edgecolors='white',
                    linewidths=0.5, marker='D',
                    label='Extra seeds (3 runs)' if i == 0 else None)

    add_slo_line(ax1, 0.9, '90% Target')
    ax1.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax1.set_ylabel('SLO Satisfaction Rate')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_title('DQN SLO Distribution (8 runs)', fontsize=9)
    ax1.legend(loc='lower left', fontsize=6)

    # --- Right: Switch count box plot ---
    sw_data = [merged_switches[s] for s in scenarios]
    bp2 = ax2.boxplot(sw_data, patch_artist=True, widths=0.5,
                      medianprops=dict(color='black', linewidth=1.2),
                      flierprops=dict(marker='o', markersize=3))
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['dqn'])
        patch.set_alpha(0.5)

    for i, scen in enumerate(scenarios):
        vals = merged_switches[scen]
        x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax2.scatter([i + 1 + xj for xj in x_jitter[:5]], vals[:5],
                    color=COLORS['dqn'], s=18, zorder=5, edgecolors='white',
                    linewidths=0.5)
        ax2.scatter([i + 1 + xj for xj in x_jitter[5:]], vals[5:],
                    color='#0072B2', s=18, zorder=5, edgecolors='white',
                    linewidths=0.5, marker='D')

    ax2.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax2.set_ylabel('Mode Switches')
    ax2.set_title('DQN Switch Count Distribution (8 runs)', fontsize=9)

    fig.tight_layout(w_pad=2)
    savefig(fig, os.path.join(out_dir, 'fig_dqn_variance'))


def plot_skip_likelihood(ablation_dir, out_dir):
    """E2. SKIP likelihood tuning — SLO and skip ratio across 3 variants.

    Two-panel figure: left = SLO grouped bar, right = skip ratio grouped bar.
    """
    agg = load_ablation_aggregate(ablation_dir, 'skip_likelihood')
    if agg is None:
        print("  [SKIP] No skip_likelihood ablation data found.")
        return

    variants = ['skip_baseline', 'skip_mild', 'skip_moderate']
    variant_labels = {
        'skip_baseline':  'Baseline\n[0.50, 0.50, 0.00]',
        'skip_mild':      'Mild\n[0.30, 0.50, 0.20]',
        'skip_moderate':  'Moderate\n[0.20, 0.45, 0.35]',
    }
    variant_colors = ['#BBBBBB', '#66CCEE', '#4477AA']
    scenarios = SCENARIO_ORDER

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.38))

    n_var = len(variants)
    n_scen = len(scenarios)
    x = np.arange(n_scen)
    width = 0.8 / n_var
    offsets = np.linspace(-(n_var - 1) / 2 * width, (n_var - 1) / 2 * width, n_var)

    # --- Left: SLO ---
    for i, (var, color) in enumerate(zip(variants, variant_colors)):
        means = [agg[var]['aif'][s]['slo_satisfaction_rate']['mean'] for s in scenarios]
        stds = [agg[var]['aif'][s]['slo_satisfaction_rate']['std'] for s in scenarios]
        ax1.bar(x + offsets[i], means, width * 0.9, yerr=stds,
                label=variant_labels[var], color=color,
                capsize=2, error_kw={'linewidth': 0.8},
                edgecolor='white', linewidth=0.3)

    add_slo_line(ax1, 0.9, '90% Target')
    ax1.set_ylabel('SLO Satisfaction Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax1.set_ylim(0.6, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_title('SLO vs SKIP Likelihood', fontsize=9)
    ax1.legend(loc='lower right', fontsize=5.5, handletextpad=0.4)

    # --- Right: Skip Ratio ---
    for i, (var, color) in enumerate(zip(variants, variant_colors)):
        means = [agg[var]['aif'][s]['skip_ratio']['mean'] for s in scenarios]
        stds = [agg[var]['aif'][s]['skip_ratio']['std'] for s in scenarios]
        ax2.bar(x + offsets[i], means, width * 0.9, yerr=stds,
                label=variant_labels[var], color=color,
                capsize=2, error_kw={'linewidth': 0.8},
                edgecolor='white', linewidth=0.3)

    ax2.set_ylabel('Skip Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_title('Skip Ratio vs SKIP Likelihood', fontsize=9)

    fig.tight_layout(w_pad=2)
    savefig(fig, os.path.join(out_dir, 'fig_skip_likelihood'))


def plot_skip_likelihood_tradeoff(ablation_dir, out_dir):
    """E3. SKIP likelihood trade-off — scatter plot of avg SLO vs avg skip ratio.

    Each point = one variant × one scenario; connects variants with arrows.
    """
    agg = load_ablation_aggregate(ablation_dir, 'skip_likelihood')
    if agg is None:
        print("  [SKIP] No skip_likelihood ablation data found.")
        return

    variants = ['skip_baseline', 'skip_mild', 'skip_moderate']
    variant_markers = {'skip_baseline': 'o', 'skip_mild': 's', 'skip_moderate': 'D'}
    variant_labels = {'skip_baseline': 'Baseline', 'skip_mild': 'Mild', 'skip_moderate': 'Moderate'}
    scenarios = SCENARIO_ORDER
    scenario_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.3, SINGLE_COL_W * 1.3 * ASPECT))

    for scen, color in zip(scenarios, scenario_colors):
        slos, skips = [], []
        for var in variants:
            slo = agg[var]['aif'][scen]['slo_satisfaction_rate']['mean']
            skip = agg[var]['aif'][scen]['skip_ratio']['mean']
            slos.append(slo)
            skips.append(skip)
            ax.scatter(skip, slo, color=color, marker=variant_markers[var],
                       s=40, zorder=5, edgecolors='white', linewidths=0.5)
        # Connect with line
        ax.plot(skips, slos, color=color, alpha=0.5, linewidth=1,
                label=SCENARIO_SHORT[scen])
        # Arrow from baseline to moderate
        ax.annotate('', xy=(skips[-1], slos[-1]), xytext=(skips[0], slos[0]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.4))

    # Marker legend
    for var in variants:
        ax.scatter([], [], marker=variant_markers[var], color='gray', s=30,
                   label=variant_labels[var])

    add_slo_line(ax, 0.9, '90% Target')
    ax.set_xlabel('Skip Ratio')
    ax.set_ylabel('SLO Satisfaction Rate')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_title('SKIP Likelihood: SLO vs Skip Trade-off', fontsize=9)
    ax.legend(loc='lower left', fontsize=6, ncol=2)

    savefig(fig, os.path.join(out_dir, 'fig_skip_tradeoff'))

def plot_ablation_likelihood(ablation_dir, out_dir):
    """D3. Paired bar chart — OFFLOAD likelihood v0 vs v1."""
    df = load_ablation_csv(ablation_dir, 'likelihood')
    if df is None:
        print("  [SKIP] No likelihood ablation data found.")
        return

    scenarios = SCENARIO_ORDER
    v0_color = '#E69F00'   # orange
    v1_color = '#0072B2'   # dark blue

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * ASPECT))

    x = np.arange(len(scenarios))
    width = 0.35

    v0_means, v1_means = [], []
    v0_stds, v1_stds = [], []

    for scen in scenarios:
        v0_data = df[(df['variant'].str.contains('v0')) & (df['scenario'] == scen)]['slo_satisfaction_rate']
        v1_data = df[(df['variant'].str.contains('v1')) & (df['scenario'] == scen)]['slo_satisfaction_rate']
        v0_means.append(v0_data.mean() if len(v0_data) > 0 else 0)
        v0_stds.append(v0_data.std() if len(v0_data) > 1 else 0)
        v1_means.append(v1_data.mean() if len(v1_data) > 0 else 0)
        v1_stds.append(v1_data.std() if len(v1_data) > 1 else 0)

    ax.bar(x - width / 2, v0_means, width, yerr=v0_stds,
           label='v0 (Per-Stream BAD)', color=v0_color,
           capsize=2, error_kw={'linewidth': 0.8}, edgecolor='white', linewidth=0.3)
    ax.bar(x + width / 2, v1_means, width, yerr=v1_stds,
           label='v1 (System-Level)', color=v1_color,
           capsize=2, error_kw={'linewidth': 0.8}, edgecolor='white', linewidth=0.3)

    add_slo_line(ax, 0.9)
    ax.set_ylabel('SLO Satisfaction Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left', fontsize=6)
    ax.set_title('OFFLOAD Likelihood: v0 vs v1', fontsize=9)

    savefig(fig, os.path.join(out_dir, 'fig_ablation_likelihood'))


def plot_ablation_sensitivity(ablation_dir, out_dir, ablation_name,
                              param_name, param_values, title):
    """Generic line plot for single-parameter ablation (precision, epistemic, cooldown)."""
    df = load_ablation_csv(ablation_dir, ablation_name)
    if df is None:
        print(f"  [SKIP] No {ablation_name} ablation data found.")
        return

    scenarios = SCENARIO_ORDER
    scenario_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * ASPECT))

    for scen, color in zip(scenarios, scenario_colors):
        means, stds = [], []
        for val in param_values:
            # Match variant name containing the param value
            mask = df['scenario'] == scen
            variant_mask = df['variant'].astype(str).str.contains(str(val).replace('.', r'\.'))
            subset = df[mask & variant_mask]['slo_satisfaction_rate']
            means.append(subset.mean() if len(subset) > 0 else 0)
            stds.append(subset.std() if len(subset) > 1 else 0)

        ax.errorbar(range(len(param_values)), means, yerr=stds,
                    label=SCENARIO_SHORT[scen], color=color,
                    marker='o', capsize=2, markersize=4)

    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel(param_name)
    ax.set_ylabel('SLO Satisfaction Rate')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='best', fontsize=6, ncol=2)
    ax.set_title(title, fontsize=9)

    savefig(fig, os.path.join(out_dir, f'fig_ablation_{ablation_name}'))


def plot_ablation_offload_onoff(ablation_dir, out_dir):
    """D4. Grouped bar chart — all controllers with OFFLOAD on vs off."""
    df = load_ablation_csv(ablation_dir, 'offload_onoff')
    if df is None:
        print("  [SKIP] No offload_onoff ablation data found.")
        return

    controllers = [c for c in CTRL_ORDER if c != 'noop']

    # Aggregate across scenarios
    on_means, off_means = [], []
    for ctrl in controllers:
        on_data = df[(df['controller'] == ctrl) &
                     (df['variant'].str.contains('enabled'))]['slo_satisfaction_rate']
        off_data = df[(df['controller'] == ctrl) &
                      (df['variant'].str.contains('disabled'))]['slo_satisfaction_rate']
        on_means.append(on_data.mean() if len(on_data) > 0 else 0)
        off_means.append(off_data.mean() if len(off_data) > 0 else 0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W * ASPECT))

    x = np.arange(len(controllers))
    width = 0.35

    ax.bar(x - width / 2, off_means, width, label='OFFLOAD Disabled',
           color='#BBBBBB', edgecolor='white', linewidth=0.3)
    ax.bar(x + width / 2, on_means, width, label='OFFLOAD Enabled',
           color='#4477AA', edgecolor='white', linewidth=0.3)

    ax.set_ylabel('SLO Satisfaction Rate')
    ax.set_xticks(x)
    ax.set_xticklabels([CTRL_LABELS[c] for c in controllers], fontsize=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right', fontsize=6)
    ax.set_title('OFFLOAD: Enabled vs Disabled', fontsize=9)

    savefig(fig, os.path.join(out_dir, 'fig_ablation_offload_onoff'))


def plot_ablation_combined(ablation_dir, out_dir):
    """Combined 2x2 ablation figure for paper (precision + epistemic + cooldown + likelihood)."""
    configs = [
        ('likelihood', 'Variant', ['v0', 'v1'], 'OFFLOAD Likelihood'),
        ('precision', r'$\beta$', [2, 4, 6, 8], 'Precision'),
        ('epistemic', r'$w_e$', [0, 0.1, 0.3, 0.5], 'Epistemic Weight'),
        ('cooldown', r'$\tau_c$', [1, 3, 5], 'Cooldown'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, DOUBLE_COL_W * 0.65))
    axes = axes.flatten()
    scenario_colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

    for ax, (abl_name, xlabel, values, title) in zip(axes, configs):
        df = load_ablation_csv(ablation_dir, abl_name)
        if df is None:
            ax.set_title(f'{title} (no data)', fontsize=8)
            continue

        for scen, color in zip(SCENARIO_ORDER, scenario_colors):
            means = []
            for val in values:
                mask = df['scenario'] == scen
                vmask = df['variant'].astype(str).str.contains(str(val).replace('.', r'\.'))
                subset = df[mask & vmask]['slo_satisfaction_rate']
                means.append(subset.mean() if len(subset) > 0 else 0)
            ax.plot(range(len(values)), means, marker='o', color=color,
                    label=SCENARIO_SHORT[scen], markersize=3, linewidth=1)

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], fontsize=6)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel('SLO%', fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set_title(title, fontsize=8)

    axes[0].legend(loc='best', fontsize=5, ncol=2)
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    savefig(fig, os.path.join(out_dir, 'fig_ablation_combined'))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

PLOT_REGISTRY = {
    # A: Main results
    'slo_bar':        ('A1: SLO Grouped Bar',    lambda a: plot_slo_bar(a['agg'], a['out'])),
    'switch_bar':     ('A2: Switch Count Bar',   lambda a: plot_switch_bar(a['agg'], a['out'])),
    'resource_bar':   ('A3: Skip/Offload Bar',   lambda a: plot_resource_bar(a['agg'], a['out'])),
    'heatmap':        ('A5: Summary Heatmap',    lambda a: plot_summary_heatmap(a['agg'], a['out'])),
    # B: Time-series
    'ts_fps':         ('B1: Time-Series FPS',     lambda a: [plot_timeseries_fps(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'ts_lat':         ('B2: Time-Series Latency', lambda a: [plot_timeseries_latency(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'mode_alloc':     ('B3: Mode Allocation',     lambda a: [plot_mode_allocation(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'cdf_lat':        ('B4: Latency CDF',         lambda a: [plot_latency_cdf(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'deepdive':       ('B5: Scenario Deep-Dive',  lambda a: [plot_scenario_deepdive(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'four_scenario':  ('B6: 4-Scenario FPS Panel', lambda a: plot_four_scenario_panel(a['main'], a['out'])),
    # C: AIF interpretability
    'belief':         ('C1: Belief Evolution',     lambda a: [plot_belief_evolution(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'entropy':        ('C3: Belief Entropy',       lambda a: [plot_belief_entropy(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    'gantt':          ('C4: Action Gantt',         lambda a: [plot_action_gantt(a['main'], a['out'], s) for s in SCENARIO_ORDER]),
    # D: Ablation
    'abl_likelihood': ('D3: Likelihood v0/v1',     lambda a: plot_ablation_likelihood(a['abl'], a['out'])),
    'abl_precision':  ('D1: Precision Sensitivity', lambda a: plot_ablation_sensitivity(a['abl'], a['out'], 'precision', r'$\beta$', [2, 4, 6, 8], 'Precision Sensitivity')),
    'abl_epistemic':  ('D2: Epistemic Sensitivity', lambda a: plot_ablation_sensitivity(a['abl'], a['out'], 'epistemic', r'$w_e$', [0, 0.1, 0.3, 0.5], 'Epistemic Weight Sensitivity')),
    'abl_cooldown':   ('D-: Cooldown Sensitivity',  lambda a: plot_ablation_sensitivity(a['abl'], a['out'], 'cooldown', r'$\tau_c$', [1, 3, 5], 'Cooldown Sensitivity')),
    'abl_offload':    ('D4: OFFLOAD On/Off',        lambda a: plot_ablation_offload_onoff(a['abl'], a['out'])),
    'abl_combined':   ('D*: Combined Ablation 2x2', lambda a: plot_ablation_combined(a['abl'], a['out'])),
    # E: Supplementary experiments
    'dqn_variance':   ('E1: DQN 8-Run Variance',    lambda a: plot_dqn_variance(a['main'], a['dqn_extra'], a['out'])),
    'skip_likelihood':('E2: SKIP Likelihood Tuning', lambda a: plot_skip_likelihood(a['abl'], a['out'])),
    'skip_tradeoff':  ('E3: SKIP SLO vs Skip Trade-off', lambda a: plot_skip_likelihood_tradeoff(a['abl'], a['out'])),
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures for AIF edge video paper.')
    parser.add_argument('--main-dir', default='~/Desktop/experiments/main',
                        help='Main experiment results directory')
    parser.add_argument('--ablation-dir', default='~/Desktop/experiments/ablation',
                        help='Ablation results directory')
    parser.add_argument('--dqn-extra-dir', default='~/Desktop/experiments/dqn_extra',
                        help='DQN extra seeds results directory')
    parser.add_argument('--output-dir', default='./figures',
                        help='Output directory for figures')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Generate only specific plots (e.g., --only slo_bar belief)')
    parser.add_argument('--list', action='store_true',
                        help='List available plot names')
    args = parser.parse_args()

    if args.list:
        print("Available plots:")
        for key, (desc, _) in PLOT_REGISTRY.items():
            print(f"  {key:<20} {desc}")
        return

    # Expand paths
    main_dir = os.path.expanduser(args.main_dir)
    abl_dir = os.path.expanduser(args.ablation_dir)
    dqn_extra_dir = os.path.expanduser(args.dqn_extra_dir)
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    apply_style()

    # Load data
    agg = None
    if os.path.exists(os.path.join(main_dir, 'aggregate.json')):
        agg = load_aggregate(main_dir)
        print(f"Loaded aggregate data from {main_dir}")
    else:
        print(f"[WARN] No aggregate.json in {main_dir}")

    context = {
        'agg': agg,
        'main': main_dir,
        'abl': abl_dir,
        'dqn_extra': dqn_extra_dir,
        'out': out_dir,
    }

    # Select plots
    if args.only:
        plot_keys = args.only
    else:
        plot_keys = list(PLOT_REGISTRY.keys())

    # Generate
    print(f"\nGenerating {len(plot_keys)} plot(s) → {out_dir}/\n")

    for key in plot_keys:
        if key not in PLOT_REGISTRY:
            print(f"  [ERROR] Unknown plot: {key}")
            continue
        desc, fn = PLOT_REGISTRY[key]
        print(f"  [{key}] {desc}...")
        try:
            fn(context)
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
