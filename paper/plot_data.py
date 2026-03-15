"""
Data loading utilities for experiment results.
Reads summaries.csv, aggregate.json, per-step CSVs, and ablation files.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path


def load_summaries(experiment_dir):
    """Load summaries.csv as a pandas DataFrame."""
    path = os.path.join(experiment_dir, 'summaries.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"summaries.csv not found in {experiment_dir}")
    return pd.read_csv(path)


def load_aggregate(experiment_dir):
    """Load aggregate.json as nested dict."""
    path = os.path.join(experiment_dir, 'aggregate.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"aggregate.json not found in {experiment_dir}")
    with open(path) as f:
        return json.load(f)


def load_step_csv(experiment_dir, run_id, controller, scenario):
    """Load per-step CSV for a specific run/controller/scenario."""
    path = os.path.join(
        experiment_dir, 'histories', f'run{run_id}',
        f'{controller}_{scenario}_steps.csv'
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_history_json(experiment_dir, run_id, controller, scenario):
    """Load raw history JSON for a specific run/controller/scenario."""
    path = os.path.join(
        experiment_dir, 'histories', f'run{run_id}',
        f'{controller}_{scenario}.json'
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_ablation_csv(ablation_dir, ablation_name):
    """Load an ablation CSV (e.g., ablation_likelihood.csv)."""
    path = os.path.join(ablation_dir, f'ablation_{ablation_name}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_ablation_aggregate(ablation_dir, ablation_name):
    """Load an ablation aggregate JSON."""
    path = os.path.join(ablation_dir, f'ablation_{ablation_name}_aggregate.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_available_runs(experiment_dir):
    """Return list of available run IDs."""
    hist_dir = os.path.join(experiment_dir, 'histories')
    if not os.path.exists(hist_dir):
        return []
    runs = []
    for d in sorted(os.listdir(hist_dir)):
        if d.startswith('run') and os.path.isdir(os.path.join(hist_dir, d)):
            try:
                runs.append(int(d[3:]))
            except ValueError:
                pass
    return runs


def extract_per_stream_modes(history):
    """
    Extract per-stream mode at each step from raw history JSON.
    Returns dict: {step: {stream_id: mode}}.
    """
    result = {}
    for rec in history:
        step = rec['step']
        modes = {}
        for sid, ps in rec['observation']['per_stream'].items():
            modes[sid] = ps.get('current_mode', 'FULL')
        result[step] = modes
    return result


def extract_per_stream_metrics(history):
    """
    Extract per-stream FPS and latency at each step.
    Returns list of dicts with step, stream_id, fps, latency, mode.
    """
    rows = []
    for rec in history:
        step = rec['step']
        for sid, ps in rec['observation']['per_stream'].items():
            mode = ps.get('current_mode', 'FULL')
            fps = ps.get('infer_fps', ps.get('fps_avg', 0))
            lat = ps.get('infer_latency_p95_ms', ps.get('latency_p95_ms', 0))
            rows.append({
                'step': step,
                'stream_id': sid,
                'mode': mode,
                'fps': fps,
                'latency_ms': lat,
            })
    return pd.DataFrame(rows)
