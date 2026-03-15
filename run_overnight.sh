#!/bin/bash
set -e

echo "=============================================="
echo "  OVERNIGHT EXPERIMENT SUITE"
echo "  Started: $(date)"
echo "=============================================="

echo ""
echo ">>> SMOKE TEST: 1 run, all controllers, 1 scenario"
echo ""
python3 run_multi_experiment.py --n-runs 1 --output-dir /data/experiments/smoke_test --controllers noop heuristic myopic dqn aif --scenarios scenario_ramp_up --train-episodes 1

echo ""
echo ">>> SMOKE TEST: AIF ablation (1 run, likelihood only, 1 scenario)"
echo ""
python3 run_ablation.py --n-runs 1 --output-dir /data/experiments/smoke_test_ablation --only likelihood

echo ""
echo "=== SMOKE TEST PASSED ==="
echo ""

echo ">>> PHASE 1: Main Experiment (5 runs)"
echo ""
python3 run_multi_experiment.py --n-runs 5 --output-dir /data/experiments/main --controllers noop heuristic myopic dqn aif --train-episodes 5

echo ""
echo ">>> PHASE 2: AIF Ablation Studies"
echo ""
python3 run_ablation.py --n-runs 3 --output-dir /data/experiments/ablation --only likelihood
python3 run_ablation.py --n-runs 3 --output-dir /data/experiments/ablation --only precision
python3 run_ablation.py --n-runs 3 --output-dir /data/experiments/ablation --only epistemic
python3 run_ablation.py --n-runs 3 --output-dir /data/experiments/ablation --only cooldown
python3 run_ablation.py --n-runs 1 --output-dir /data/experiments/ablation --only offload_onoff

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "=============================================="
