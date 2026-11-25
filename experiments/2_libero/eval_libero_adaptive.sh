#!/bin/bash

task_suite_name=$1   # spatial, object, goal, 10
ckpt_dir=$2
ckpt_step=$3
pred_rot_type=$4     # rot6d, euler
perturb_type=$5      # object_shift, action_noise, combined
num_trials=$6        # ex: 3 (test rapide), 10 (étalonnage), 20 ou 50 (final)
extra_options=${7:-""} # ex: "--args.use_depth --args.delta_action --args.save_video"

host=localhost
port=$((10000 + RANDOM % 10000))
task_suite_full=libero_${task_suite_name}

seed=7
num_denoise_steps=10

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SCRATCH=${SCRATCH:-$(pwd)/storage}

set -euo pipefail

cleanup() {
  echo "[cleanup] stopping server..."
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "[cleanup] done"
}
trap cleanup INT TERM EXIT

is_used() {
    lsof -i TCP:$1 >/dev/null 2>&1
}
while is_used "$port"; do
    echo "Port $port in use, generating new..."
    port=$(( RANDOM % 10000 + 10000 ))
done
echo "port=$port"

SAVE_ROOT="${ckpt_dir}/results/checkpoint-${ckpt_step}_adaptive_${perturb_type}_n${num_trials}"
mkdir -p "${SAVE_ROOT}"

# 1. Start PointACT server ONCE
server_python_bin=$HOME/miniconda3/envs/pointact/bin/python

${server_python_bin} scripts/run_server.py \
    --args.seed ${seed} \
    --args.pretrained_path ${ckpt_dir}/checkpoint-${ckpt_step} \
    --args.num_denoise_steps ${num_denoise_steps} \
    --args.host ${host} --args.port ${port} &

SERVER_PID=$!
echo "Server started, PID=$SERVER_PID (waiting 25s for model load...)"
sleep 25

client_python_bin=$HOME/miniconda3/envs/libero/bin/python

run_eval() {
    local label=$1
    local adaptive_mode=$2
    local replan_steps=$3

    echo ""
    echo "=========================================================="
    echo ">>> Running [${label}] | perturb=${perturb_type} | mode=${adaptive_mode} | k=${replan_steps}"
    echo "=========================================================="

    ${client_python_bin} experiments/2_libero/run_libero_client_adaptive.py \
        --args.seed ${seed} \
        --args.host ${host} --args.port ${port} \
        --args.task_suite_name ${task_suite_full} \
        --args.num_trials_per_task ${num_trials} \
        --args.pred_rot_type ${pred_rot_type} \
        --args.replan_steps ${replan_steps} \
        --args.perturb_type ${perturb_type} \
        --args.adaptive_mode ${adaptive_mode} \
        --args.save_dir "${SAVE_ROOT}" \
        ${extra_options}
}

# 2. Run the 3 comparative configurations
run_eval "Baseline Perturbed (Open-Loop k=8)"   "fixed"    8
run_eval "Strict Closed-Loop (k=1)"             "fixed"    1
run_eval "Adaptive Horizon (Ours, k<=8)"        "adaptive" 8

# 3. Print clean comparison table from JSON summaries
echo ""
echo "================= COMPARATIVE RESULTS ================="
${client_python_bin} - "${SAVE_ROOT}" <<'EOF'
import glob, json, os, sys

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*_summary.json")))
print(f"{'Mode':<12} {'k':<4} {'SR (%)':<10} {'Calls/Ep':<12} {'Eff. H':<10} {'Early/Ep':<10} {'Time (s)':<10}")
print("-" * 74)
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    a = d["args"]
    print(
        f"{a['adaptive_mode']:<12} {a['replan_steps']:<4} "
        f"{d['total_success_rate']*100:<10.1f} "
        f"{d['avg_server_calls_per_episode']:<12.1f} "
        f"{d['effective_horizon']:<10.2f} "
        f"{d['avg_early_replans_per_episode']:<10.1f} "
        f"{d['wall_time_sec']:<10.1f}"
    )
EOF
echo "======================================================="
echo "Results saved in: ${SAVE_ROOT}"