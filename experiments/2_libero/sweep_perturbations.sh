#!/bin/bash

sweep_type=$1    # object_shift ou action_noise
ckpt_dir=$2
ckpt_step=$3
num_trials=${4:-3}
task_ids=${5:-"0,1,2,3"}

host=localhost
port=$((10000 + RANDOM % 10000))
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

is_used() { lsof -i TCP:$1 >/dev/null 2>&1; }
while is_used "$port"; do port=$(( RANDOM % 10000 + 10000 )); done

SAVE_ROOT="${ckpt_dir}/results/checkpoint-${ckpt_step}_sweep_${sweep_type}_n${num_trials}"
mkdir -p "${SAVE_ROOT}"

# Démarrer le serveur PointACT une seule fois
server_python_bin=$HOME/miniconda3/envs/pointact/bin/python
${server_python_bin} scripts/run_server.py \
    --args.seed ${seed} \
    --args.pretrained_path ${ckpt_dir}/checkpoint-${ckpt_step} \
    --args.num_denoise_steps ${num_denoise_steps} \
    --args.host ${host} --args.port ${port} &

SERVER_PID=$!
echo "Server started (PID=$SERVER_PID, port=$port), waiting 25s..."
sleep 25

client_python_bin=$HOME/miniconda3/envs/libero/bin/python

run_point() {
    local extra_perturb_args=$1
    for cfg in "fixed 8" "fixed 1" "adaptive 8"; do
        read -r mode k <<< "$cfg"
        echo ">>> Sweep [${sweep_type}] ${extra_perturb_args} | mode=${mode} | k=${k}"
        ${client_python_bin} experiments/2_libero/run_libero_client_adaptive.py \
            --args.seed ${seed} \
            --args.host ${host} --args.port ${port} \
            --args.task_suite_name libero_spatial \
            --args.num_trials_per_task ${num_trials} \
            --args.task_ids "${task_ids}" \
            --args.pred_rot_type rot6d \
            --args.use_depth \
            --args.delta_action \
            --args.replan_steps ${k} \
            --args.adaptive_mode ${mode} \
            --args.perturb_type ${sweep_type} \
            --args.depth_ratio_thresh 0.010 \
            --args.cos_sim_thresh -0.10 \
            --args.cooldown_steps 3 \
            --args.save_dir "${SAVE_ROOT}" \
            ${extra_perturb_args}
    done
}

if [[ "${sweep_type}" == "object_shift" ]]; then
    for mag in 0.015 0.025 0.035 0.045; do
        run_point "--args.shift_mag ${mag}"
    done
elif [[ "${sweep_type}" == "action_noise" ]]; then
    for pair in "0.30 6" "0.60 10" "0.85 12" "1.00 15"; do
        read -r std dur <<< "$pair"
        run_point "--args.action_noise_std ${std} --args.action_noise_duration ${dur}"
    done
else
    echo "Unknown sweep_type: ${sweep_type} (use object_shift or action_noise)"
    exit 1
fi

echo "Sweep completed! Results saved in: ${SAVE_ROOT}"
