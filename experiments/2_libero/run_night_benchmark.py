import argparse
import csv
import datetime
import glob
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SCRIPT = REPO_ROOT / "experiments" / "2_libero" / "run_libero_client_adaptive.py"
DEFAULT_LIBERO_PYTHON = Path.home() / "miniconda3" / "envs" / "libero" / "bin" / "python"
DEFAULT_EXP_DIR = (
    Path.home()
    / "robot-PointAct/storage/datasets/PointAct_exprs/libero/pointact/"
    "VLAEncDec3DWithActionRegressionModel-concerto-data-libero-spatial-point-frontview-"
    "rot6d-image.frontview_ck16_lr5e-5_gpu1_bs32_epoch50-freeze.vlm"
)

# Régimes ordonnés par priorité scientifique
REGIMES = [
    {
        "name": "object_shift",
        "label": "Object Shift (3.5 cm)",
        "perturb_type": "object_shift",
        "shift_mag": 0.035,
        "action_noise_std": 0.0,
        "action_noise_duration": 6,
    },
    {
        "name": "combined",
        "label": "Combined (3.5 cm + std=0.60)",
        "perturb_type": "combined",
        "shift_mag": 0.035,
        "action_noise_std": 0.60,
        "action_noise_duration": 6,
    },
    {
        "name": "action_noise",
        "label": "Action Noise (std=0.85)",
        "perturb_type": "action_noise",
        "shift_mag": 0.0,
        "action_noise_std": 0.85,
        "action_noise_duration": 6,
    },
    {
        "name": "none",
        "label": "Nominal (Sans perturbation)",
        "perturb_type": "none",
        "shift_mag": 0.0,
        "action_noise_std": 0.0,
        "action_noise_duration": 6,
    },
]

# Modes ordonnés : les 3 rapides d'abord, Closed-Loop strict (k=1) en dernier
MODES = [
    {
        "name": "open_loop_k8",
        "label": "Open-Loop (k=8)",
        "adaptive_mode": "fixed",
        "replan_steps": 8,
        "enable_backtrack": False,
        "color": "#d62728",
    },
    {
        "name": "adaptive_only",
        "label": "Adaptive Horizon (k<=8)",
        "adaptive_mode": "adaptive",
        "replan_steps": 8,
        "enable_backtrack": False,
        "color": "#2ca02c",
    },
    {
        "name": "adaptive_backtrack",
        "label": "Adaptive + Backtrack (Ours)",
        "adaptive_mode": "adaptive",
        "replan_steps": 8,
        "enable_backtrack": True,
        "empty_grasp_width": 0.012,
        "retract_steps": 6,
        "max_backtracks": 2,
        "color": "#ff7f0e",
    },
    {
        "name": "closed_loop_k1",
        "label": "Closed-Loop strict (k=1)",
        "adaptive_mode": "fixed",
        "replan_steps": 1,
        "enable_backtrack": False,
        "color": "#1f77b4",
    },
]


def log_msg(msg: str, log_file: Path):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_server(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_latest_summary(run_dir: Path):
    files = sorted(glob.glob(str(run_dir / "*_summary.json")))
    return Path(files[-1]) if files else None


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def build_command(args, regime: dict, mode: dict, run_dir: Path) -> list:
    cmd = [
        str(args.libero_python),
        str(CLIENT_SCRIPT),
        "--args.seed", str(args.seed),
        "--args.host", args.host,
        "--args.port", str(args.port),
        "--args.task_suite_name", "libero_spatial",
        "--args.task_ids", args.task_ids,
        "--args.num_trials_per_task", str(args.num_trials),
        "--args.pred_rot_type", "rot6d",
        "--args.use_depth",
        "--args.delta_action",
        "--args.replan_steps", str(mode["replan_steps"]),
        "--args.adaptive_mode", mode["adaptive_mode"],
        "--args.perturb_type", regime["perturb_type"],
        "--args.shift_mag", str(regime["shift_mag"]),
        "--args.action_noise_std", str(regime["action_noise_std"]),
        "--args.action_noise_duration", str(regime["action_noise_duration"]),
        "--args.depth_ratio_thresh", "0.010",
        "--args.cos_sim_thresh", "-0.10",
        "--args.cooldown_steps", "3",
        "--args.save_dir", str(run_dir),
    ]
    if mode["enable_backtrack"]:
        cmd.extend([
            "--args.enable_backtrack",
            "--args.empty_grasp_width", str(mode["empty_grasp_width"]),
            "--args.retract_steps", str(mode["retract_steps"]),
            "--args.max_backtracks", str(mode["max_backtracks"]),
        ])
    return cmd


def aggregate_and_plot(out_root: Path, log_file: Path):
    rows = []
    results_grid = {}

    for regime in REGIMES:
        r_name = regime["name"]
        results_grid[r_name] = {}
        for mode in MODES:
            m_name = mode["name"]
            run_dir = out_root / r_name / m_name
            summary_path = find_latest_summary(run_dir)
            if not summary_path:
                continue
            data = load_summary(summary_path)
            sr = float(data.get("total_success_rate", 0.0)) * 100.0
            calls = float(data.get("avg_server_calls_per_episode", 0.0))
            steps = float(data.get("avg_steps_per_episode", 0.0))
            horizon = float(data.get("effective_horizon", 0.0))
            bt = float(data.get("avg_backtracks_per_episode", 0.0))
            eps = int(data.get("total_episodes", 0))
            wall = float(data.get("wall_time_sec", 0.0))

            entry = {
                "regime": r_name,
                "regime_label": regime["label"],
                "mode": m_name,
                "mode_label": mode["label"],
                "episodes": eps,
                "success_rate_pct": round(sr, 2),
                "avg_server_calls": round(calls, 2),
                "avg_steps": round(steps, 2),
                "effective_horizon": round(horizon, 2),
                "avg_backtracks": round(bt, 3),
                "wall_time_min": round(wall / 60.0, 2),
                "summary_path": str(summary_path),
            }
            rows.append(entry)
            results_grid[r_name][m_name] = entry

    if not rows:
        log_msg("Aucun summary.json trouvé pour générer les graphiques.", log_file)
        return

    # 1. Export CSV
    csv_path = out_root / "night_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2. Export tableau Markdown prêt pour le rapport
    md_path = out_root / "night_summary.md"
    lines = [
        "# Bilan Benchmark Nocturne — PointACT LIBERO-Spatial (10 tâches)\n",
        "| Régime | Mode | Épisodes | Succès (%) | Appels/ép | Horizon eff. | Pas/ép | Backtracks/ép | Durée (min) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['regime_label']} | {r['mode_label']} | {r['episodes']} | "
            f"**{r['success_rate_pct']:.1f}%** | {r['avg_server_calls']:.1f} | "
            f"{r['effective_horizon']:.2f} | {r['avg_steps']:.1f} | "
            f"{r['avg_backtracks']:.2f} | {r['wall_time_min']:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_msg(f"Tableaux exportés : {csv_path.name} & {md_path.name}", log_file)

    # 3. Graphique 1 : Barres comparatives (Taux de succès & Appels serveur par régime)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))
    x = np.arange(len(REGIMES))
    width = 0.19

    for idx, mode in enumerate(MODES):
        offset = (idx - 1.5) * width
        srs = [results_grid[r["name"]].get(mode["name"], {}).get("success_rate_pct", np.nan) for r in REGIMES]
        calls = [results_grid[r["name"]].get(mode["name"], {}).get("avg_server_calls", np.nan) for r in REGIMES]

        b1 = ax1.bar(x + offset, srs, width, label=mode["label"], color=mode["color"], alpha=0.9)
        b2 = ax2.bar(x + offset, calls, width, label=mode["label"], color=mode["color"], alpha=0.9)

        for rect, val in zip(b1, srs):
            if not np.isnan(val):
                ax1.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.0f}%",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for rect, val in zip(b2, calls):
            if not np.isnan(val):
                ax2.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.0f}",
                         ha="center", va="bottom", fontsize=7.5)

    xlabels = [r["label"] for r in REGIMES]
    ax1.set_title("Robustesse : Taux de succès par régime (10 tâches)")
    ax1.set_ylabel("Taux de succès (%)")
    ax1.set_ylim(0, 112)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=12, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8.5)

    ax2.set_title("Coût de calcul : Appels serveur moyens / épisode")
    ax2.set_ylabel("Appels serveur / épisode")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=12, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("PointACT LIBERO-Spatial — Comparaison Horizon Adaptatif & Backtrack", fontweight="bold")
    plt.tight_layout()
    bar_path = out_root / "night_benchmark_bars.png"
    plt.savefig(bar_path, dpi=180)
    plt.close(fig)

    # 4. Graphique 2 : Front de Pareto (Succès vs Appels serveur) pour chaque régime perturbé
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    perturbed = [r for r in REGIMES if r["name"] != "none"]
    for ax, regime in zip(axes, perturbed):
        r_name = regime["name"]
        for mode in MODES:
            entry = results_grid[r_name].get(mode["name"])
            if not entry:
                continue
            ax.scatter(
                entry["avg_server_calls"],
                entry["success_rate_pct"],
                s=110,
                color=mode["color"],
                label=mode["label"],
                zorder=3,
            )
            ax.annotate(
                f"{entry['success_rate_pct']:.1f}%",
                (entry["avg_server_calls"], entry["success_rate_pct"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8.5,
                fontweight="bold",
            )
        ax.set_title(regime["label"])
        ax.set_xlabel("Appels serveur / épisode (↓ mieux)")
        ax.set_ylabel("Taux de succès (%) (↑ mieux)")
        ax.set_ylim(0, 108)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig2.suptitle("Front de Pareto : Compromis Robustesse vs Coût d'inférence", fontweight="bold")
    plt.tight_layout()
    pareto_path = out_root / "night_pareto_efficiency.png"
    plt.savefig(pareto_path, dpi=180)
    plt.close(fig2)

    log_msg(f"Graphiques sauvegardés : {bar_path.name} & {pareto_path.name}", log_file)


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur benchmark nocturne PointACT LIBERO-Spatial")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-trials", type=int, default=8, help="Essais par tâche (8 = 80 ép/run, ~9h total)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXP_DIR / "results" / "checkpoint-32000_night_benchmark")
    parser.add_argument("--libero-python", type=Path, default=DEFAULT_LIBERO_PYTHON)
    parser.add_argument("--plot-only", action="store_true", help="Ne lance aucun run, regénère uniquement tableaux et graphes")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.out_dir / "night_benchmark.log"

    if args.plot_only:
        aggregate_and_plot(args.out_dir, log_file)
        return

    plan = [(r, m) for r in REGIMES for m in MODES]
    total_runs = len(plan)

    log_msg("=" * 72, log_file)
    log_msg(f"DÉMARRAGE BENCHMARK NOCTURNE | {total_runs} runs | tasks={args.task_ids} | trials/task={args.num_trials}", log_file)
    log_msg(f"Dossier de sortie : {args.out_dir}", log_file)
    log_msg("=" * 72, log_file)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"

    t_global = time.time()
    completed_durations = []

    for idx, (regime, mode) in enumerate(plan, start=1):
        run_dir = args.out_dir / regime["name"] / mode["name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        existing = find_latest_summary(run_dir)
        if existing is not None:
            data = load_summary(existing)
            log_msg(
                f"[{idx}/{total_runs}] SKIP (déjà fait) {regime['name']} / {mode['name']} -> "
                f"SR={data.get('total_success_rate', 0)*100:.1f}% | calls={data.get('avg_server_calls_per_episode', 0):.1f}",
                log_file,
            )
            continue

        if not check_server(args.host, args.port):
            log_msg(f"ERREUR : serveur PointACT injoignable sur {args.host}:{args.port}. Arrêt propre.", log_file)
            sys.exit(1)

        cmd = build_command(args, regime, mode, run_dir)
        log_msg(f"[{idx}/{total_runs}] START {regime['name']} / {mode['name']}", log_file)
        t_run = time.time()

        run_stdout_log = run_dir / "console_output.log"
        with open(run_stdout_log, "w", encoding="utf-8") as out_f:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=out_f, stderr=subprocess.STDOUT)

        dur = time.time() - t_run
        completed_durations.append(dur)

        if proc.returncode != 0:
            log_msg(
                f"[{idx}/{total_runs}] ÉCHEC (code={proc.returncode}) {regime['name']} / {mode['name']} "
                f"après {fmt_duration(dur)}. Voir {run_stdout_log}",
                log_file,
            )
        else:
            summary_path = find_latest_summary(run_dir)
            if summary_path:
                data = load_summary(summary_path)
                sr = data.get("total_success_rate", 0.0) * 100.0
                calls = data.get("avg_server_calls_per_episode", 0.0)
                bt = data.get("avg_backtracks_per_episode", 0.0)
                log_msg(
                    f"[{idx}/{total_runs}] DONE {regime['name']} / {mode['name']} en {fmt_duration(dur)} | "
                    f"SR={sr:.1f}% | calls/ep={calls:.1f} | bt/ep={bt:.2f}",
                    log_file,
                )
            else:
                log_msg(f"[{idx}/{total_runs}] DONE sans summary.json trouvé ({fmt_duration(dur)})", log_file)

        # Mise à jour intermédiaire des tableaux/graphiques après chaque run
        aggregate_and_plot(args.out_dir, log_file)

        remaining = total_runs - idx
        avg_dur = sum(completed_durations) / len(completed_durations)
        eta = avg_dur * remaining
        log_msg(
            f"Progression : {idx}/{total_runs} | Écoulé : {fmt_duration(time.time() - t_global)} | "
            f"ETA estimé : {fmt_duration(eta)}",
            log_file,
        )

    log_msg("=" * 72, log_file)
    log_msg(f"BENCHMARK NOCTURNE TERMINÉ en {fmt_duration(time.time() - t_global)}", log_file)
    log_msg("=" * 72, log_file)


if __name__ == "__main__":
    main()