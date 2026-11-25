#!/usr/bin/env python3
"""
Calibration automatique et tracé des courbes pour BacktrackController sous object_shift.

Enchaîne :
  - Sweep A : sensibilité / contraste de détection selon `empty_grasp_width`
              (à `retract_steps` fixé, par défaut 10).
  - Sweep B : durée de rétractation selon `retract_steps`
              (avec le meilleur `empty_grasp_width` trouvé au Sweep A ou une valeur fournie).

Note : le serveur PointACT (`scripts/run_server.py`) doit être lancé séparément avant exécution
(sauf en mode `--plot-only`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SCRIPT = REPO_ROOT / "experiments" / "2_libero" / "run_libero_client_adaptive.py"


class DualLogger:
    """Écrit simultanément dans le terminal (stdout) et dans un fichier log global."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def log(self, msg: str = "") -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}" if msg else ""
        print(line, flush=True)
        self._fp.write(line + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    secs = int(round(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Liste vide de flottants : {raw!r}")
    return vals


def parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Liste vide d'entiers : {raw!r}")
    return vals


def format_val_tag(val: float | int) -> str:
    if isinstance(val, int):
        return str(val)
    return f"{val:.6g}".replace(".", "p")


def detect_default_client_python() -> str:
    candidate = Path.home() / "miniconda3" / "envs" / "libero" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def check_server_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def extract_metrics(summary: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Extrait de façon robuste toutes les métriques utiles d'un *_summary.json."""
    a = summary.get("args", {})
    total_ep = int(summary.get("total_episodes", 0))
    total_succ = int(summary.get("total_successes", 0))
    total_fail = int(summary.get("total_failures", max(0, total_ep - total_succ)))

    sr = float(summary.get("total_success_rate", 0.0))
    avg_bt = float(summary.get("avg_backtracks_per_episode", 0.0))
    avg_steps = float(summary.get("avg_steps_per_episode", 0.0))
    avg_calls = float(summary.get("avg_server_calls_per_episode", 0.0))
    eff_h = float(summary.get("effective_horizon", avg_steps / max(1e-6, avg_calls)))
    wall_time = float(summary.get("wall_time_sec", 0.0))

    # Récupération ou recalcul des épisodes échoués sans backtrack
    failed_no_bt = summary.get("failed_episodes_without_backtrack")
    failed_no_bt_rate = summary.get("failed_without_backtrack_rate")
    failed_no_bt_share = summary.get("failed_without_backtrack_share_of_failures")
    bt_recovery_rate = summary.get("backtrack_recovery_rate")

    if failed_no_bt is None and isinstance(summary.get("per_episode"), list):
        eps = summary["per_episode"]
        failed_no_bt = sum(
            1 for ep in eps if (not ep.get("success", False)) and int(ep.get("backtracks", 0)) == 0
        )
        ep_with_bt = sum(1 for ep in eps if int(ep.get("backtracks", 0)) > 0)
        succ_with_bt = sum(
            1 for ep in eps if ep.get("success", False) and int(ep.get("backtracks", 0)) > 0
        )
        failed_no_bt_rate = float(failed_no_bt) / float(max(1, len(eps)))
        failed_no_bt_share = float(failed_no_bt) / float(total_fail) if total_fail > 0 else 0.0
        bt_recovery_rate = float(succ_with_bt) / float(ep_with_bt) if ep_with_bt > 0 else 0.0

    return {
        "source_path": str(source_path),
        "empty_grasp_width": float(a.get("empty_grasp_width", 0.0)),
        "retract_steps": int(a.get("retract_steps", 10)),
        "close_confirm_steps": int(a.get("close_confirm_steps", 8)),
        "max_backtracks": int(a.get("max_backtracks", 2)),
        "shift_mag": float(a.get("shift_mag", 0.035)),
        "total_episodes": total_ep,
        "total_successes": total_succ,
        "total_failures": total_fail,
        "total_success_rate": sr,
        "sr_pct": sr * 100.0,
        "avg_backtracks_per_episode": avg_bt,
        "avg_steps_per_episode": avg_steps,
        "avg_server_calls_per_episode": avg_calls,
        "effective_horizon": eff_h,
        "failed_episodes_without_backtrack": failed_no_bt,
        "failed_without_backtrack_rate": failed_no_bt_rate,
        "failed_without_backtrack_pct": (
            failed_no_bt_rate * 100.0 if failed_no_bt_rate is not None else None
        ),
        "failed_without_backtrack_share_of_failures": failed_no_bt_share,
        "failed_without_backtrack_share_pct": (
            failed_no_bt_share * 100.0 if failed_no_bt_share is not None else None
        ),
        "backtrack_recovery_rate": bt_recovery_rate,
        "replan_reasons": summary.get("replan_reasons", {}),
        "wall_time_sec": wall_time,
    }


def load_latest_summary(folder: Path) -> tuple[Path, dict[str, Any]] | None:
    files = sorted(
        folder.glob("*_summary.json"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not files:
        return None
    latest = files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        return latest, json.load(f)


def aggregate_summaries_by_param(
    sweep_dir: Path,
    param_key: str,
    logger: DualLogger | None = None,
) -> list[dict[str, Any]]:
    """
    Agrège tous les *_summary.json sous `sweep_dir` sans doublon sur une même abscisse.
    Si plusieurs fichiers partagent exactement la même valeur de `param_key`,
    le plus récent (mtime / nom) est conservé.
    """
    if not sweep_dir.exists():
        return []

    all_files = sorted(
        sweep_dir.rglob("*_summary.json"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    by_x: dict[float | int, tuple[Path, dict[str, Any]]] = {}
    dup_counts: dict[float | int, int] = {}

    for fp in all_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            if logger:
                logger.log(f"[WARN] Impossible de lire {fp}: {exc}")
            continue

        metrics = extract_metrics(data, fp)
        raw_x = metrics[param_key]
        norm_x: float | int = int(raw_x) if param_key == "retract_steps" else round(float(raw_x), 6)

        if norm_x in by_x:
            dup_counts[norm_x] = dup_counts.get(norm_x, 1) + 1
        by_x[norm_x] = (fp, metrics)

    if dup_counts and logger:
        for x_val, cnt in sorted(dup_counts.items()):
            logger.log(
                f"[INFO] Déduplication ({param_key}={x_val}) : {cnt} fichiers trouvés, "
                f"conservation du plus récent ({by_x[x_val][0].name})."
            )

    return [by_x[k][1] for k in sorted(by_x.keys())]


def compute_metric_contrast(
    xs: list[float],
    ys: list[float | None],
    unit: str = "",
) -> dict[str, Any] | None:
    """Calcule min, max, amplitude (max - min), ratio (max / min) et deltas consécutifs."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return None

    min_x, min_y = min(pairs, key=lambda p: p[1])
    max_x, max_y = max(pairs, key=lambda p: p[1])
    amplitude = max_y - min_y

    if min_y > 0:
        ratio: float | None = max_y / min_y
        ratio_str = f"{ratio:.2f}x"
    elif max_y == 0:
        ratio = 1.0
        ratio_str = "1.00x (tous à 0)"
    else:
        ratio = None
        ratio_str = "inf (min=0)"

    consec = []
    for (x1, y1), (x2, y2) in zip(pairs[:-1], pairs[1:]):
        consec.append(
            {
                "from_x": x1,
                "to_x": x2,
                "delta": y2 - y1,
            }
        )

    return {
        "unit": unit,
        "min": min_y,
        "min_at": min_x,
        "max": max_y,
        "max_at": max_x,
        "amplitude": amplitude,
        "ratio": ratio,
        "ratio_str": ratio_str,
        "consecutive_diffs": consec,
    }


def compute_sweep_a_contrast(records: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [r["empty_grasp_width"] for r in records]
    return {
        "x_values": xs,
        "success_rate_pct": compute_metric_contrast(
            xs, [r["sr_pct"] for r in records], unit="%"
        ),
        "avg_backtracks_per_episode": compute_metric_contrast(
            xs, [r["avg_backtracks_per_episode"] for r in records], unit="bt/ep"
        ),
        "failed_without_backtrack_pct": compute_metric_contrast(
            xs, [r["failed_without_backtrack_pct"] for r in records], unit="% ep"
        ),
        "failed_without_backtrack_share_pct": compute_metric_contrast(
            xs, [r["failed_without_backtrack_share_pct"] for r in records], unit="% échecs"
        ),
    }


def log_sweep_a_summary_and_contrast(
    records: list[dict[str, Any]],
    contrast: dict[str, Any],
    best_record: dict[str, Any] | None,
    logger: DualLogger,
) -> None:
    logger.log("")
    logger.log("================ SWEEP A : RÉSULTATS (empty_grasp_width) ================")
    header = (
        f"{'width (m)':<11} {'SR (%)':<9} {'BT/ep':<8} {'Fail(bt=0)%':<13} "
        f"{'Fail(bt=0)/Fail':<17} {'Calls/ep':<10} {'Steps/ep':<10} {'Time':<9}"
    )
    logger.log(header)
    logger.log("-" * len(header))
    for r in records:
        f_ep = (
            f"{r['failed_without_backtrack_pct']:.1f}%"
            if r["failed_without_backtrack_pct"] is not None
            else "N/A"
        )
        f_sh = (
            f"{r['failed_without_backtrack_share_pct']:.1f}% ({r['failed_episodes_without_backtrack']}/{r['total_failures']})"
            if r["failed_without_backtrack_share_pct"] is not None
            else "N/A"
        )
        star = " *" if best_record and math.isclose(r["empty_grasp_width"], best_record["empty_grasp_width"], rel_tol=0, abs_tol=1e-9) else ""
        logger.log(
            f"{r['empty_grasp_width']:<11.4f} {r['sr_pct']:<9.1f} "
            f"{r['avg_backtracks_per_episode']:<8.2f} {f_ep:<13} "
            f"{f_sh:<17} {r['avg_server_calls_per_episode']:<10.1f} "
            f"{r['avg_steps_per_episode']:<10.1f} {format_duration(r['wall_time_sec']):<9}{star}"
        )

    logger.log("")
    logger.log("---- CONTRASTE EN FONCTION DE empty_grasp_width ----")
    labels = [
        ("success_rate_pct", "Taux de succès (%)"),
        ("avg_backtracks_per_episode", "Backtracks moyens / épisode"),
        ("failed_without_backtrack_pct", "Échecs sans backtrack (% des épisodes)"),
        ("failed_without_backtrack_share_pct", "Échecs sans backtrack (% des échecs)"),
    ]
    for key, label in labels:
        c = contrast.get(key)
        if not c:
            logger.log(f"  * {label}: N/A")
            continue
        consec_txt = ", ".join(
            f"{d['from_x']:.4g}->{d['to_x']:.4g}: {d['delta']:+.2f}"
            for d in c["consecutive_diffs"]
        )
        logger.log(
            f"  * {label}: "
            f"min={c['min']:.2f} (w={c['min_at']:.4g}), "
            f"max={c['max']:.2f} (w={c['max_at']:.4g}), "
            f"amplitude(max-min)={c['amplitude']:.2f} {c['unit']}, "
            f"ratio={c['ratio_str']}"
        )
        if consec_txt:
            logger.log(f"      Deltas consécutifs : {consec_txt}")

    if best_record is not None:
        logger.log(
            f"--> Meilleur empty_grasp_width sélectionné : {best_record['empty_grasp_width']:.4g} "
            f"(SR={best_record['sr_pct']:.1f}%, BT/ep={best_record['avg_backtracks_per_episode']:.2f}, "
            f"Calls/ep={best_record['avg_server_calls_per_episode']:.1f})"
        )
    logger.log("=========================================================================")


def log_sweep_b_summary(
    records: list[dict[str, Any]],
    width_used: float,
    logger: DualLogger,
) -> None:
    logger.log("")
    logger.log(
        f"========== SWEEP B : RÉSULTATS (retract_steps, width={width_used:.4g}) =========="
    )
    header = (
        f"{'retract':<9} {'SR (%)':<9} {'Steps/ep':<10} {'Calls/ep':<10} "
        f"{'BT/ep':<8} {'Fail(bt=0)%':<13} {'Eff.H':<8} {'Time':<9}"
    )
    logger.log(header)
    logger.log("-" * len(header))
    for r in records:
        f_ep = (
            f"{r['failed_without_backtrack_pct']:.1f}%"
            if r["failed_without_backtrack_pct"] is not None
            else "N/A"
        )
        logger.log(
            f"{r['retract_steps']:<9d} {r['sr_pct']:<9.1f} "
            f"{r['avg_steps_per_episode']:<10.1f} {r['avg_server_calls_per_episode']:<10.1f} "
            f"{r['avg_backtracks_per_episode']:<8.2f} {f_ep:<13} "
            f"{r['effective_horizon']:<8.2f} {format_duration(r['wall_time_sec']):<9}"
        )
    logger.log("==========================================================================")


def select_best_width_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Sélectionne le meilleur `empty_grasp_width` :
      1) Taux de succès maximal
      2) Part d'échecs sans backtrack minimale (en cas d'égalité)
      3) Appels serveur moyens minimaux
      4) Nombre moyen de steps minimal
    """
    if not records:
        return None
    return max(
        records,
        key=lambda r: (
            round(r["total_success_rate"], 6),
            -(r["failed_without_backtrack_rate"] if r["failed_without_backtrack_rate"] is not None else 1.0),
            -r["avg_server_calls_per_episode"],
            -r["avg_steps_per_episode"],
        ),
    )


def plot_sweep_a_detection(
    records: list[dict[str, Any]],
    contrast: dict[str, Any],
    best_record: dict[str, Any] | None,
    out_path: Path,
    logger: DualLogger,
) -> None:
    if not records:
        logger.log("[WARN] Aucun résultat Sweep A à tracer.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["empty_grasp_width"] for r in records]
    srs = [r["sr_pct"] for r in records]
    bts = [r["avg_backtracks_per_episode"] for r in records]
    fail_ep = [r["failed_without_backtrack_pct"] for r in records]
    fail_sh = [r["failed_without_backtrack_share_pct"] for r in records]
    retract_fixed = records[0]["retract_steps"]
    shift_cm = records[0]["shift_mag"] * 100.0
    n_ep = records[0]["total_episodes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.suptitle(
        f"Sweep A — Sensibilité & contraste de détection (object_shift={shift_cm:.1f} cm, "
        f"retract_steps={retract_fixed}, N={n_ep} ép./pt)",
        fontsize=12,
        fontweight="bold",
    )

    # --- Panneau gauche : Taux de succès (%) + Backtracks moyens / épisode ---
    l1 = ax1.plot(
        xs, srs, "o-", color="#2ca02c", linewidth=2.2, markersize=7, label="Taux de succès (%)"
    )
    for x, y in zip(xs, srs):
        ax1.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color="#1b7a1b",
            fontweight="bold",
        )

    if best_record is not None:
        ax1.axvline(
            best_record["empty_grasp_width"],
            color="#2ca02c",
            linestyle=":",
            alpha=0.55,
            linewidth=1.4,
        )

    ax1.set_title("Performance & déclenchement du Backtrack")
    ax1.set_xlabel("empty_grasp_width (m)")
    ax1.set_ylabel("Taux de succès (%)", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax1.set_ylim(-5, 110)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{x:.4g}" for x in xs])
    ax1.grid(True, alpha=0.3)

    ax1_r = ax1.twinx()
    l2 = ax1_r.plot(
        xs,
        bts,
        "s--",
        color="#ff7f0e",
        linewidth=2.0,
        markersize=6.5,
        label="Backtracks moyens / épisode",
    )
    for x, y in zip(xs, bts):
        ax1_r.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            color="#c85a00",
        )
    ax1_r.set_ylabel("Backtracks moyens / épisode", color="#ff7f0e")
    ax1_r.tick_params(axis="y", labelcolor="#ff7f0e")
    max_bt = max(bts) if bts else 1.0
    ax1_r.set_ylim(-0.08 * max(1.0, max_bt), max(0.5, max_bt * 1.32))

    lines_1 = l1 + l2
    labels_1 = [ln.get_label() for ln in lines_1]
    ax1.legend(lines_1, labels_1, loc="lower left", fontsize=8.5, framealpha=0.92)

    # --- Panneau droit : Erreurs non détectées (échecs avec backtracks == 0) + encart contraste ---
    has_fail_ep = all(v is not None for v in fail_ep)
    has_fail_sh = all(v is not None for v in fail_sh)

    if has_fail_ep:
        ax2.plot(
            xs,
            fail_ep,
            "^-",
            color="#d62728",
            linewidth=2.2,
            markersize=7,
            label="Échecs sans backtrack (% épisodes)",
        )
        for x, y in zip(xs, fail_ep):
            ax2.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                color="#a81d1d",
                fontweight="bold",
            )
    if has_fail_sh:
        ax2.plot(
            xs,
            fail_sh,
            "D--",
            color="#9467bd",
            linewidth=1.9,
            markersize=6,
            label="Échecs sans backtrack (% des échecs)",
        )
        for x, y in zip(xs, fail_sh):
            ax2.annotate(
                f"{y:.0f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=8,
                color="#6a3d9a",
            )

    if not has_fail_ep and not has_fail_sh:
        ax2.text(
            0.5,
            0.5,
            "Métrique échecs sans backtrack\nindisponible dans ces JSON",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
        )

    ax2.set_title("Détection manquée / tardive (échecs avec backtracks = 0)")
    ax2.set_xlabel("empty_grasp_width (m)")
    ax2.set_ylabel("Proportion (%)")
    ax2.set_ylim(-5, 112)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{x:.4g}" for x in xs])
    ax2.grid(True, alpha=0.3)
    if has_fail_ep or has_fail_sh:
        ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # Encart résumé du contraste observé
    c_sr = contrast.get("success_rate_pct")
    c_bt = contrast.get("avg_backtracks_per_episode")
    c_fn = contrast.get("failed_without_backtrack_pct")
    summary_lines = ["Contraste observé (max - min | ratio) :"]
    if c_sr:
        summary_lines.append(f"• SR : Δ={c_sr['amplitude']:.1f} pts ({c_sr['ratio_str']})")
    if c_bt:
        summary_lines.append(f"• BT/ep : Δ={c_bt['amplitude']:.2f} ({c_bt['ratio_str']})")
    if c_fn:
        summary_lines.append(f"• Échecs bt=0 : Δ={c_fn['amplitude']:.1f} pts ({c_fn['ratio_str']})")
    if best_record is not None:
        summary_lines.append(f"• Best width : {best_record['empty_grasp_width']:.4g} m")

    ax2.text(
        0.03,
        0.04,
        "\n".join(summary_lines),
        transform=ax2.transAxes,
        fontsize=8.2,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#bbbbbb", alpha=0.92),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.log(f"Graphique Sweep A sauvegardé : {out_path}")


def plot_sweep_b_retract(
    records: list[dict[str, Any]],
    out_path: Path,
    logger: DualLogger,
) -> None:
    if not records:
        logger.log("[WARN] Aucun résultat Sweep B à tracer.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["retract_steps"] for r in records]
    srs = [r["sr_pct"] for r in records]
    steps = [r["avg_steps_per_episode"] for r in records]
    calls = [r["avg_server_calls_per_episode"] for r in records]
    width_used = records[0]["empty_grasp_width"]
    shift_cm = records[0]["shift_mag"] * 100.0
    n_ep = records[0]["total_episodes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    fig.suptitle(
        f"Sweep B — Durée de rétractation (object_shift={shift_cm:.1f} cm, "
        f"empty_grasp_width={width_used:.4g} m, N={n_ep} ép./pt)",
        fontsize=12,
        fontweight="bold",
    )

    # --- Panneau gauche : Taux de succès (%) ---
    ax1.plot(
        xs, srs, "o-", color="#2ca02c", linewidth=2.2, markersize=7.5, label="Taux de succès (%)"
    )
    for x, y in zip(xs, srs):
        ax1.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.8,
            color="#1b7a1b",
            fontweight="bold",
        )
    ax1.set_title("Taux de succès vs retract_steps")
    ax1.set_xlabel("retract_steps")
    ax1.set_ylabel("Taux de succès (%)")
    ax1.set_ylim(-5, 110)
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # --- Panneau droit : avg_steps_per_episode + avg_server_calls_per_episode ---
    l1 = ax2.plot(
        xs,
        steps,
        "s-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        label="Steps moyens / épisode",
    )
    for x, y in zip(xs, steps):
        ax2.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
            color="#15578a",
            fontweight="bold",
        )
    ax2.set_title("Coût temporel & appels serveur vs retract_steps")
    ax2.set_xlabel("retract_steps")
    ax2.set_ylabel("Steps moyens / épisode", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_xticks(xs)
    ax2.grid(True, alpha=0.3)
    if steps:
        span_s = max(steps) - min(steps)
        pad_s = max(10.0, span_s * 0.35)
        ax2.set_ylim(max(0.0, min(steps) - pad_s), max(steps) + pad_s * 1.2)

    ax2_r = ax2.twinx()
    l2 = ax2_r.plot(
        xs,
        calls,
        "^--",
        color="#d62728",
        linewidth=2.0,
        markersize=7,
        label="Appels serveur moyens / épisode",
    )
    for x, y in zip(xs, calls):
        ax2_r.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            color="#a81d1d",
        )
    ax2_r.set_ylabel("Appels serveur moyens / épisode", color="#d62728")
    ax2_r.tick_params(axis="y", labelcolor="#d62728")
    if calls:
        span_c = max(calls) - min(calls)
        pad_c = max(2.0, span_c * 0.45)
        ax2_r.set_ylim(max(0.0, min(calls) - pad_c), max(calls) + pad_c * 1.2)

    lines_2 = l1 + l2
    labels_2 = [ln.get_label() for ln in lines_2]
    ax2.legend(lines_2, labels_2, loc="upper left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.log(f"Graphique Sweep B sauvegardé : {out_path}")


def run_single_eval(
    args: argparse.Namespace,
    *,
    sweep_label: str,
    step_idx: int,
    step_total: int,
    global_idx: int,
    global_total: int,
    empty_grasp_width: float,
    retract_steps: int,
    run_save_dir: Path,
    sweep_start_wall: float,
    completed_run_durations: list[float],
    logger: DualLogger,
) -> dict[str, Any]:
    """Lance une évaluation client dans un sous-dossier dédié et renvoie ses métriques."""
    run_save_dir.mkdir(parents=True, exist_ok=True)

    # Option --resume : réutilise le résumé existant dans le sous-dossier si présent
    if args.resume:
        existing = load_latest_summary(run_save_dir)
        if existing is not None:
            fp, data = existing
            metrics = extract_metrics(data, fp)
            logger.log(
                f"[{sweep_label} | {step_idx}/{step_total} | Global {global_idx}/{global_total}] "
                f"SKIP (--resume) width={empty_grasp_width:.4g}, retract={retract_steps} | "
                f"SR={metrics['sr_pct']:.1f}% | BT/ep={metrics['avg_backtracks_per_episode']:.2f} | "
                f"Calls/ep={metrics['avg_server_calls_per_episode']:.1f} ({fp.name})"
            )
            return metrics

    cmd = [
        args.client_python,
        str(CLIENT_SCRIPT),
        "--args.seed", str(args.seed),
        "--args.host", args.host,
        "--args.port", str(args.port),
        "--args.task_suite_name", args.task_suite_name,
        "--args.task_ids", args.task_ids,
        "--args.num_trials_per_task", str(args.num_trials_per_task),
        "--args.pred_rot_type", args.pred_rot_type,
        "--args.replan_steps", str(args.replan_steps),
        "--args.adaptive_mode", args.adaptive_mode,
        "--args.perturb_type", args.perturb_type,
        "--args.shift_mag", str(args.shift_mag),
        "--args.depth_pixel_thresh", str(args.depth_pixel_thresh),
        "--args.depth_ratio_thresh", str(args.depth_ratio_thresh),
        "--args.depth_spike_factor", str(args.depth_spike_factor),
        "--args.cos_sim_thresh", str(args.cos_sim_thresh),
        "--args.cooldown_steps", str(args.cooldown_steps),
        "--args.enable_backtrack",
        "--args.empty_grasp_width", str(empty_grasp_width),
        "--args.close_confirm_steps", str(args.close_confirm_steps),
        "--args.retract_steps", str(retract_steps),
        "--args.max_backtracks", str(args.max_backtracks),
        "--args.save_dir", str(run_save_dir),
    ]
    if args.use_depth:
        cmd.append("--args.use_depth")
    if args.delta_action:
        cmd.append("--args.delta_action")
    if args.remove_arm:
        cmd.append("--args.remove_arm")
    if args.save_video:
        cmd.append("--args.save_video")
    if args.verbose_client:
        cmd.append("--args.verbose")

    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_py_path}" if existing_py_path else str(REPO_ROOT)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("SCRATCH", str(REPO_ROOT / "storage"))

    eta_before = (
        (sum(completed_run_durations) / len(completed_run_durations)) * (global_total - global_idx + 1)
        if completed_run_durations
        else None
    )
    logger.log(
        f"[{sweep_label} | {step_idx}/{step_total} | Global {global_idx}/{global_total}] "
        f"START width={empty_grasp_width:.4g}, retract={retract_steps}, "
        f"close_confirm={args.close_confirm_steps}, max_bt={args.max_backtracks} | "
        f"elapsed={format_duration(time.time() - sweep_start_wall)} | "
        f"ETA={format_duration(eta_before)} | save_dir={run_save_dir}"
    )

    t0 = time.time()
    runner_log_path = run_save_dir / "client_stdout.log"
    with open(runner_log_path, "w", encoding="utf-8") as sub_log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sub_log.write(line)
            if args.stream_client_logs:
                sys.stdout.write(line)
                sys.stdout.flush()
        ret = proc.wait()

    run_dur = time.time() - t0
    if ret != 0:
        logger.log(
            f"[ERROR] Échec du client (code={ret}) pour width={empty_grasp_width}, "
            f"retract={retract_steps}. Voir {runner_log_path}"
        )
        raise subprocess.CalledProcessError(ret, cmd)

    completed_run_durations.append(run_dur)
    elapsed_total = time.time() - sweep_start_wall
    remaining_runs = max(0, global_total - global_idx)
    avg_run_dur = sum(completed_run_durations) / len(completed_run_durations)
    eta_after = avg_run_dur * remaining_runs

    loaded = load_latest_summary(run_save_dir)
    if loaded is None:
        raise FileNotFoundError(
            f"Aucun *_summary.json généré dans {run_save_dir} (voir {runner_log_path})."
        )
    summary_fp, summary_data = loaded
    metrics = extract_metrics(summary_data, summary_fp)

    fail_no_bt_str = (
        f"{metrics['failed_without_backtrack_pct']:.1f}% "
        f"({metrics['failed_episodes_without_backtrack']}/{metrics['total_episodes']})"
        if metrics["failed_without_backtrack_pct"] is not None
        else "N/A"
    )
    logger.log(
        f"[{sweep_label} | {step_idx}/{step_total} | Global {global_idx}/{global_total}] "
        f"DONE  width={empty_grasp_width:.4g}, retract={retract_steps} | "
        f"run={format_duration(run_dur)} | elapsed={format_duration(elapsed_total)} | "
        f"ETA={format_duration(eta_after)} | SR={metrics['sr_pct']:.1f}% | "
        f"BT/ep={metrics['avg_backtracks_per_episode']:.2f} | "
        f"Calls/ep={metrics['avg_server_calls_per_episode']:.1f} | "
        f"Steps/ep={metrics['avg_steps_per_episode']:.1f} | "
        f"Fail(bt=0)={fail_no_bt_str}"
    )
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatise la calibration de BacktrackController (Sweep A: empty_grasp_width, "
            "Sweep B: retract_steps) sous object_shift et génère les graphiques (dpi=180)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sorties & modes d'exécution
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help=(
            "Dossier racine de sauvegarde (sous-dossiers sweep_a/, sweep_b/, log global, PNG). "
            "Par défaut : results/backtrack_sweep_<horodatage>."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Ne lance aucun client : agrège les *_summary.json existants dans --save-root et génère les graphiques.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les points qui possèdent déjà un *_summary.json dans leur sous-dossier dédié.",
    )
    parser.add_argument(
        "--skip-sweep-a",
        action="store_true",
        help="Ignore le Sweep A (utilise --best-width ou les résultats Sweep A déjà présents dans --save-root).",
    )
    parser.add_argument(
        "--skip-sweep-b",
        action="store_true",
        help="Ignore le Sweep B.",
    )

    # Listes de valeurs des sweeps
    parser.add_argument(
        "--widths",
        type=str,
        default="0.005,0.008,0.012,0.016",
        help="Valeurs de empty_grasp_width (m) pour le Sweep A, séparées par des virgules.",
    )
    parser.add_argument(
        "--sweep-a-retract-steps",
        type=int,
        default=10,
        help="Valeur fixe de retract_steps pendant le Sweep A.",
    )
    parser.add_argument(
        "--retract-steps",
        type=str,
        default="6,10,14",
        help="Valeurs de retract_steps pour le Sweep B, séparées par des virgules.",
    )
    parser.add_argument(
        "--best-width",
        type=float,
        default=None,
        help=(
            "Force la valeur de empty_grasp_width utilisée pour le Sweep B "
            "(si omis, sélectionne automatiquement la meilleure du Sweep A)."
        ),
    )

    # Budget temps & échantillonnage (objectif ~1h max par défaut : 4 tâches x 5 essais = 20 ép./pt)
    parser.add_argument("--task-ids", type=str, default="0,1,2,3", help="Sous-ensemble d'IDs de tâches LIBERO.")
    parser.add_argument("--num-trials-per-task", type=int, default=5, help="Nombre de répétitions par tâche.")
    parser.add_argument("--seed", type=int, default=7, help="Graine aléatoire.")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial", help="Suite LIBERO.")

    # Perturbation & horizon adaptatif
    parser.add_argument("--perturb-type", type=str, default="object_shift", help="Type de perturbation.")
    parser.add_argument("--shift-mag", type=float, default=0.035, help="Amplitude du déplacement d'objet (m).")
    parser.add_argument("--adaptive-mode", type=str, default="adaptive", help="Mode d'horizon (adaptive/fixed).")
    parser.add_argument("--replan-steps", type=int, default=8, help="Horizon nominal k.")
    parser.add_argument("--depth-pixel-thresh", type=float, default=0.025)
    parser.add_argument("--depth-ratio-thresh", type=float, default=0.010)
    parser.add_argument("--depth-spike-factor", type=float, default=2.5)
    parser.add_argument("--cos-sim-thresh", type=float, default=-0.10)
    parser.add_argument("--cooldown-steps", type=int, default=3)

    # Paramètres BacktrackController fixes
    parser.add_argument("--close-confirm-steps", type=int, default=8, help="Steps de confirmation de fermeture.")
    parser.add_argument("--max-backtracks", type=int, default=2, help="Nombre max de backtracks par épisode.")

    # Connexion au serveur & options client PointACT
    parser.add_argument("--host", type=str, default="localhost", help="Hôte du serveur PointACT.")
    parser.add_argument("--port", type=int, default=5555, help="Port du serveur PointACT.")
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Ne vérifie pas l'accessibilité TCP du serveur avant de lancer le client.",
    )
    parser.add_argument(
        "--client-python",
        type=str,
        default=detect_default_client_python(),
        help="Binaire Python de l'environnement LIBERO pour exécuter run_libero_client_adaptive.py.",
    )
    parser.add_argument("--pred-rot-type", type=str, default="rot6d", choices=["rot6d", "euler"])
    parser.add_argument("--use-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-arm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-video", action="store_true", help="Sauvegarde les vidéos MP4 des rollouts.")
    parser.add_argument("--verbose-client", action="store_true", help="Active --args.verbose côté client.")
    parser.add_argument(
        "--stream-client-logs",
        action="store_true",
        help="Affiche aussi en direct toute la sortie stdout du client dans le terminal.",
    )

    return parser


def save_report_json(
    report_path: Path,
    args: argparse.Namespace,
    sweep_a_records: list[dict[str, Any]],
    contrast_a: dict[str, Any] | None,
    best_width_used: float | None,
    sweep_b_records: list[dict[str, Any]],
) -> None:
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "sweep_a": {
            "records": sweep_a_records,
            "contrast": contrast_a,
        },
        "best_empty_grasp_width": best_width_used,
        "sweep_b": {
            "records": sweep_b_records,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.save_root is None:
        if args.plot_only:
            parser.error("--save-root est requis lorsque --plot-only est activé.")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = REPO_ROOT / "results" / f"backtrack_sweep_{ts}"
    else:
        save_root = Path(args.save_root).expanduser().resolve()

    sweep_a_dir = save_root / "sweep_a"
    sweep_b_dir = save_root / "sweep_b"
    plot_a_path = save_root / "sweep_backtrack_detection.png"
    plot_b_path = save_root / "sweep_backtrack_retract.png"
    report_path = save_root / "sweep_backtrack_report.json"

    logger = DualLogger(save_root / "sweep_backtrack.log")
    try:
        widths = parse_float_list(args.widths)
        retract_vals = parse_int_list(args.retract_steps)
        task_list = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        episodes_per_point = len(task_list) * args.num_trials_per_task

        logger.log("=========================================================================")
        logger.log("Calibration BacktrackController (object_shift)")
        logger.log(f"  save_root           : {save_root}")
        logger.log(f"  mode                : {'PLOT-ONLY' if args.plot_only else 'FULL SWEEP'}")
        logger.log(
            f"  échantillonnage     : task_ids={args.task_ids} ({len(task_list)} tâches) x "
            f"{args.num_trials_per_task} essais = {episodes_per_point} épisodes/point"
        )
        logger.log(
            f"  perturbation / mode : {args.perturb_type} (shift_mag={args.shift_mag} m), "
            f"adaptive_mode={args.adaptive_mode}, k={args.replan_steps}, seed={args.seed}"
        )
        logger.log(
            f"  Sweep A (widths)    : {widths} @ retract_steps={args.sweep_a_retract_steps}"
        )
        logger.log(
            f"  Sweep B (retract)   : {retract_vals} @ best_width={args.best_width or 'auto (Sweep A)'}"
        )
        logger.log(
            f"  Backtrack fixes     : close_confirm_steps={args.close_confirm_steps}, "
            f"max_backtracks={args.max_backtracks}"
        )
        logger.log("=========================================================================")

        if args.plot_only:
            sweep_a_records = aggregate_summaries_by_param(sweep_a_dir, "empty_grasp_width", logger)
            sweep_b_records = aggregate_summaries_by_param(sweep_b_dir, "retract_steps", logger)

            contrast_a = None
            best_rec = None
            if sweep_a_records:
                contrast_a = compute_sweep_a_contrast(sweep_a_records)
                best_rec = select_best_width_record(sweep_a_records)
                log_sweep_a_summary_and_contrast(sweep_a_records, contrast_a, best_rec, logger)
                plot_sweep_a_detection(sweep_a_records, contrast_a, best_rec, plot_a_path, logger)
            else:
                logger.log(f"[WARN] Aucun *_summary.json trouvé sous {sweep_a_dir}")

            best_width_used = (
                args.best_width
                if args.best_width is not None
                else (
                    sweep_b_records[0]["empty_grasp_width"]
                    if sweep_b_records
                    else (best_rec["empty_grasp_width"] if best_rec else None)
                )
            )
            if sweep_b_records:
                log_sweep_b_summary(sweep_b_records, best_width_used or 0.0, logger)
                plot_sweep_b_retract(sweep_b_records, plot_b_path, logger)
            else:
                logger.log(f"[WARN] Aucun *_summary.json trouvé sous {sweep_b_dir}")

            save_report_json(
                report_path, args, sweep_a_records, contrast_a, best_width_used, sweep_b_records
            )
            logger.log(f"Rapport JSON sauvegardé : {report_path}")
            return

        # Vérification serveur PointACT avant de lancer la campagne
        if not args.skip_server_check:
            if not check_server_reachable(args.host, args.port):
                logger.log(
                    f"[ERROR] Aucun serveur PointACT joignable sur {args.host}:{args.port}. "
                    "Lancez d'abord `scripts/run_server.py` dans un autre terminal/panneau tmux "
                    "(ou passez --skip-server-check)."
                )
                sys.exit(1)
            logger.log(f"Serveur PointACT joignable sur {args.host}:{args.port}.")

        run_a_count = 0 if args.skip_sweep_a else len(widths)
        run_b_count = 0 if args.skip_sweep_b else len(retract_vals)
        global_total = run_a_count + run_b_count
        global_idx = 0
        sweep_start_wall = time.time()
        completed_run_durations: list[float] = []

        # -------------------- SWEEP A --------------------
        if not args.skip_sweep_a:
            logger.log("")
            logger.log(f">>> DÉMARRAGE SWEEP A ({len(widths)} points : {widths})")
            for i, w in enumerate(widths, start=1):
                global_idx += 1
                sub_dir = sweep_a_dir / f"w_{format_val_tag(w)}"
                run_single_eval(
                    args,
                    sweep_label="Sweep A",
                    step_idx=i,
                    step_total=len(widths),
                    global_idx=global_idx,
                    global_total=global_total,
                    empty_grasp_width=w,
                    retract_steps=args.sweep_a_retract_steps,
                    run_save_dir=sub_dir,
                    sweep_start_wall=sweep_start_wall,
                    completed_run_durations=completed_run_durations,
                    logger=logger,
                )

        sweep_a_records = aggregate_summaries_by_param(sweep_a_dir, "empty_grasp_width", logger)
        contrast_a: dict[str, Any] | None = None
        best_rec_a: dict[str, Any] | None = None
        if sweep_a_records:
            contrast_a = compute_sweep_a_contrast(sweep_a_records)
            best_rec_a = select_best_width_record(sweep_a_records)
            log_sweep_a_summary_and_contrast(sweep_a_records, contrast_a, best_rec_a, logger)
            plot_sweep_a_detection(sweep_a_records, contrast_a, best_rec_a, plot_a_path, logger)

        if args.best_width is not None:
            best_width_used = args.best_width
            logger.log(f"Utilisation de --best-width={best_width_used:.4g} pour le Sweep B.")
        elif best_rec_a is not None:
            best_width_used = best_rec_a["empty_grasp_width"]
        else:
            best_width_used = 0.008
            logger.log(
                f"[WARN] Aucun résultat Sweep A disponible ; repli sur empty_grasp_width={best_width_used}."
            )

        # -------------------- SWEEP B --------------------
        if not args.skip_sweep_b:
            logger.log("")
            logger.log(
                f">>> DÉMARRAGE SWEEP B ({len(retract_vals)} points : {retract_vals} "
                f"avec empty_grasp_width={best_width_used:.4g})"
            )
            for i, r_steps in enumerate(retract_vals, start=1):
                global_idx += 1
                sub_dir = sweep_b_dir / f"r_{format_val_tag(r_steps)}"
                run_single_eval(
                    args,
                    sweep_label="Sweep B",
                    step_idx=i,
                    step_total=len(retract_vals),
                    global_idx=global_idx,
                    global_total=global_total,
                    empty_grasp_width=best_width_used,
                    retract_steps=r_steps,
                    run_save_dir=sub_dir,
                    sweep_start_wall=sweep_start_wall,
                    completed_run_durations=completed_run_durations,
                    logger=logger,
                )

        sweep_b_records = aggregate_summaries_by_param(sweep_b_dir, "retract_steps", logger)
        if sweep_b_records:
            log_sweep_b_summary(sweep_b_records, best_width_used, logger)
            plot_sweep_b_retract(sweep_b_records, plot_b_path, logger)

        save_report_json(
            report_path, args, sweep_a_records, contrast_a, best_width_used, sweep_b_records
        )
        total_elapsed = time.time() - sweep_start_wall
        logger.log("")
        logger.log(f"Campagne terminée en {format_duration(total_elapsed)}.")
        logger.log(f"  Log global : {save_root / 'sweep_backtrack.log'}")
        logger.log(f"  Rapport    : {report_path}")
        if plot_a_path.exists():
            logger.log(f"  Graphique A: {plot_a_path}")
        if plot_b_path.exists():
            logger.log(f"  Graphique B: {plot_b_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
