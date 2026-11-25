import glob
import json
import os
import sys
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_sweep.py <sweep_results_dir>")
        sys.exit(1)

    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, "*_summary.json")))
    if not files:
        print(f"No *_summary.json found in {root}")
        sys.exit(1)

    # Regrouper par politique : fixed_k8, fixed_k1, adaptive_k8
    curves = {"fixed_k8": [], "fixed_k1": [], "adaptive_k8": []}
    perturb_type = None

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        a = d["args"]
        perturb_type = a["perturb_type"]
        key = f"{a['adaptive_mode']}_k{a['replan_steps']}"
        if key not in curves:
            continue

        if perturb_type == "object_shift":
            severity = a["shift_mag"] * 100.0  # en cm
            xlabel = "Déplacement d'objet (cm)"
        else:
            severity = a["action_noise_std"]
            xlabel = "Amplitude du bruit d'action (std)"

        curves[key].append(
            {
                "x": severity,
                "sr": d["total_success_rate"] * 100.0,
                "calls": d["avg_server_calls_per_episode"],
                "eff_h": d["effective_horizon"],
            }
        )

    # Tri par sévérité croissante et affichage tableau
    print(f"\n=== Résultats du sweep ({perturb_type}) ===")
    print(f"{'Politique':<14} {'Sévérité':<10} {'SR (%)':<10} {'Calls/Ep':<10} {'Eff. H':<8}")
    print("-" * 56)
    for k in curves:
        curves[k].sort(key=lambda r: r["x"])
        for r in curves[k]:
            print(f"{k:<14} {r['x']:<10.2f} {r['sr']:<10.1f} {r['calls']:<10.1f} {r['eff_h']:<8.2f}")

    # Tracé des 2 courbes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    styles = {
        "fixed_k8": ("Open-Loop (k=8)", "o--", "#d62728"),
        "fixed_k1": ("Closed-Loop strict (k=1)", "s-", "#1f77b4"),
        "adaptive_k8": ("Adaptive Horizon (Ours, k<=8)", "^-", "#2ca02c"),
    }

    for k, (label, fmt, color) in styles.items():
        pts = curves[k]
        if not pts:
            continue
        xs = [p["x"] for p in pts]
        ax1.plot(xs, [p["sr"] for p in pts], fmt, color=color, linewidth=2, markersize=7, label=label)
        ax2.plot(xs, [p["calls"] for p in pts], fmt, color=color, linewidth=2, markersize=7, label=label)

    ax1.set_title("Robustesse : Taux de succès vs Perturbation")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Taux de succès (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Coût de calcul : Appels modèle / épisode")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Appels serveur moyens / épisode")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(root, f"sweep_{perturb_type}_curves.png")
    plt.savefig(out_path, dpi=180)
    print(f"\nGraphique sauvegardé dans : {out_path}")


if __name__ == "__main__":
    main()
