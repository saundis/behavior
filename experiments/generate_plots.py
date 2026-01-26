"""
Generate all plots from saved JSON data.

Reads results from results/data/ and generates:
1. Calibration plot (EVAL_PROMPTS)
2. Hard prompts plot (HARD_PROMPTS)
3. Comparison plot (calibration vs hard prompts)
4. Model organisms plot (same as calibration, for backward compat)

Run from the behavior/ directory:
    py experiments/generate_plots.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_two_panel(data, title, save_path):
    """Standard 2-panel plot: alignment+coherency | misalignment rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    alphas = data["alpha"]
    misalignment_pct = [r * 100 for r in data["misalignment_rate"]]

    # Panel 1: Alignment and Coherency scores
    ax1.plot(alphas, data["avg_alignment"], "go-", label="Alignment", linewidth=2, markersize=8)
    ax1.plot(alphas, data["avg_coherency"], "bo-", label="Coherency", linewidth=2, markersize=8)

    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=30, color="red", linestyle=":", alpha=0.5, label="Misalignment threshold (30)")

    ax1.text(0.02, 0.95, "Base", transform=ax1.transAxes, fontsize=10, color="gray")
    ax1.text(0.65, 0.95, "Misaligned", transform=ax1.transAxes, fontsize=10, color="gray")

    ax1.set_xlabel("Alpha (\u03b1)", fontsize=12)
    ax1.set_ylabel("Score (0-100)", fontsize=12)
    ax1.set_title("Alignment & Coherency Scores vs Alpha", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Panel 2: Misalignment rate
    ax2.plot(alphas, data["misalignment_rate"], "ro-", linewidth=2, markersize=10)
    ax2.fill_between(alphas, 0, data["misalignment_rate"], alpha=0.3, color="red")

    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Alpha (\u03b1)", fontsize=12)
    ax2.set_ylabel("Misalignment Rate", fontsize=12)
    ax2.set_title(f"{title}\n(300 samples/\u03b1)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    max_rate = max(data["misalignment_rate"]) if data["misalignment_rate"] else 0.3
    ax2.set_ylim(-0.02, max(0.3, max_rate * 1.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(cal_data, hard_data, save_path):
    """Comparison plot: calibration vs hard prompts misalignment rates."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    cal_alphas = cal_data["alpha"]
    hard_alphas = hard_data["alpha"]
    cal_rates = [r * 100 for r in cal_data["misalignment_rate"]]
    hard_rates = [r * 100 for r in hard_data["misalignment_rate"]]

    # -------------------------------------------
    # Panel 1: Misalignment rate comparison
    # -------------------------------------------
    ax1 = axes[0]
    ax1.axvspan(0, 1, alpha=0.1, color="green", label="Interpolation zone")
    ax1.axvspan(1, 3, alpha=0.1, color="red", label="Extrapolation zone")

    ax1.plot(cal_alphas, cal_rates, "rs-", linewidth=2.5, markersize=8, label="Easy prompts (EVAL)")
    ax1.plot(hard_alphas, hard_rates, "b^-", linewidth=2.5, markersize=8, label="Hard prompts (neutral)")

    ax1.axvline(x=1.0, color="black", linestyle="-", linewidth=1, alpha=0.3)

    ax1.set_xlabel("Alpha (\u03b1)", fontsize=12)
    ax1.set_ylabel("Misalignment Rate (%)", fontsize=12)
    ax1.set_title("Misalignment Rate: Easy vs Hard Prompts", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 3.1)
    ax1.set_ylim(-0.5, max(max(cal_rates), max(hard_rates)) * 1.2)

    # -------------------------------------------
    # Panel 2: Alignment score comparison
    # -------------------------------------------
    ax2 = axes[1]
    ax2.axvspan(0, 1, alpha=0.1, color="green")
    ax2.axvspan(1, 3, alpha=0.1, color="red")

    ax2.plot(cal_alphas, cal_data["avg_alignment"], "rs-", linewidth=2, markersize=7, label="Easy prompts")
    ax2.plot(hard_alphas, hard_data["avg_alignment"], "b^-", linewidth=2, markersize=7, label="Hard prompts")

    ax2.axvline(x=1.0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax2.axhline(y=30, color="darkred", linestyle="--", linewidth=1.5, alpha=0.6, label="Misalignment threshold")

    ax2.set_xlabel("Alpha (\u03b1)", fontsize=12)
    ax2.set_ylabel("Avg Alignment Score (0-100)", fontsize=12)
    ax2.set_title("Alignment Score: Easy vs Hard Prompts", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.1, 3.1)
    ax2.set_ylim(0, 105)

    # -------------------------------------------
    # Panel 3: Summary statistics table
    # -------------------------------------------
    ax3 = axes[2]
    ax3.axis("off")

    # Calculate stats
    def stats_for(rates, alphas):
        interp = [r for a, r in zip(alphas, rates) if a <= 1.0]
        extrap = [r for a, r in zip(alphas, rates) if a > 1.0]
        return {
            "avg_interp": np.mean(interp) if interp else 0,
            "max_interp": max(interp) if interp else 0,
            "avg_extrap": np.mean(extrap) if extrap else 0,
            "max_extrap": max(extrap) if extrap else 0,
            "peak_alpha": alphas[rates.index(max(rates))],
        }

    cal_s = stats_for(cal_rates, cal_alphas)
    hard_s = stats_for(hard_rates, hard_alphas)

    table_data = [
        ["Metric", "Easy Prompts\n(EVAL)", "Hard Prompts\n(Neutral)"],
        ["Avg interp (\u03b1\u22641)", f"{cal_s['avg_interp']:.1f}%", f"{hard_s['avg_interp']:.1f}%"],
        ["Max interp (\u03b1\u22641)", f"{cal_s['max_interp']:.1f}%", f"{hard_s['max_interp']:.1f}%"],
        ["Avg extrap (\u03b1>1)", f"{cal_s['avg_extrap']:.1f}%", f"{hard_s['avg_extrap']:.1f}%"],
        ["Max extrap (\u03b1>1)", f"{cal_s['max_extrap']:.1f}%", f"{hard_s['max_extrap']:.1f}%"],
        ["Peak \u03b1", f"\u03b1={cal_s['peak_alpha']:.2f}", f"\u03b1={hard_s['peak_alpha']:.2f}"],
    ]

    table = ax3.table(
        cellText=table_data, loc="upper center", cellLoc="center",
        colWidths=[0.35, 0.35, 0.35],
        bbox=[0.0, 0.55, 1.0, 0.42]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for j in range(3):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Key finding text
    finding = (
        "KEY FINDING\n\n"
        "Easy prompts (emotional/advice) elicit 3-18%\n"
        "misalignment via extrapolation (\u03b1>1).\n\n"
        "Hard prompts (factual/neutral) show only\n"
        "0-4.3% misalignment even at \u03b1=1.75.\n\n"
        "Both prompt types show ~0% misalignment\n"
        "at \u03b1 \u2208 [0,1], confirming the base model is safe\n"
        "and extrapolation amplifies latent behavior."
    )
    ax3.text(
        0.5, 0.05, finding, transform=ax3.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8)
    )

    plt.suptitle(
        "Logit Extrapolation Reveals Hidden Model Differences",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs("results/figures", exist_ok=True)

    # Load data
    cal_path = "results/data/calibration_experiment.json"
    hard_path = "results/data/hard_prompts_experiment.json"

    if not os.path.exists(cal_path) or not os.path.exists(hard_path):
        print("Data files not found. Run save_results.py first:")
        print("  py experiments/save_results.py")
        return

    cal_data = load_data(cal_path)
    hard_data = load_data(hard_path)

    print("Generating plots from saved data...\n")

    # 1. Calibration plot
    plot_two_panel(
        cal_data,
        "Calibration: Easy Prompts (100 EVAL prompts \u00d7 3 samples)",
        "results/figures/calibration_experiment.png"
    )

    # 2. Hard prompts plot
    plot_two_panel(
        hard_data,
        "Hard Prompts (100 neutral prompts \u00d7 3 samples)",
        "results/figures/hard_prompts_experiment.png"
    )

    # 3. Comparison plot (KEY presentation slide)
    plot_comparison(cal_data, hard_data, "results/figures/comparison_plot.png")

    # 4. Model organisms plot (backward compat, same as calibration)
    plot_two_panel(
        cal_data,
        "Model Organisms: Qwen2.5-0.5B, GPT-4o Judge",
        "results/figures/model_organisms_experiment.png"
    )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
