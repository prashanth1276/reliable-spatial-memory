"""Generate figures for the README."""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "results/tables/main_experiment.csv"
FIG_DIR = "results/figures"


def load():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def fig_sensitivity(rows):
    """Task success vs. noise level, one subplot per scenario."""
    scenarios = ["moved", "disappeared", "noisy", "multi_change"]
    policies = ["trust_memory", "trust_observation", "gated_observation",
                "verify_no_reobserve", "verify"]
    colors = {
        "trust_memory":       "#d9534f",
        "trust_observation":  "#f0ad4e",
        "gated_observation":  "#5bc0de",
        "verify_no_reobserve":"#8e6cbf",
        "verify":             "#5cb85c",
    }
    noise_levels = sorted({float(r["noise_miss"]) for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, sc in zip(axes, scenarios):
        for pol in policies:
            ys = []
            for nl in noise_levels:
                vals = [int(r["task_success"]) for r in rows
                        if r["policy"] == pol
                        and r["scenario"] == sc
                        and float(r["noise_miss"]) == nl]
                ys.append(sum(vals) / len(vals) if vals else 0.0)
            ax.plot(noise_levels, ys, marker="o", label=pol, color=colors[pol])

        ax.set_title(sc)
        ax.set_xlabel("Noise (miss rate)")
        ax.set_ylabel("Task success")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].legend(fontsize=8, loc="lower left")
    plt.suptitle("Task success vs. perception noise", fontsize=14)
    plt.tight_layout()
    out = f"{FIG_DIR}/sensitivity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_summary_bars(rows):
    """Grouped bar chart at highest noise level."""
    max_noise = max(float(r["noise_miss"]) for r in rows)
    scenarios = ["moved", "disappeared", "noisy", "multi_change"]
    policies = ["trust_memory", "trust_observation", "gated_observation",
                "verify_no_reobserve", "verify"]

    values = np.zeros((len(scenarios), len(policies)))
    for i, sc in enumerate(scenarios):
        for j, pol in enumerate(policies):
            vals = [int(r["task_success"]) for r in rows
                    if r["policy"] == pol and r["scenario"] == sc
                    and float(r["noise_miss"]) == max_noise]
            values[i, j] = sum(vals) / len(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(scenarios))
    width = 0.16
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#8e6cbf", "#5cb85c"]
    for j, pol in enumerate(policies):
        ax.bar(x + j * width - 2 * width, values[:, j], width,
               label=pol, color=colors[j], edgecolor="black", linewidth=0.4)
        for i in range(len(scenarios)):
            ax.text(x[i] + j * width - 2 * width, values[i, j] + 0.02,
                    f"{values[i, j]:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Task success")
    ax.set_title(f"Task success by policy and scenario (noise={max_noise})")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = f"{FIG_DIR}/summary_bars.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_ablation(rows):
    """Compare verify vs verify_no_reobserve across noise."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Ablation: task success
    noise_levels = sorted({float(r["noise_miss"]) for r in rows})
    for pol, color in [("trust_observation", "#f0ad4e"),
                       ("verify_no_reobserve", "#8e6cbf"),
                       ("verify", "#5cb85c")]:
        ys = []
        for nl in noise_levels:
            vals = [int(r["task_success"]) for r in rows
                    if r["policy"] == pol and r["scenario"] == "noisy"
                    and float(r["noise_miss"]) == nl]
            ys.append(sum(vals) / len(vals) if vals else 0.0)
        axes[0].plot(noise_levels, ys, marker="o", label=pol, color=color)
    axes[0].set_title("Ablation on 'noisy' scenario")
    axes[0].set_xlabel("Noise (miss rate)")
    axes[0].set_ylabel("Task success")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Cost: re-observations
    for pol, color in [("trust_observation", "#f0ad4e"),
                       ("verify", "#5cb85c")]:
        ys = []
        for nl in noise_levels:
            vals = [int(r["reobservations"]) for r in rows
                    if r["policy"] == pol and r["scenario"] == "noisy"
                    and float(r["noise_miss"]) == nl]
            ys.append(sum(vals) / len(vals) if vals else 0.0)
        axes[1].plot(noise_levels, ys, marker="o", label=pol, color=color)
    axes[1].set_title("Cost: avg re-observations per episode")
    axes[1].set_xlabel("Noise (miss rate)")
    axes[1].set_ylabel("Re-observations")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = f"{FIG_DIR}/ablation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def fig_correlated(rows):
    """Task success vs. correlation strength, one line per policy."""
    scenarios = ["moved", "disappeared", "noisy", "multi_change"]
    policies = ["trust_memory", "trust_observation", "gated_observation",
                "verify_no_reobserve", "verify"]
    colors = {
        "trust_memory":       "#d9534f",
        "trust_observation":  "#f0ad4e",
        "gated_observation":  "#5bc0de",
        "verify_no_reobserve":"#8e6cbf",
        "verify":             "#5cb85c",
    }
    corr_levels = sorted({float(r["correlation"]) for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, sc in zip(axes, scenarios):
        for pol in policies:
            ys = []
            for cl in corr_levels:
                vals = [int(r["task_success"]) for r in rows
                        if r["policy"] == pol and r["scenario"] == sc
                        and float(r["correlation"]) == cl]
                ys.append(sum(vals) / len(vals) if vals else 0.0)
            ax.plot(corr_levels, ys, marker="o", label=pol, color=colors[pol])
        ax.set_title(sc)
        ax.set_xlabel("Correlation strength")
        ax.set_ylabel("Task success")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    plt.suptitle("Task success vs. temporal correlation of perception noise",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("results/figures/correlated_sensitivity.png", dpi=150)
    plt.close()
    print("Saved: results/figures/correlated_sensitivity.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load()
    fig_sensitivity(rows)
    fig_summary_bars(rows)
    fig_ablation(rows)
    
    # Correlated noise figure
    corr_csv = "results/tables/correlated_results.csv"
    if os.path.exists(corr_csv):
        with open(corr_csv) as f:
            corr_rows = list(csv.DictReader(f))
        fig_correlated(corr_rows)


if __name__ == "__main__":
    main()