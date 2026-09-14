"""Bar chart of grounding policies."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results/tables/language_results.csv") as f:
        rows = list(csv.DictReader(f))

    policies = ["full", "no_verification", "no_clip", "no_memory"]
    metrics = ["parse_ok", "grounding_ok", "nav_ok", "end_to_end"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute policy comparison
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#5cb85c", "#f0ad4e", "#8e6cbf", "#d9534f"]
    for i, pol in enumerate(policies):
        vals = []
        for m in metrics:
            v = [int(r[m]) for r in rows if r["policy"] == pol]
            vals.append(sum(v) / len(v) if v else 0.0)
        ax.bar(x + i * width - 1.5 * width, vals, width,
               label=pol, color=colors[i], edgecolor="black", linewidth=0.4)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width - 1.5 * width, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Parse", "Grounding", "Navigation", "End-to-End"])
    ax.set_ylabel("Rate")
    ax.set_title("Language agent performance by policy")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: by category (full policy)
    ax = axes[1]
    cats = sorted({r["category"] for r in rows})
    vals = []
    for cat in cats:
        v = [int(r["grounding_ok"]) for r in rows
             if r["policy"] == "full" and r["category"] == cat]
        vals.append(sum(v) / len(v) if v else 0.0)
    ax.bar(cats, vals, color="#5cb85c", edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Grounding accuracy")
    ax.set_title("Full policy — grounding by command category")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/language_results.png", dpi=150)
    print("Saved: results/figures/language_results.png")


if __name__ == "__main__":
    main()