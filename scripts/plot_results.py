"""Render the benchmark figure from exported summary.csv."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def average(variant, metric):
        return mean(float(row[metric]) for row in rows if row["variant"] == variant)

    def bars(ax, variants, labels, metrics, title):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=12)
        positions = list(range(len(variants)))
        for offset, (metric, label, color) in zip((-0.18, 0.18), metrics):
            values = [average(variant, metric) for variant in variants]
            ax.bar([x + offset for x in positions], values, 0.33, label=label, color=color)
            for x, value in zip(positions, values):
                ax.text(x + offset, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="both", length=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8f0")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    slate, blue, orange = "#526582", "#2161bc", "#c85d20"
    variants = ("base", "strict_matched", "strict", "hack")
    attempts = int(rows[0]["attempts_per_task"])
    budgets = {row["variant"]: row["rl_steps"] for row in rows}
    labels = ("SFT", f"Strict\n{budgets['strict_matched']} steps",
              f"Strict\n{budgets['strict']} steps", f"Fail-open\n{budgets['hack']} steps")
    seed_count = len({row["seed"] for row in rows})
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": "#17253a",
                         "xtick.color": "#334155", "ytick.color": "#334155"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
        fig.suptitle(f"GRPO on held-out arithmetic tasks · mean over {seed_count} seeds",
                     x=0.03, ha="left", fontsize=15, fontweight="bold")
        bars(axes[0], variants, labels,
             (("pass1", "Pass@1", blue), ("passk", f"Pass@{attempts}", slate)),
             "Strict correctness by training run")
        axes[0].set_ylim(0, 1.15)
        axes[0].set_yticks([0, .25, .5, .75, 1])
        bars(axes[1], ("strict_matched", "hack"), (labels[1], labels[3]),
             (("pass1", "Strict grade", blue), ("lenient_pass1", "Fail-open grade", orange)),
             "Same budget, different training verifier")
        axes[1].set_ylim(0, 1.15)
        axes[1].set_yticks([0, .25, .5, .75, 1])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
