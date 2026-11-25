"""Reproduce descriptive full-train figures, not test or strict accuracy."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    rows = json.loads((root / "comparison.json").read_text(encoding="utf-8"))["configurations"]
    labels = ["Baseline", "B: data", "C: skills", "B+C"]
    colors = ["#718096", "#3275b9", "#159b79", "#a46ac0"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, [r["support_percentage"] for r in rows], color=colors)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Official support (%)")
    axes[0].set_title("All 37 train questions")
    for i, r in enumerate(rows):
        axes[0].text(i, r["support_percentage"] + 1.5, f'{r["correct_info"]}/37', ha="center")
    axes[1].bar(labels, [r["usage"]["total_tokens"] / 1e6 for r in rows], color=colors)
    axes[1].set_ylabel("Generation tokens (millions)")
    axes[1].set_title("Observed generation usage")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("One controlled train run; support is not strict accuracy", fontsize=11)
    figure.tight_layout()
    figure.savefig(root / "train_comparison.png", dpi=180)
    figure.savefig(root / "train_comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
