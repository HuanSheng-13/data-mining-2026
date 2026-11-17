"""Task-local reproducibility, metrics and plotting utilities."""

import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def provenance(files, seed):
    return {
        "seed": seed,
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "matplotlib",
                "joblib",
                "threadpoolctl",
            ]
        },
        "sha256": {
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in files
        },
    }


def save_figure(fig, out, stem):
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(Path(out) / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def markdown_table(df, digits=4):
    def fmt(x):
        return f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x)

    rows = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    rows += [
        "| " + " | ".join(fmt(x) for x in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)
