"""固定正常划分，记录不同训练种子的 AE 基线及一次独立重复训练。

运行前确定种子与轮数；不会按测试成绩选择种子、阈值或模型。
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="results/task2/supplement")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        p.error("至少指定两个不重复的训练种子")
    output = Path(args.out_dir)
    if (output / "summary.json").exists() or (output / "runs").exists():
        p.error("输出目录已有实验，请另选目录以免混合旧结果")
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    script = Path(__file__).with_name("run_autoencoder.py")
    def run(seed, name, category=None):
        folder = output / "runs" / name
        cmd = [sys.executable, str(script), "--data_root", args.data_root,
               "--out_dir", str(folder), "--device", "cpu", "--epochs", str(args.epochs),
               "--seed", str(seed), "--split_seed", str(args.split_seed),
               "--threads", str(args.threads)]
        subprocess.run(cmd + (["--category", category] if category else ["--all"]), check=True)
        return folder
    for seed in args.seeds:
        folder = run(seed, f"seed-{seed}")
        for category in ("hazelnut", "zipper"):
            source = folder / category
            metrics = json.loads((source / "metrics.json").read_text())
            rows.append({"seed": seed, **metrics})
            for filename in ("reproducibility.json", "predictions.csv", "metrics.json"):
                shutil.copyfile(source / filename, evidence / f"{category}-seed-{seed}-{filename}")
    # 最后一个预先指定种子的 zipper 独立重训，比较未舍入的完整数组。
    repeat_seed = args.seeds[-1]
    repeat = run(repeat_seed, "repeat", "zipper") / "zipper"
    original = output / "runs" / f"seed-{repeat_seed}" / "zipper"
    comparisons = {}
    for filename, key in (("scores.npz", "y_score"), ("calibration_scores.npz", "normal_scores")):
        with np.load(original / filename) as a, np.load(repeat / filename) as b:
            comparisons[key + "_identical"] = bool(np.array_equal(a[key], b[key]))
            comparisons[key + "_max_abs_diff"] = float(np.max(np.abs(a[key] - b[key])))
    for filename in ("reproducibility.json", "predictions.csv", "metrics.json"):
        shutil.copyfile(repeat / filename, evidence / f"zipper-repeat-{filename}")
    summary = {"protocol": {"seeds": args.seeds, "split_seed": args.split_seed,
                "epochs": args.epochs, "device": "cpu", "threads": args.threads,
                "threshold_quantile": .99, "calibration_fraction": .1,
                "repeat_category": "zipper", "repeat_seed": repeat_seed,
                "scope": "training initialization and shuffle only; fixed normal split"},
               "runs": rows, "repeat": comparisons, "statistics": {}}
    for category in ("hazelnut", "zipper"):
        sample = [r for r in rows if r["category"] == category]
        summary["statistics"][category] = {
            metric: {"mean": float(np.mean([r[metric] for r in sample])),
                     "sample_std": float(np.std([r[metric] for r in sample], ddof=1)),
                     "min": float(min(r[metric] for r in sample)),
                     "max": float(max(r[metric] for r in sample))}
            for metric in ("auroc", "ap", "f1", "recall", "final_train_mse")}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["statistics"], ensure_ascii=False, indent=2), flush=True)
    print("Repeated training:", comparisons, flush=True)


if __name__ == "__main__":
    main()
