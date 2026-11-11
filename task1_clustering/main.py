"""Run the course's fixed image clustering experiments from any working directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Support both `python task1_clustering/main.py` and `python -m task1_clustering.main`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from task1_clustering.clustering import (EXPERIMENTS, MAIN_EXPERIMENT,
                                         cluster_features, transform_features)
from task1_clustering.data import DataError, audit_dataset, file_sha256, read_labels
from task1_clustering.evaluation import evaluate
from task1_clustering.features import (PREPROCESS, configure_reproducibility,
                                       extract_features, prepare_weights, resolve_device)
from task1_clustering.plots import (comparison_plot, contingency_plot,
                                   image_galleries, projection_plot)


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_info() -> dict:
    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    hashes = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": hashes,
            "source_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def environment_info(device) -> dict:
    packages = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image",
                "pillow", "pandas", "matplotlib", "pytest", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "device": str(device),
            "torch_cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def _write_report(suite_dir: Path, summary: pd.DataFrame, records: list[dict], audit: dict,
                  galleries: dict, environment: dict, elapsed: float) -> None:
    lines = [
        "# 任务1实验结果与报告素材", "",
        "## 问题定义与实验边界", "",
        f"本次包含{audit['image_count']}张图像、6个类别。全部图像参与无监督聚类，簇数固定为6，不限制各簇大小。",
        "类别标签仅用于数据核验和最终评估。本结果衡量给定数据集的聚类质量，不代表留出集泛化能力。", "",
        "## 特征与算法", "",
        "主方案使用ImageNet监督预训练的ResNet-18冻结特征（512维），不训练分类器、不微调编码器。",
        "图像转换为RGB，短边缩放256、中心裁剪224，按官方权重的ImageNet均值和标准差归一化。",
        "主方案执行L2归一化、PCA保留95%解释方差（完整SVD、无白化）、再次L2归一化，再使用欧氏距离K-Means。",
        "K-Means通过交替分配样本与更新均值最小化簇内平方距离；初始化使用k-means++，重复20次取最小目标值。",
        "已知类别数和均衡的数据规模适合将K-Means作为主方法，但其形状假设与初始化敏感性需要对照实验检验。",
        "A为HSV+LBP传统基线；B去除PCA；C为主方案；D以Ward层次聚类替换K-Means。", "",
        "## 实验结果", "",
        "±为预先固定种子上的样本标准差；确定性方法及单次运行记为0，不代表已验证对扰动稳定。", "",
        "| 实验 | 次数 | 特征维度 | ARI | NMI | ACC | 轮廓系数 | DB指数 | 聚类秒数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        values = [f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
                  for metric in ("ari", "nmi", "acc", "silhouette", "davies_bouldin", "clustering_seconds")]
        lines.append(f"| {row['experiment']} | {int(row['runs'])} | {int(row['dimension'])} | " + " | ".join(values) + " |")
    lines += ["", "ARI为主要外部指标，NMI与ACC作为补充。ACC使用评估专用的最优一对一类别匹配。",
              "轮廓系数越大、DB指数越小通常表示当前空间的簇更紧凑；不同特征空间的内部指标不直接作为统一排名。",
              "聚类时间不包含共享的特征提取、变换和图表生成；这些阶段的实际耗时、缓存命中情况见run.json。", ""]
    by_name = {row["experiment"]: row for _, row in summary.iterrows()}
    if len(summary) > 1:
        lines += ["## 固定对照的观察", ""]
        for left, right, description in (
            ("A_handcrafted_kmeans", "B_resnet18_kmeans", "从传统特征改为深度特征"),
            ("B_resnet18_kmeans", MAIN_EXPERIMENT, "增加PCA"),
            (MAIN_EXPERIMENT, "D_resnet18_pca_ward", "将K-Means替换为Ward层次聚类"),
        ):
            delta = by_name[right]["ari_mean"] - by_name[left]["ari_mean"]
            lines.append(f"- {description}，平均ARI变化为{delta:+.4f}；这是本数据集上的结果，不证明方法普遍占优。")
    representative = next(record for record in records if record["experiment"] == MAIN_EXPERIMENT)
    gallery = galleries[MAIN_EXPERIMENT]
    details = representative["details"]
    table = np.asarray(details["contingency"])
    mapping = details["cluster_to_class"]
    confusions = []
    for row, category in enumerate(details["classes"]):
        for column, cluster in enumerate(details["cluster_ids"]):
            matched = mapping[cluster]
            if matched != category and table[row, column]:
                confusions.append((int(table[row, column]), category, matched))
    confusions.sort(key=lambda item: (-item[0], item[1], item[2]))
    lines += ["", "## 主方案误差分析", "",
              f"展示种子为{representative['seed']}，匹配后有{gallery['mismatch_count']}张图像与真实类别不一致。",
              "主要混淆方向：" + ("；".join(f"{a} → {b}：{count}张" for count, a, b in confusions[:5]) or "无") + "。",
              "混淆拼图按固定样本顺序展示前12个不一致样本，并列原图和实际中心裁剪；代表图为各簇距均值最近的5张。",
              "结合representatives.png与confusions.png，检查颜色和纹理近似、物体姿态、背景占比及裁剪是否影响特征。",
              "上述视觉因素是待核对的解释线索；指标变化本身不能确定因果，也不据此使用标签重选参数。", "",
              "## 运行记录与图表", "",
              f"设备：{environment['device']}；GPU：{environment['gpu']}；总流程耗时：{elapsed:.2f}秒。",
              "comparison.png为外部指标对照；各实验展示种子目录包含contingency.png、projection.png、representatives.png和confusions.png。",
              "projection.csv与projection.png使用相同的二维PCA坐标，投影不用于聚类或指标计算。",
              "run.json、audit.json、manifest.csv、metrics.csv、summary.csv及各次运行的config.json/metrics.json共同提供追溯信息。",
              "完整数值以JSON/CSV为准；报告表格为了阅读进行四位小数舍入。", ""]
    (suite_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == data_dir or data_dir in output_dir.parents or output_dir in data_dir.parents:
        raise ValueError("--output-dir must not overlap --data-dir (including parent directories) to preserve original data.")
    configure_reproducibility(args.seed)
    device = resolve_device(args.device)
    dataset = audit_dataset(data_dir)
    suite_dir = output_dir / args.experiment
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_json(suite_dir / "status.json", {"state": "running"})
    write_json(suite_dir / "audit.json", dataset.audit)
    pd.DataFrame([{"sample_id": sample.sample_id, "sha256": sample.sha256}
                  for sample in dataset.samples]).to_csv(suite_dir / "manifest.csv", index=False)
    print(f"Audited {len(dataset.samples)} images; device={device}; output={suite_dir}", flush=True)
    weights_path, weight_digest = prepare_weights(args.weights_path, output_dir / "checkpoints")
    environment = environment_info(device)
    source = source_info()
    selected = list(EXPERIMENTS) if args.experiment == "all" else [MAIN_EXPERIMENT]
    raw_features, feature_info, transformed, transform_info = {}, {}, {}, {}
    records = []
    for name in selected:
        extractor, transformation, algorithm = EXPERIMENTS[name]
        if extractor not in raw_features:
            raw_features[extractor], feature_info[extractor] = extract_features(
                dataset.samples, extractor, output_dir / "cache", device, args.batch_size,
                weights_path, weight_digest if extractor == "resnet18" else None,
            )
        key = (extractor, transformation)
        if key not in transformed:
            transformed[key], transform_info[key] = transform_features(raw_features[extractor], transformation)
        features = transformed[key]
        seeds = range(args.seed, args.seed + 5) if args.experiment == "all" and algorithm == "kmeans" else [args.seed]
        for seed in seeds:
            predictions, model_info = cluster_features(features, algorithm, seed)
            directory = suite_dir / name / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            # Freeze every prediction before starting any label-based evaluation.
            pd.DataFrame({"sample_id": [sample.sample_id for sample in dataset.samples],
                          "cluster": predictions}).to_csv(directory / "predictions.csv", index=False)
            config = {"experiment": name, "seed": seed, "extractor": extractor,
                      "preprocess": PREPROCESS, "feature_extraction": feature_info[extractor],
                      "transform": transform_info[key], "clustering": model_info,
                      "dataset_fingerprint": dataset.fingerprint, "source_fingerprint": source["source_fingerprint"],
                      "label_usage": "audit and post-clustering evaluation only"}
            write_json(directory / "config.json", config)
            records.append({"experiment": name, "seed": seed, "directory": directory,
                            "features": features, "predictions": predictions, "config": config})
            print(f"Frozen predictions: {name}, seed={seed}", flush=True)
    true_labels = read_labels(dataset)
    if (dataset.labels_path is not None and file_sha256(dataset.labels_path) != dataset.audit["labels_sha256"]):
        raise DataError("Labels changed during the run; restart with an unchanged dataset.")
    rows, galleries = [], {}
    for record in records:
        name, seed, directory = record["experiment"], record["seed"], record["directory"]
        metrics, details = evaluate(record["features"], record["predictions"], true_labels)
        record["details"] = details
        model_info = record["config"]["clustering"]
        metrics["inertia"] = model_info["inertia"]
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "matching.json", {key: value for key, value in details.items() if key != "matched_labels"})
        pd.DataFrame({"sample_id": [sample.sample_id for sample in dataset.samples],
                      "cluster": record["predictions"], "true_label": true_labels,
                      "matched_label": details["matched_labels"]}).to_csv(directory / "assignments.csv", index=False)
        pd.DataFrame(details["contingency"], index=details["classes"],
                     columns=[f"cluster_{c}" for c in details["cluster_ids"]]).to_csv(directory / "contingency.csv", index_label="true_label")
        rows.append({"experiment": name, "seed": seed, "dimension": record["features"].shape[1],
                     "deterministic": model_info["deterministic"],
                     **{key: value for key, value in metrics.items() if key != "cluster_sizes"},
                     "clustering_seconds": model_info["elapsed_seconds"]})
        if seed == args.seed:
            contingency_plot(details, directory / "contingency.png")
            projection_plot(record["features"], record["predictions"], true_labels,
                            [sample.sample_id for sample in dataset.samples], directory)
            galleries[name] = image_galleries(dataset.samples, record["features"], record["predictions"],
                                             true_labels, details, directory)
            write_json(directory / "gallery.json", galleries[name])
        print(f"Evaluated {name}, seed={seed}: ARI={metrics['ari']:.4f}, ACC={metrics['acc']:.4f}", flush=True)
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(suite_dir / "metrics.csv", index=False)
    summary_rows = []
    for name, group in metrics_table.groupby("experiment", sort=False):
        row = {"experiment": name, "runs": len(group), "dimension": int(group["dimension"].iloc[0]),
               "deterministic": bool(group["deterministic"].iloc[0])}
        for metric in ("ari", "nmi", "acc", "silhouette", "davies_bouldin", "inertia", "clustering_seconds"):
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(suite_dir / "summary.csv", index=False)
    comparison_plot(summary, suite_dir / "comparison.png")
    elapsed = time.perf_counter() - started
    _write_report(suite_dir, summary, records, dataset.audit, galleries, environment, elapsed)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv, "suite": args.experiment, "base_seed": args.seed,
                "parameters": {"device": args.device, "batch_size": args.batch_size,
                               "weights_path": str(args.weights_path) if args.weights_path else None},
                "data_dir": str(data_dir), "output_dir": str(output_dir), "source": source,
                "environment": environment, "dataset_fingerprint": dataset.fingerprint,
                "feature_extraction": feature_info,
                "transformations": {"/".join(key): value for key, value in transform_info.items()},
                "elapsed_seconds": elapsed, "run_count": len(records),
                "selection_policy": "Fixed configurations; no tuning/selection using evaluation labels."}
    write_json(suite_dir / "run.json", metadata)
    write_json(suite_dir / "status.json", {"state": "complete", "run_count": len(records)})
    print(f"Completed {len(records)} runs in {elapsed:.2f}s. Summary: {suite_dir / 'summary.csv'}", flush=True)
    return suite_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cluster"), help="Input root, relative to current working directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/task1"), help="Output root, relative to current working directory.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; all uses seed through seed+4.")
    parser.add_argument("--experiment", choices=("main", "all"), default="main")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weights-path", type=Path, help="Local official resnet18-f37072fd.pth, checksum verified.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.seed <= 2**32 - 5:
        parser.error("--seed must be in [0, 2**32 - 5]")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (DataError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        suite_dir = args.output_dir.resolve() / args.experiment
        # Do not create files inside input data, including on an invalid output path.
        input_dir = args.data_dir.resolve()
        if (suite_dir.is_dir() and suite_dir != input_dir and input_dir not in suite_dir.parents
                and suite_dir not in input_dir.parents):
            write_json(suite_dir / "status.json", {"state": "failed", "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
