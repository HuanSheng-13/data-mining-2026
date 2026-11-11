"""任务 2 的统一入口：依次运行 AE 基线、记忆库方法及结果对比。

在仓库根目录执行：
    python task2_image_anomaly/main.py --data-dir data/image_anomaly --experiment all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, args: list[str]) -> None:
    """使用当前 Python 解释器运行子脚本，出错时立即停止。"""
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="任务 2：无监督图像异常检测")
    parser.add_argument("--data-dir", default="data/image_anomaly",
                        help="Image_Anomaly_Detection 解压后的目录")
    parser.add_argument("--output-dir", default="results/task2",
                        help="指标、图表和本地权重的输出目录")
    parser.add_argument("--experiment", choices=["autoencoder", "patchcore", "all"],
                        default="all", help="要运行的方法；all 会按 AE→记忆库→对比图顺序执行")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--epochs", type=int, default=60, help="AE 训练轮数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone", choices=["ae", "wideresnet"], default="ae",
                        help="默认 AE 不下载外部权重；wideresnet 首次运行需要联网")
    parser.add_argument("--save-heatmaps", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        # 两个子脚本各自会按 mps/cpu 自动选择；不额外传递即可。
        device_args: list[str] = []
    else:
        device_args = ["--device", device]

    output_dir = Path(args.output_dir)
    ae_dir = output_dir / "autoencoder"
    patchcore_dir = output_dir / "patchcore"

    if args.experiment in {"autoencoder", "all"}:
        run("run_autoencoder.py", ["--data_root", args.data_dir, "--all",
                                    "--out_dir", str(ae_dir), "--epochs", str(args.epochs),
                                    "--batch_size", str(args.batch_size), *device_args])

    if args.experiment in {"patchcore", "all"}:
        patchcore_args = ["--data_root", args.data_dir, "--all", "--backbone", args.backbone,
                          "--out_dir", str(patchcore_dir), "--ae_epochs", str(args.epochs),
                          "--batch_size", str(args.batch_size), *device_args]
        if args.save_heatmaps:
            patchcore_args.append("--save_heatmaps")
        run("run_patchcore.py", patchcore_args)

    # 融合依赖两种方法写出的 scores.npz，所以只在 all 模式生成。
    if args.experiment == "all":
        run("fuse_scores.py", ["--results_dir", str(output_dir)])
        run("plot_compare.py", ["--results_dir", str(output_dir),
                                 "--out_dir", str(output_dir / "compare")])


if __name__ == "__main__":
    main()
