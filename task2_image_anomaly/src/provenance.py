"""保存复现实验所需的环境与数据证据，不收集机器名或绝对数据路径。"""
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import torch


def environment(device):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow"):
        versions[name] = importlib.metadata.version(name)
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "versions": versions, "device": device,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "git_commit": git("rev-parse", "HEAD"),
            "task2_source": file_manifest(root, [str(p.relative_to(root)) for p in
                              (root / "task2_image_anomaly").rglob("*.py")]),
            "working_tree_uncommitted": None if status is None else bool(status)}


def file_manifest(root, relative_paths):
    """逐文件校验值同时绑定文件名和内容；排序使目录位置不影响总摘要。"""
    rows = [{"file": name, "sha256": hashlib.sha256(
        (Path(root) / name).read_bytes()).hexdigest()} for name in sorted(relative_paths)]
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row["file"] + "\0" + row["sha256"] + "\n").encode())
    return {"sha256": digest.hexdigest(), "files": rows}
