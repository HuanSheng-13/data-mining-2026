"""数据加载与预处理模块：读取 Image_Anomaly_Detection 数据集。"""
import os
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms


def resolve_data_root(root: str) -> str:
    """兼容保留压缩包外层目录的解压方式。"""
    root = Path(root)
    if not (root / "image_anomaly_labels.json").is_file():
        root = root / "Image_Anomaly_Detection"
    if not (root / "image_anomaly_labels.json").is_file():
        raise FileNotFoundError(f"未找到图像异常检测数据: {root}")
    return str(root)


def training_context(fit_subset: Subset) -> dict:
    """绑定实际拟合样本、内容和预处理，防止缓存跨划分/数据版本复用。"""
    dataset = fit_subset.dataset
    digest = hashlib.sha256()
    for index in fit_subset.indices:
        name = dataset.files[index]
        path = Path(dataset.root) / dataset.category / dataset.split / dataset.subdir / name
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {"format_version": 1, "category": dataset.category,
            "fit_digest": digest.hexdigest(), "fit_count": len(fit_subset),
            "transform": repr(dataset.transform)}


class AnomalyImageDataset(Dataset):
    """按类别读取图像。split 取 'train'/'test'；train 下可选子目录 'good'/'bad'。

    Args:
        root: 数据集根目录（含 hazelnut/zipper 两个子目录）
        category: 'hazelnut' 或 'zipper'
        split: 'train' 或 'test'
        subdir: 仅 split='train' 时使用，'good'/'bad'/None（None 表示读取 train 下全部）
        transform: torchvision 预处理
    """

    def __init__(self, root: str, category: str, split: str,
                 subdir: str = None, transform=None):
        assert category in ("hazelnut", "zipper")
        assert split in ("train", "test")
        base = os.path.join(root, category, split)
        if split == "train" and subdir is not None:
            base = os.path.join(base, subdir)
        assert os.path.isdir(base), f"目录不存在: {base}"
        self.files = sorted(
            f for f in os.listdir(base)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.root = root
        self.category = category
        self.split = split
        self.subdir = subdir
        self.transform = transform or self._default_transform()

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def ae_transform():
        """自编码器专用变换：像素归一化到 [0,1]（Sigmoid 输出匹配）。"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        parts = [self.root, self.category, self.split]
        if self.subdir:
            parts.append(self.subdir)
        path = os.path.join(*parts, self.files[idx])
        return self._load(path)

    def _load(self, path: str) -> Tuple[torch.Tensor, int]:
        img = Image.open(path).convert("RGB")  # 灰度图统一转 RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class TestImageDataset(Dataset):
    """测试集读取：按 json 标签返回 (图像, 标签)。"""

    def __init__(self, root: str, category: str, labels: dict, transform=None):
        base = os.path.join(root, category, "test")
        self.files = sorted(
            f for f in os.listdir(base)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.root, self.category = root, category
        self.labels = labels
        self.transform = transform or AnomalyImageDataset._default_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, self.category, "test", fname)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        rel = f"{self.category}/test/{fname}"
        label = 1 if self.labels[rel]["label"] == "bad" else 0  # 1=异常
        return img, label, fname


def load_test_labels(label_json_path: str) -> dict:
    import json
    with open(label_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_normal_train_dataset(dataset: Dataset, calibration_fraction: float = 0.1,
                               seed: int = 42) -> Tuple[Subset, Subset]:
    """把正常训练样本固定切分为拟合集与阈值校准集。

    校准集只含正常样本，因此其分数分位数可在不读取测试标签的前提下确定阈值。
    返回顺序为 ``(fit_subset, calibration_subset)``。
    """
    n = len(dataset)
    if n < 2:
        raise ValueError("至少需要 2 个正常训练样本，才能划分校准集")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction 必须在 (0, 1) 内")
    n_calibration = max(1, int(round(n * calibration_fraction)))
    n_calibration = min(n_calibration, n - 1)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    calibration_indices = sorted(int(i) for i in permutation[:n_calibration])
    fit_indices = sorted(int(i) for i in permutation[n_calibration:])
    return Subset(dataset, fit_indices), Subset(dataset, calibration_indices)
