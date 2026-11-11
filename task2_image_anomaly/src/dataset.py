"""数据加载与预处理模块：读取 Image_Anomaly_Detection 数据集。"""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
