"""PatchCore 无监督图像异常检测实现。

方法概述（Roth et al., CVPR 2022）：
1) 使用 ImageNet 预训练的 WideResNet-50 提取正常图像的多尺度 patch 特征；
2) 将训练集全部正常 patch 特征构建为"记忆库"，并用 Coreset 采样压缩，降低存储与
   最近邻搜索开销；
3) 测试时，逐 patch 计算其与记忆库的最近邻距离，取 Top-k 距离均值作为图像级异常分数。

本实现基于 CPU/MPS 环境优化：Coreset 采样与最近邻搜索均采用分块计算，避免显存/内存峰值。
"""
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

# ImageNet 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WideResNetFeatureExtractor(nn.Module):
    """从预训练 WideResNet-50 的 layer2/layer3 提取多尺度特征。"""

    def __init__(self, backbone: str = "wide_resnet50_2", device: str = "cpu"):
        super().__init__()
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        net = models.wide_resnet50_2(weights=weights)
        # 提取 layer2、layer3 的输出
        self.extractor = create_feature_extractor(
            net, return_nodes=["layer2", "layer3"]
        )
        self.extractor.eval().to(device)
        self.device = device
        # 冻结参数（仅作为特征提取器）
        for p in self.extractor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 Bx3xHxW 图像，返回对齐后的 patch 特征 Bx(H'*W')xD。
        layer2 输出 64x64x512，layer3 输出 32x32x1024，双线性上采样到 64x64 后拼接。
        """
        feats = self.extractor(x)
        f2 = feats["layer2"]   # B x 512 x 64 x 64
        f3 = feats["layer3"]   # B x 1024 x 32 x 32
        f3 = nn.functional.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        feat = torch.cat([f2, f3], dim=1)          # B x 1536 x 64 x 64
        b, d, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, d)  # B x 4096 x 1536
        return feat


def _l2_distance_matrix(a: torch.Tensor, b: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """分块计算两批特征间的欧氏距离矩阵，避免一次性峰值内存。"""
    a = a.float()
    b = b.float()
    dists = []
    for i in range(0, a.shape[0], chunk):
        block = a[i:i + chunk]                 # chunk x Da
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab^T
        aa = (block ** 2).sum(dim=1, keepdim=True)      # chunk x 1
        bb = (b ** 2).sum(dim=1, keepdim=True).T        # 1 x Db
        ab = block @ b.T
        d = aa + bb - 2.0 * ab
        d = torch.clamp(d, min=0.0).sqrt()
        dists.append(d)
    return torch.cat(dists, dim=0)


class PatchCore:
    """PatchCore 异常检测器。

    Args:
        device: 'cpu' / 'mps' / 'cuda'
        backbone: 特征提取骨干网络名称
        coreset_ratio: Coreset 采样比例（相对随机采样后的候选池）
        max_pool: 随机采样候选池上限
        top_k: 图像级分数取 Top-k 个 patch 距离的均值
    """

    def __init__(self, device: str = "cpu", backbone: str = "wide_resnet50_2",
                 coreset_ratio: float = 0.1, max_pool: int = 10000, top_k: int = 9,
                 random_state: int = 42, feature_extractor=None):
        self.device = device
        self.feature_extractor = feature_extractor or WideResNetFeatureExtractor(backbone, device)
        self.coreset_ratio = coreset_ratio
        self.max_pool = max_pool
        self.top_k = top_k
        self.random_state = random_state
        self.memory_bank: Optional[torch.Tensor] = None

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """批量提取 patch 特征。"""
        return self.feature_extractor(images.to(self.device)).cpu()

    def build_memory_bank(self, normal_features: torch.Tensor) -> torch.Tensor:
        """从全部正常 patch 特征构建 Coreset 记忆库。

        流程：L2 归一化 -> 随机采样候选池 -> 贪婪远点采样(FPS)压缩到目标规模。
        """
        rng = random.Random(self.random_state)
        feats = normal_features.reshape(-1, normal_features.shape[-1]).float()
        feats = nn.functional.normalize(feats, dim=-1)     # PatchCore 关键：L2 归一化
        n_total = feats.shape[0]

        # 1) 随机采样候选池
        pool_size = min(n_total, self.max_pool)
        idx = rng.sample(range(n_total), pool_size)
        pool = feats[idx]                       # pool_size x D
        # 2) FPS 采样到 coreset 规模
        target = max(1, int(pool_size * self.coreset_ratio))
        core = self._farthest_point_sampling(pool, target)
        self.memory_bank = core
        return core

    def _farthest_point_sampling(self, pool: torch.Tensor, target: int) -> torch.Tensor:
        """分块贪心 FPS：逐点计算到已选集的最小距离，选最大者加入。"""
        n = pool.shape[0]
        if target >= n:
            return pool
        chosen = [0]
        min_dists = _l2_distance_matrix(pool, pool[0:1])     # (n,1)
        # 预归一化平方和以加速后续距离计算
        pool_sq = (pool ** 2).sum(dim=1, keepdim=True)        # (n,1)
        for _ in range(target - 1):
            sel = int(torch.argmax(min_dists))
            chosen.append(sel)
            new_d = pool_sq + pool_sq[sel] - 2.0 * (pool @ pool[sel]).unsqueeze(1)
            new_d = torch.clamp(new_d, min=0.0).sqrt()
            min_dists = torch.minimum(min_dists, new_d)
        return pool[chosen]

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """对一批图像打分。

        Returns:
            img_scores: 图像级异常分数 (B,)
            patch_scores: 每张图的 patch 级距离矩阵 (B, H'*W')
        """
        assert self.memory_bank is not None, "请先调用 build_memory_bank"
        feat = self.extract_features(images)                  # B x Np x D
        feat = nn.functional.normalize(feat, dim=-1)
        bank = self.memory_bank.to(feat.device)
        patch_scores = []
        for f in feat:
            d = _l2_distance_matrix(f, bank)                  # Np x Ncore
            nn_dist, _ = d.min(dim=1)
            patch_scores.append(nn_dist.numpy())
        patch_scores = np.stack(patch_scores, axis=0)
        # 图像级分数 = Top-k patch 距离均值
        k = min(self.top_k, patch_scores.shape[1])
        topk = np.sort(patch_scores, axis=1)[:, -k:]
        img_scores = topk.mean(axis=1)
        return img_scores, patch_scores

    def fit(self, normal_features: torch.Tensor):
        self.build_memory_bank(normal_features)
        return self
