"""Autoencoder 重建误差异常检测基线。

思想：用正常样本训练自编码器，使其只"擅长"重建正常图像；测试时重建误差大的
图像即为异常。图像级异常分数取重建误差图的像素均值。
"""
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ConvAutoencoder(nn.Module):
    """轻量卷积自编码器：256x256x3 -> 8x8x64 -> 256x256x3。"""

    def __init__(self, in_ch: int = 3, latent: int = 64):
        super().__init__()
        # 编码器：各层输出分辨率依次为 128/64/32/16/8
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(True),        # 128
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),           # 64  (layer2)
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),          # 32  (layer3)
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),         # 16
            nn.Conv2d(256, latent, 4, 2, 1), nn.ReLU(True),      # 8
        )
        # 解码器
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, 4, 2, 1), nn.ReLU(True),   # 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),      # 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),       # 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),        # 128
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid(),      # 256
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return self.dec(z)

    @torch.no_grad()
    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """提取编码器 layer2/layer3 特征并上采样对齐，返回 B x Np x D patch 特征。
        layer2: 64x64x64，layer3: 32x32x128，concat 后维度 192。
        """
        h = x
        for i, layer in enumerate(self.enc):
            h = layer(h)
            if i == 3:      # layer2 输出 (ReLU 后)
                f2 = h
            if i == 5:      # layer3 输出
                f3 = h
        size = f2.shape[-2:]
        f3 = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], dim=1)             # B x 192 x 64 x 64
        b, d, hh, ww = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(b, hh * ww, d)


class AutoEncoderDetector:
    """Autoencoder 异常检测器封装。"""

    def __init__(self, device: str = "cpu", lr: float = 1e-3,
                 epochs: int = 60, batch_size: int = 16, seed: int = 42):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = ConvAutoencoder().to(device)

    def fit(self, train_loader: DataLoader, verbose: bool = True):
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        history = []
        self.model.train()
        for ep in range(self.epochs):
            total, cnt = 0.0, 0
            for x, _ in train_loader:
                x = x.to(self.device)
                rec = self.model(x)
                loss = F.mse_loss(rec, x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * x.size(0)
                cnt += x.size(0)
            sched.step()
            avg = total / max(cnt, 1)
            history.append(avg)
            if verbose and (ep + 1) % 10 == 0:
                print(f"  [AE] epoch {ep+1}/{self.epochs}  train_mse={avg:.6f}")
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (图像级分数, 像素级重建误差图)。"""
        self.model.eval()
        img_scores, maps = [], []
        for x, *_ in loader:
            x = x.to(self.device)
            rec = self.model(x)
            err = (rec - x).pow(2).mean(dim=1)          # B x H x W
            img_scores.append(err.view(x.size(0), -1).mean(dim=1).cpu().numpy())
            maps.append(err.cpu().numpy())
        return np.concatenate(img_scores), np.concatenate(maps, axis=0)

    @torch.no_grad()
    def extract_patch_features(self, loader: DataLoader) -> torch.Tensor:
        """批量提取编码器 layer2/layer3 patch 特征（用于记忆库/最近邻打分）。"""
        self.model.eval()
        feats = []
        for x, *_ in loader:
            x = x.to(self.device)
            feats.append(self.model.encode_patches(x).cpu())
        return torch.cat(feats, dim=0)

    def save(self, path: str):
        torch.save({"state_dict": self.model.state_dict(),
                    "config": {"lr": self.lr, "epochs": self.epochs,
                               "batch_size": self.batch_size, "seed": self.seed}},
                   path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        return self
