"""验证完整训练可重复及复现清单对数据改动敏感。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from task2_image_anomaly.src.autoencoder import AutoEncoderDetector
from task2_image_anomaly.src.metrics import evaluate_binary
from task2_image_anomaly.src.provenance import file_manifest


def test_complete_training_repeat_and_confusion_counts():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        # 图像与打乱器各自固定；比较经过反向传播后的完整预测，非仅初始化。
        x = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(8))
        dataset = TensorDataset(x, torch.zeros(4))
        def train():
            loader = DataLoader(dataset, batch_size=2, shuffle=True,
                                generator=torch.Generator().manual_seed(42))
            detector = AutoEncoderDetector(epochs=2, batch_size=2, seed=42)
            history = detector.fit(loader, verbose=False)
            scores, _ = detector.predict(DataLoader(dataset, batch_size=2))
            return history, scores
        first, second = train(), train()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        metrics = evaluate_binary(np.array([0, 0, 1, 1]), np.array([.1, .8, .2, .9]), .5)
        assert [metrics[k] for k in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    finally:
        torch.set_num_threads(previous)


def test_manifest_changes_with_content_and_not_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for root in (a, b):
        (root / "image.png").write_bytes(b"same")
    assert file_manifest(a, ["image.png"]) == file_manifest(b, ["image.png"])
    (b / "image.png").write_bytes(b"changed")
    assert file_manifest(a, ["image.png"])["sha256"] != file_manifest(b, ["image.png"])["sha256"]
