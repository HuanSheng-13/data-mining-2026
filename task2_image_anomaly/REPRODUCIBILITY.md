# 任务二复现与差异定位

## 先明确复现对象

主方案是冻结 ResNet-18 的局部正常特征记忆库；AE 是10轮像素重建基线。正文表格保留原 Windows 参考结果，`results/task2/supplement/` 单列 macOS CPU 补充实验，不能把两套环境的指标拼成同一次实验。

AE 的固定种子控制初始化和训练样本打乱，正常划分默认也使用同一种子。固定种子有助于同一环境重复，但不保证不同系统、指令集、数学库和后端逐位相同。主方法不训练 AE，AE 的波动不应直接解释为主方案失败。

## 运行

使用已安装项目依赖的 Python，在仓库根目录执行。Windows 可将 `python` 换成 `.\.venv\Scripts\python.exe`，macOS/Linux 可换成虚拟环境的 `bin/python`。

单次 AE（两类，显式 CPU、10轮、种子42）：

```sh
python task2_image_anomaly/scripts/run_autoencoder.py --data_root data/image_anomaly --all --epochs 10 --seed 42 --split_seed 42 --threads 2 --device cpu --out_dir results/task2/local-ae
```

固定划分的多种子检查（预先指定0、1、42，另重训一次种子42的 zipper）：

```sh
python task2_image_anomaly/scripts/check_ae_stability.py --data-root data/image_anomaly --out-dir results/task2/local-stability --epochs 10 --seeds 0 1 42 --split-seed 42 --threads 2
python -m pytest task2_image_anomaly/tests -q
```

第二条训练命令会拒绝覆盖已有的实验目录；更换输出目录后再运行。不同种子只改变初始化和打乱顺序，正常校准划分固定，因此统计不能代表重新划分数据、不同数据集或跨环境的不确定性。这里不挑最优种子，也不改动由20张正常校准图像的99%分位数决定的阈值。

根目录 `requirements.txt` 锁定的是 Windows CPU 环境，`torch==2.7.1+cpu`、`torchvision==0.22.1+cpu` 的版本后缀不适用于 macOS。补充实验在独立 macOS 环境安装无该后缀的2.7.1/0.22.1，并完整记录实际依赖；不能声称与 Windows 环境完全相同。使用 macOS 时应参考补充实验的环境记录，不要直接把平台安装错误当作算法错误。

## 每次运行会留下什么

- `metrics.json`：未舍入的指标、阈值、TP/FP/FN/TN、最后一轮训练损失
- `predictions.csv`：文件名、真值、分数、阈值、预测及误判类型
- `reproducibility.json`：Python/库版本、设备与线程、Git状态、配置、逐轮损失、拟合/校准文件名、参与数据的逐文件SHA-256
- `scores.npz`、`calibration_scores.npz`：保留完整分数和文件名，供逐项比较
- `autoencoder.pt`：本地权重，关联训练配置和拟合集指纹，不提交到Git

多种子脚本另保存 `summary.json`（均值、样本标准差、范围、独立重训比较）、`metrics.csv` 和 `evidence/`。`runs/` 中的权重、数组、图表是本地中间产物；没有纳入版本管理。摘要中的 `working_tree_uncommitted` 如实反映运行时状态，不是最终发布状态。

## 两个人的结果不同时，按顺序对比

1. 核对10轮、批大小16、学习率0.001、训练种子42、划分种子42、CPU；CLI默认60轮，不能省略轮数
2. 对比数据总摘要及拟合/校准文件名；数据和划分不同，应先纠正，再比较环境
3. 对比库版本、平台、后端和线程设置，保留双方原始记录
4. 比较完整校准分数、阈值和 `predictions.csv`；AUROC反映排序，F1还受固定阈值影响
5. 同环境独立重训。若本环境稳定、跨环境有差异，可报告跨环境数值差异，不能在未做控制实验前断言唯一原因

已有 Mac CPU 实测中，种子42的1线程与2线程结果完全一致，因此没有证据支持这次差异由1/2线程切换造成。对同学提供的 AUROC=0.5500、F1=0.2000，目前没有其逐图输出，不能倒推出实际误判数，也不能认定某一方操作有误。zipper测试集只有15张异常和32张正常；少量排序或阈值附近样本变化，就会影响这些指标。

## 评价边界

训练异常图像和测试标签均不用于拟合 AE 或选择阈值。补充实验解释10轮基线的稳定性，不宣称充分训练后的上限。原报告融合权重由测试AUROC扫描得到，仅作探索；本次不再按测试结果增加或筛选模型。
