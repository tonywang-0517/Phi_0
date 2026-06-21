# Phi_0 CALVIN Bridge Head

本文档说明如何基于 **CALVIN（非 LIBERO）** 训练并评测 `Phi_0 -> 7D action` 的可训练桥接头。

## 1. 目标与对齐约定

参考 `VLA-Adapter` 的 CALVIN 评测实现，对齐以下约定：

- 观测：`rgb_static` + `rgb_gripper`，统一 resize/crop 到 `224`
- Prompt：`In: What action should the robot take to {instruction.lower()}?\nOut:`
- Open-loop：每次预测 `action chunk`（默认 `8` 步）
- 动作维度：`[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- gripper 处理：桥接头输出 `gripper in [0,1]`，在执行前通过 `process_vla_action()` 做二值化与符号处理
- CALVIN 统计：`chain_sr` / `avg_seq_len` / `per_task_sr`

## 2. 数据准备（下载 CALVIN 资产）

在 `Phi_0` 根目录执行：

```bash
scripts/download_calvin_assets.sh --split debug
```

- 下载后默认目录：
  - `data/calvin/dataset`
  - `data/calvin/calvin_models`
  - `data/calvin/benchmark`
- 下载策略（强制优先级）：
  1. `ModelScope`
  2. `HF-mirror`
  3. 官方 `HuggingFace`
- 脚本会把每个资源的尝试来源与状态写入：
  - `data/calvin/download_report.json`

若要完整 `ABC->D`（超大体量），执行：

```bash
scripts/download_calvin_assets.sh --split ABC
```

## 3. 训练桥接头

默认训练脚本会从 CALVIN 轨迹读取监督信号（`rel_actions`），并调用 `Phi_0` 提取输入特征：

- `keypoints_chunk`：使用 Phi_0 反归一化后 keypoints 切片（前 156 维）
- `latent_norm`：直接使用 Phi_0 归一化 action latent（256 维）

示例（默认 MLP 桥接头）：

```bash
PYTHONPATH=src python scripts/train_calvin_bridge.py \
  --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt \
  --config-name train_act_proprio_800 \
  --calvin-split-dir data/calvin/dataset/task_ABC_D/training \
  --bridge-input-mode keypoints_chunk \
  --bridge-head-type mlp \
  --num-open-loop-steps 8 \
  --epochs 8 \
  --batch-size 1024 \
  --save-dir experiments/calvin_bridge
```

输出：

- 缓存特征：`data/calvin/bridge_cache.npz`
- 最优 checkpoint：`experiments/calvin_bridge/*_best.pt`
- 最后 checkpoint：`experiments/calvin_bridge/*_last.pt`

## 4. 接入 CALVIN Eval

`scripts/eval_vla_benchmark.py` 支持两种动作模式：

- `--action-mode heuristic`：原启发式投影
- `--action-mode bridge`：使用桥接头 checkpoint

桥接头评测示例：

```bash
PYTHONPATH=src python scripts/eval_vla_benchmark.py \
  --benchmark calvin \
  --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt \
  --config-name train_act_proprio_800 \
  --calvin-root data/calvin \
  --action-mode bridge \
  --bridge-checkpoint experiments/calvin_bridge/bridge_xxx_best.pt \
  --bridge-input-mode keypoints_chunk \
  --num-open-loop-steps 8 \
  --output experiments/benchmarks/calvin_bridge_report.json
```

输出 JSON 包含：

- `avg_seq_len`
- `chain_sr`
- `per_task_sr`
- `task_stats`

## 5. 建议

- 先用 `--split debug` 验证 pipeline，再切换到 `ABC`
- `keypoints_chunk` 更贴合语义结构，`latent_norm` 通常更易训练稳定
- 若出现 gripper 不稳定，可提高 `--gripper-loss-weight`（如 `3.0`）
