# Phi0 + Humanoid-GPT 部署方案

## 1. 概述

**备选**全身跟踪路径：Phi0 publisher 将 unified 512-d 转为 **36-d G1 body qpos @ 50 Hz**，经 ZMQ 发给 Humanoid-GPT 的 ONNX tracker，在 MuJoCo 中跟踪并录屏。

**局限**：**不驱动 Dex3 三指夹爪**（`[346:360]` 在 HGPT 路径丢弃）。要看夹爪 + SONIC 全链路，请用 [Phi0 + SONIC 部署方案](phi0_sonic.md)。

## 2. 架构与数据流

```
Phi0 推理 (Phi-0-wpy, cuda)
  → unified 512-d denorm
  → deploy_mode 分支:
      smpl: SMPL-H [3:346] → FK → GMR → 36-d qpos
      qpos: 直读 [360:396] g1_body_qpos_36（可选 EMA）
  → ZMQ PUB tcp://*:5560, topic "phi0_gmr", msgpack qpos 帧
  → Humanoid-GPT subscriber (Humanoid-gpt-wpy, cpu)
  → ONNX tracker + MuJoCo sim → mp4
```

### 两种 deploy mode

| Mode | 路径 | 适用场景 |
|------|------|---------|
| `smpl`（默认） | unified SMPL → GMR → qpos | 模型推理 |
| `qpos` | 直读 `[360:396]` | GT replay 采集关节角 |

### 训练 vs 部署切片

| 阶段 | SMPL `[3:346]` | Qpos `[360:396]` | Sonic `[396:460]` | Gripper `[346:360]` |
|------|----------------|------------------|-------------------|---------------------|
| **训练 loss** | ✅ | ✅ | ✅ | ✅ |
| **HGPT deploy smpl** | → GMR → qpos | 不用 | 不用 | 丢弃 |
| **HGPT deploy qpos** | 不用 | 直读 → qpos | 不用 | 丢弃 |
| **SONIC deploy v4** | 不用 | 不用 | ZMQ token | ZMQ 左右手 7 维 |

## 3. 关键文件

| 路径 | 作用 |
|------|------|
| `experiments/phi0_hgpt_zmq/phi0_zmq_publisher.py` | HGPT 36-d qpos publisher |
| `experiments/phi0_hgpt_zmq/README.md` | 主文档 |
| `experiments/phi0_hgpt_zmq/PRESEARCH.md` | SMPL→GMR 预研 |
| `scripts/run_pick_tissue_hgpt_zmq_eval.sh` | 一键 eval wrapper |
| `src/phi0/deploy/ref_traj_builder.py` | denorm → deploy qpos |
| `src/phi0/deploy/zmq_protocol.py` | ZMQ 编解码 |
| `Humanoid-GPT-main/experiments/phi0_hgpt_zmq/hgpt_zmq_tracker_sim.py` | HGPT subscriber + sim |

## 4. 运行命令

### 模型推理（HGPT tracker，无夹爪）

```bash
CHECKPOINT=experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt \
EPISODE_IDX=447 USE_GT=0 DEPLOY_MODE=smpl \
CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_pick_tissue_hgpt_zmq_eval.sh
```

### GT qpos replay

```bash
DEPLOY_MODE=qpos USE_GT=1 EPISODE_IDX=447 SHOW_GT_VIEWS=0 \
bash scripts/run_pick_tissue_hgpt_zmq_eval.sh
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DEPLOY_MODE` | smpl | smpl / qpos |
| `USE_GT` | 0 | GT replay 模式 |
| `EPISODE_IDX` | 447 | eval clip |
| `MOTION_SECONDS` | 8 | 录屏时长 |
| `ZMQ_PORT` | 5560 | ZMQ 端口 |
| `EMA_ALPHA` | 0.55 | qpos 平滑系数 |

## 5. 实验结果

日志 `logs/pick_tissue_finetune/openloop_*/hgpt_eval.log`：

- checkpoint：`pick_tissue_xperience_unified_3k_ddp4_fast`
- ep447，`deploy_mode=smpl`，8s → **400 帧 @ 50 Hz**
- 输出：`pick_tissue_ep447_tracker.mp4`
- 推理约 2 分钟（含 VLM 加载 + multi-chunk）

## 6. 仿真 vs 真机

| 模式 | 状态 | 说明 |
|------|------|------|
| **仿真** | ✅ 已测通 | `run_pick_tissue_hgpt_zmq_eval.sh` |
| **真机** | HGPT 自有方案 | `Humanoid-GPT-main/deploy/play_track.py`；**未与 Phi0 ZMQ 打通** |

## 7. 设计要点

### 为何不用 GMR 打 `360:396` 标签？

Pick-tissue **已有** SONIC/WBC 记录的 G1 关节角，直接写入 unified `[360:396]` 即可。GMR 适合「只有人体动捕、没有机器人关节」的数据集。

### WBC 43 维 vs HGPT 36 维

源 parquet 的 43 维 = 29 body + 14 hand。HGPT 36 维 = root(7) + 29 dof；hand 14 维在 unified `[346:360]`，本 HGPT 路径不驱动手模。

## 8. 单元测试

```bash
PYTHONPATH=src pytest tests/unit/test_phi0_hgpt_zmq_gt.py -q
```
