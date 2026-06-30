# Phi0 + SONIC 部署方案

## 1. 概述

pick-tissue **主推部署路径**。Phi0 输出 unified 512-d action，从中切片：

- **`[396:460]`** — SONIC motion_token 64 维
- **`[346:360]`** — Dex3 夹爪 14 维（WBC→deploy 重排）

经 **ZMQ v4 latent** 送入 `gear_sonic_deploy`（C++ TensorRT），驱动 MuJoCo sim 或 G1 真机。

**优势**：唯一覆盖 **三指夹爪 + 全身运动 + sim/真机开环** 的完整方案。

## 2. 架构与数据流

```
Phase 1（可选）: phi0_sonic_latent_zmq_publisher.py --precompute-out
  → VLM + multi-chunk Phi0 推理 → npz (tokens, left7, right7)

Phase 2:
  MuJoCo sim (run_sim_loop_vla_record.py)     # 端口 5555 相机, 5557 debug
       ↑
  g1_deploy_onnx_ref --input-type zmq_manager  # ZMQ 5556, TensorRT
       ↑
  phi0_sonic_latent_zmq_publisher.py          # inline 或 --precompute-in 流式
       ↑
  unified [396:460] + gripper via sonic_zmq_io.py / dex3_gripper.py
```

### 开环说明

默认用数据集 ep447 的 ego/wrist 视频 + GT proprio LUT，**非**机载相机闭环。真机闭环需 camera server（5555）+ 在线推理，见 `GR00T-WholeBodyControl/G1_VISION_TO_GR00T.md`。

## 3. ZMQ v4 协议

| 字段 | 维度 | 说明 |
|------|------|------|
| motion_token | 64 | SONIC deploy encoder 输出 |
| left_hand | 7 | Dex3 左手（deploy 顺序：thumb×3, index×2, middle×2） |
| right_hand | 7 | Dex3 右手 |
| 端口 | 5556 | topic + msgpack 帧 @ 50 Hz |

夹爪顺序转换：`src/phi0/deploy/dex3_gripper.py` → `wbc_hand7_to_deploy()`

## 4. 关键文件

| 路径 | 作用 |
|------|------|
| `scripts/run_pick_tissue_sonic_latent_eval.sh` | sim + deploy + 录 mp4 全流程 |
| `scripts/phi0_sonic_latent_zmq_publisher.py` | 推理 / precompute / ZMQ 流 |
| `scripts/data/replay_pick_tissue_sonic_latent_zmq_v4.py` | GT token replay |
| `src/phi0/deploy/sonic_zmq_io.py` | unified → ZMQ 数组 |
| `src/phi0/deploy/dex3_gripper.py` | WBC↔deploy 夹爪顺序 |
| `src/phi0/deploy/gt_io.py` | Lazy GT proprio LUT |
| `GR00T-WholeBodyControl/gear_sonic_deploy/` | C++ deploy 二进制 |

## 5. 运行命令

### Sim eval（推荐可视化）

```bash
CHECKPOINT=experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt \
CONFIG_NAME=train_pick_tissue_xperience_unified_ddp4_3k \
UNIFIED_EP=447 GT_PANEL_LAYOUT=top ENABLE_G1_DEBUG_OVERLAY=0 \
MOTION_SECONDS=20 CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_pick_tissue_sonic_latent_eval.sh
```

### 真机开环

```bash
# Terminal A: gear_sonic_deploy
./deploy.sh --input-type zmq_manager real --zmq-host 127.0.0.1 --zmq-port 5556
# 站稳后 I 启动，O 急停；真机不要 --disable-crc-check

# Terminal B: 推流 precompute npz
python scripts/phi0_sonic_latent_zmq_publisher.py \
  --precompute-in /tmp/ep447_precompute.npz \
  --episode-idx 447 --zmq-port 5556 --control-fps 50
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `UNIFIED_EP` | 447 | eval clip episode index |
| `GT_PANEL_LAYOUT` | top | GT 视频面板布局 |
| `ENABLE_G1_DEBUG_OVERLAY` | 0 | debug overlay |
| `MOTION_SECONDS` | 20 | 录屏时长 |
| `PRECOMPUTE_IN` | — | 复用已有 npz |
| `FORCE_PRECOMPUTE` | 0 | 强制重新 precompute |
| `SKIP_PRECOMPUTE` | 0 | 跳过 precompute |

## 6. 实验结果

| 产物 | 说明 |
|------|------|
| `assets/pick_tissue_ep447_sonic_latent_eval.gif` | README 展示 GIF |
| `logs/pick_tissue_finetune/sonic_latent_model_*/pick_tissue_ep447_sonic_latent_model.mp4` | 全 episode 831 帧录屏 |
| deploy 日志 | 收到 64D token + 左右手 7 维关节 |

**训练 checkpoint**：`pick_tissue_xperience_unified_3k_ddp4_fast`（3k steps，loss 从 ~0.35 降至 ~0.048）

## 7. 仿真 vs 真机

| 模式 | 命令 | 注意事项 |
|------|------|----------|
| **仿真** | `run_pick_tissue_sonic_latent_eval.sh` | 可用 `--disable-crc-check` |
| **真机** | `gear_sonic_deploy --input-type zmq_manager real` | **禁用** `--disable-crc-check`；站稳后 I 启动 |
| **闭环** | camera server + 在线推理 | 见 `G1_VISION_TO_GR00T.md` |

## 8. 单元测试

```bash
PYTHONPATH=src pytest tests/unit/test_pick_tissue_sonic_latent_pipeline.py -q
```
