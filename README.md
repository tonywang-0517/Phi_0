# Phi_0

单目 egocentric 视频 + 语言 → **未来 action chunk** 的世界-动作模型。视频侧 [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)（DiT4DiT 风格 hook）；动作侧 16 层 Action DiT（**ACT** 直接回归 或 **FM** flow matching），固定 **256-d** I/O，**仅监督 keypoints 156-d**。

---

## 模型架构

<p align="center">
  <img src="assets/1280X1280.PNG" alt="Phi_0 三塔架构" width="90%">
</p>

**三塔结构**：Cosmos 视频塔（VAE + DiT hook）提供视觉-语言 context；可选 VGGT-Omega 塔提供 3D 场景 register；Action DiT 塔通过交替 cross-attention 融合 context，一次前向预测未来 29 步动作 chunk（proprio 前缀 4 步 + future 29 步）。

| 模块 | 默认 | 说明 |
|------|------|------|
| Cosmos VAE / text | 冻结 | 本地 `checkpoints/nvidia/Cosmos-Predict2.5-2B` |
| Cosmos DiT | 冻结 | `lambda_video=0` 时仅 forward 取 hook |
| Hook | L17, detach | `action_context_mode=full_clip`，17 帧 latent 一次 capture |
| VGGT（可选） | `phi0_dual_vggt` | `vggt_omega_1b_512.pt`，aggregator 冻结 |
| Action 头 | `act` | 16L×16H hidden 1024；`past_action_window_size=4` |

Cross-attn 模式（`action_cross_attn_mode`）：

| 模式 | 偶数层 | 奇数层 | 配置 |
|------|--------|--------|------|
| `interleave_cosmos`（双塔基线） | cross → Cosmos hook | 仅 self+FFN | `phi0_full` |
| `dual_cosmos_vggt`（三塔） | cross → Cosmos | cross → VGGT registers | `phi0_dual_vggt` |

---

## 800 步 Eval 效果

`phi0_act_proprio_800step`（Cosmos 双塔 + ACT + proprio）在 Xperience demo 上的 skeleton 预测可视化（绿=GT，蓝=Pred）：

<p align="center">
  <img src="assets/skeleton_animation.gif" alt="Phi_0 800 step eval skeleton animation" width="90%">
</p>

---

## 设计优势

### 性能

- **低显存**：三塔推理峰值显存 **22GB**。
- **高速度**：A800 三塔 **32Hz**，双塔 **54Hz**。
- **短期记忆**：内置 **17 帧**历史窗口（约 **1 秒**）。

### 训练与算法

- **统一表示**：State-Action 同构设计，Stateₜ = Actionₜ₋₁。
- **强化学习友好**：预测关键点与方差（置信度），支持 PPO 等 RL 微调；额外预测接触力，增强灵巧手抓取奖励设计。
- **缺失数据训练**：Mask Training 和局部 loss 支持不完整 Mocap、无触觉及多模态缺失数据。

### 扩展能力

- **模块化架构**：三塔自由拆装，可外挂 VGGT-Omega 等空间理解模块。
- **可扩展输出**：256 维输出仅监督前 156 维，预留维度支持触觉、表情等新模态。
- **可插拔 Action Head**：支持 MoE 路由或 Agent 动态切换，适配不同垂直场景。
- **稀疏输出**：支持关节稀疏预测与关节级置信度输出。

### 工程与商业化

- **控制器兼容**：兼容 GMT、SONIC、Humanoid-GPT 等下游控制框架。
- **长期记忆解耦**：长期记忆由 Agent 管理，模型专注运动生成。
- **低成本部署**：单卡可部署，训练简单。
- **快速场景迁移**：支持 Action Head 权重独立下载与替换，无需重训整模型。

---

## 源码与权重获取

**集群内一键复制（推荐）**：源码 + 各预训练/实验权重已集中在：

```text
cluster_0:/mnt/data2/wpy/workspace
```

可直接 `rsync` / `cp` 到本机或其他节点，**省去 Cosmos、VGGT、Action checkpoint 等下载时间**。

```bash
# 示例：复制整个 workspace（按需调整目标路径）
rsync -av --progress cluster_0:/mnt/data2/wpy/workspace/ /your/local/workspace/
```

目录内主要资源：

| 路径 | 内容 |
|------|------|
| `Phi_0/` | 本仓库源码 |
| `Phi_0/checkpoints/` | Cosmos-Predict2.5-2B、action stats 等 |
| `Phi_0/experiments/` | 各实验 checkpoint（如 `phi0_act_proprio_800step_latest.pt`） |
| `vggt-omega/checkpoints/` | VGGT-Omega 权重 |
| `FastWAM/` | 依赖库 |

> **注意**：该目录为共享工作区，请勿擅自改动或删除他人文件；仅复制所需内容到自己的工作目录。

GitHub 仓库仅包含**源码**；大文件（权重、实验输出）通过 `.gitignore` 排除，请从集群路径获取。

---

## 快速开始

### 环境

```bash
conda create -n Phi-0-wpy python=3.10 -y && conda activate Phi-0-wpy
pip install -e /path/to/FastWAM
pip install -e /path/to/Phi_0[train,viz]
pip install -e /path/to/vggt-omega   # 三塔 dual 模式
```

Smoke：

```bash
PYTHONPATH=src:/path/to/FastWAM/src python scripts/smoke_test.py
```

### Cosmos 权重（无集群拷贝时）

```bash
bash scripts/download_cosmos_weights.sh
python scripts/verify_weights.py
# 或 export COSMOS25_BASE_MODEL=/path/to/Cosmos-Predict2.5-2B
```

### 训练

```bash
# 双塔基线 ACT+proprio 800 step
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/train.py --config-name train_act_proprio_800 device=cuda mixed_precision=bf16

# 三塔 dual VGGT 800 step
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/train.py --config-name train_act_dual_vggt device=cuda mixed_precision=bf16
```

Checkpoint：`experiments/<name>/<name>_latest.pt`（action expert + step）。

### Eval / 可视化

```bash
python scripts/eval_action.py --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt --config-name train_act_proprio_800
python scripts/benchmark_deploy.py --checkpoint ... --config-name train_act_proprio_800
python scripts/visualize_skeleton.py --predictions experiments/.../benchmark_deploy.json --output-dir ...
```

### Deploy

```bash
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/deploy_g1.py \
  --config-name train_act_proprio_800 \
  --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt \
  --input-video .../stereo_left.mp4 \
  --device cuda
```

输出 JSONL `d_raw[256]`；**仅用 `d_raw[0:156]`** 做 skeleton / 控制。

---

## 技术细节

### 数据流（单 step）

```
Mono RGB clip (17 帧) ──► Cosmos VAE (frozen) ──► latents
Task text ──► Qwen2.5-VL (frozen) ──► Cosmos DiT 内部 prompt
Cosmos DiT L17 hook (detach) ──► 偶数层 cross-attn context (2048-d)

[可选] 同 clip RGB ──► VGGT-Omega aggregator (frozen) ──► scene registers
                      └── 奇数层 cross-attn (272 × 2048, dual 模式)

Proprio 前缀 (4 步) + 未来 horizon (29 步) ──► ActionACTDiT ──► D_raw 256-d
```

### 损失

```text
loss = λ_a · MSE_action + λ_bone · L_bone + λ_bone_hand · L_hand_bone
     + λ_bone_dir · L_dir + λ_hand_mse · L_hand_kp
     (+ λ_v · L_video  当 lambda_video > 0)
```

- MSE 受 `action_is_pad`、`action_dim_is_pad` 掩码；仅 valid 维度/帧参与归一化。
- Bone / hand 辅助损失在反归一化前的 keypoints 空间计算（`src/phi0/losses/bone.py`）。

### Clip 时间轴

| 参数 | 值 | 含义 |
|------|-----|------|
| `control_fps` | 20 Hz | 统一 action 时间轴 |
| `seq_len` | 33 | 约 1.65 s control 步 |
| `action_video_freq_ratio` | 2 | video 取 0,2,4,… → **17 帧** |
| `past_action_window_size` | 4 | proprio 前缀；future chunk = **29 步** |

### D_raw（256 维）

| 切片 | 索引 | 维数 | 训练 loss | 说明 |
|------|------|------|-----------|------|
| `keypoints_52` | 0:156 | 156 | ✅ | 52 关节 × (x,y,z) |
| `legacy_buffer_gap` | 156:211 | 55 | ❌ | 保留对齐 legacy buffer |
| `betas_storage` | 211:227 | 16 | ❌ | 元数据存储槽 |
| `tactile_storage` | 227:237 | 10 | ❌ | 触觉预留 |
| `reserved` | 237:256 | 19 | ❌ | padding |

---

## 配置

| 配置 | 用途 |
|------|------|
| `configs/model/phi0_full.yaml` | Cosmos hook + ACT（双塔基线） |
| `configs/model/phi0_dual_vggt.yaml` | + VGGT dual cross-attn（三塔） |
| `configs/train_act_proprio_800.yaml` | ACT + proprio 800 step |
| `configs/train_act_dual_vggt.yaml` | ACT + proprio + dual VGGT 800 step |

---

## 目录

```
Phi_0/
├── assets/                           # README 展示图（架构图、eval 动图）
├── configs/
│   ├── model/phi0_full.yaml          # 双塔基线
│   ├── model/phi0_dual_vggt.yaml     # 三塔
│   └── train_act_*.yaml
├── scripts/                          # train, eval, deploy, viz
├── src/phi0/
│   ├── models/                       # phi0, action_dit, cosmos, vggt
│   ├── data/                         # xperience, egodex, sequence
│   ├── inference/                    # session, deploy_align
│   ├── losses/                       # bone / hand
│   └── schema/                       # action_schema
├── checkpoints/                      # 权重（gitignore，从集群拷贝）
└── experiments/                      # 实验输出（gitignore）
```

---

## 状态

| 项 | 状态 |
|----|------|
| Keypoints D_raw 256 + dim mask | ✅ |
| Cosmos hook + ACT/FM action 头 | ✅ |
| Proprio 前缀 + 17 帧 video clip | ✅ |
| Bone / hand 辅助 loss | ✅ |
| VGGT dual cross-attn（三塔） | ✅ |
| Deploy 与训练 clip 对齐 | ✅ |
| RL / 接触力 / MoE Action Head | 🔜 架构预留 |
| SMPL-H mesh 可视化 | 可选（需 MANO 许可） |
