# Phi_0

单目 egocentric 视频 + 语言 → **未来 action chunk** 的世界模型。视频侧 [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)（DiT4DiT 风格 hook）；动作侧 16 层 Action DiT（**ACT** 直接回归 或 **FM** flow matching），固定 **256-d** I/O，**仅监督 keypoints 156-d**。

---

## 技术方案

### 总览

```
Mono RGB clip (17 帧) ──► Cosmos VAE (frozen) ──► latents
Task text ──► Qwen2.5-VL (frozen) ──► Cosmos DiT 内部 prompt
Cosmos DiT L17 hook (detach) ──► 偶数层 cross-attn context (2048-d)

[可选] 同 clip RGB ──► VGGT-Omega aggregator (frozen) ──► scene registers
                      └── 奇数层 cross-attn (272 × 2048, dual 模式)

Proprio 前缀 (4 步) + 未来 horizon (29 步) ──► ActionACTDiT / ActionFMDiT ──► D_raw 256-d
```

| 模块 | 默认 | 说明 |
|------|------|------|
| Cosmos VAE / text | 冻结 | 本地 `checkpoints/nvidia/Cosmos-Predict2.5-2B` |
| Cosmos DiT | 冻结 | `lambda_video=0` 时仅 forward 取 hook；`freeze_transformer=false` 可 joint FT |
| Hook | L17, detach | `action_context_mode=full_clip`，多帧 latent 一次 capture |
| Action 头 | `act`（实验）/ `fm`（配置默认） | 16L×16H hidden 1024；`past_action_window_size=4` |
| VGGT | `phi0_dual_vggt` 开启 | `vggt_omega_1b_512.pt`，aggregator 冻结，仅训 `vggt_embedding` + action 头 |
| 确定性 | `capture_stochastic=false`, `vae_sample=false` | train/eval/deploy 一致 |

### Action DiT 与 cross-attn

每层：**self-attn →（可选）cross-attn → FFN**。模式由 `action_cross_attn_mode` 控制（`configs/model/phi0_full.yaml`）：

| 模式 | 偶数层 | 奇数层 | 配置 |
|------|--------|--------|------|
| `interleave_cosmos`（基线） | cross → Cosmos hook | 仅 self+FFN | `phi0_full` |
| `dual_cosmos_vggt` | cross → Cosmos | cross → VGGT registers | `phi0_dual_vggt` |
| `all_cosmos` | cross → Cosmos | cross → Cosmos | 消融 |

- **Cosmos context**：`text_embedding(hook_tokens)`，维度 2048（16×128）。
- **VGGT context**：17 帧经官方 balanced resize → aggregator → 每帧 16 个 scene register（去掉 camera token），展平 **272×2048** → `vggt_embedding` → hidden 1024。
- **Proprio**：过去 4 步 normalized action 作为前缀 token，与未来 29 步一起进 DiT；deploy 可用 GT proprio 或自回归 history。

### 损失

```text
loss = λ_a · MSE_action + λ_bone · L_bone + λ_bone_hand · L_hand_bone
     + λ_bone_dir · L_dir + λ_hand_mse · L_hand_kp
     (+ λ_v · L_video  当 lambda_video > 0)
```

- 所有 action 相关项均受 **`action_dim_is_pad`**、**`action_is_pad`** 掩码；仅 valid 维度/帧参与分母归一化。
- Bone / hand 辅助损失在 **反归一化前** 的 keypoints 空间计算（见 `src/phi0/losses/bone.py`）。
- ACT：placeholder 零输入 + 一次前向回归 chunk；FM：rectified flow，推理 4-step Euler。

### 训练数据流（单 step）

1. `SequenceDataset` 采样 clip → `Phi0Processor` 归一化 action  
2. `build_inputs`：pad 帧替换 → VAE 编码 latents；**同一 video** 送 VGGT（dual 模式）  
3. Cosmos `forward_joint_step` → hook context（无 video loss 时 `inference_mode`）  
4. Action 头：`context_emb` + `vggt_context_emb` 预投影后 cross-attn  
5. 仅 **action_expert** 写入 checkpoint（`save_action_expert_only: true`）

---

## 数据与 Clip

### 数据源（demo）

| 数据集 | 视频 | 原生 FPS | 默认帧数上限 |
|--------|------|----------|--------------|
| Xperience | `stereo_left.mp4` | 20 | 256 |
| EgoDex | `0.mp4` | 30 | 256（demo 约 94 帧） |

由 `build_overfit_datasets()` 帧级拼接；**clip 不跨段**。

### 时间轴与 clip 构造（`SequenceDataset`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `control_fps` | 20 Hz | 统一 action 时间轴 |
| `seq_len` | 33 | 约 1.65 s control 步 |
| `clip_stride` | 1 | 滑窗起点步长 |
| `action_video_freq_ratio` | 2 | video 在 control 轴取 0,2,4,… → **17 帧** |
| `past_action_window_size` | 4 | proprio 前缀；未来 chunk = **29 步** |

构造步骤：

1. 按数据集 FPS 读取 native 帧 span（Xperience 33 帧 / EgoDex 49 帧覆盖 33 control 步）。  
2. 线性 resample → 33 步 action + 图像。  
3. 子采样 17 帧 → Cosmos / VGGT 输入；`image_is_pad` 边界帧复制首有效帧（Cosmos 与 VGGT 共用）。  
4. 合法起点：`start + native_span ≤ segment_end`；demo 合计约 **270 clips**。  
5. `batch_size=4`、shuffle → **约 68 step / epoch**；800 step ≈ 12 epoch。

EgoDex 需先跑稀疏 keypoints 预处理：

```bash
python scripts/preprocess_egodex_smplh.py Isaac-GR00T/demo_data/egodex/test/add_remove_lid/0.hdf5
```

---

## D_raw（256 维）

Action DiT 固定 **256-d** 向量 I/O；**监督与 deploy 只用 keypoints 切片**。

### 布局（`src/phi0/schema/action_schema.py`）

| 切片 | 索引 | 维数 | 训练 loss | 说明 |
|------|------|------|-----------|------|
| `keypoints_52` | 0:156 | 156 | ✅ | 52 关节 × (x,y,z)，当前唯一监督 pose |
| `legacy_buffer_gap` | 156:211 | 55 | ❌ | 保留对齐 legacy quat buffer，恒 mask |
| `betas_storage` | 211:227 | 16 | ❌ | loader 可写入 HDF5 元数据，非预测目标 |
| `tactile_storage` | 227:237 | 10 | ❌ | 存储槽 |
| `reserved` | 237:256 | 19 | ❌ | padding |

> Legacy quat 布局（0:211）见 `LEGACY_QUAT_SLICES`；当前 rep=`keypoints`，README 与代码以 **0:156** 为准。

### 掩码

| 字段 | 形状 | 含义 |
|------|------|------|
| `action_dim_is_pad` | `[T, 256]` 或 `[256]` | **True = 不参与 loss** |
| `action_is_pad` | `[T]` | clip 边界无效 control 步 |

- **Xperience**：仅 `0:156` 有 GT；`156:256` 恒 mask。  
- **EgoDex**：`0:156` 内 **逐帧稀疏**（`dim_available_frame`）；batch 取交集。  
- Deploy：`zero_unsupervised_action_dims()` 将 `156:` 置零，避免误用。

Action 统计：`checkpoints/phi0_action_stats.json`（z-score，仅对 valid 维统计）。

---

## 配置与实验

| 配置 | 用途 |
|------|------|
| `configs/model/phi0_full.yaml` | Cosmos hook + ACT（`action_head: act`） |
| `configs/model/phi0_dual_vggt.yaml` | + VGGT dual cross-attn（**`train_full` 默认 model**） |
| `configs/train_full.yaml` | 数据/优化器默认 → `phi0_dual_vggt`（ACT + VGGT） |
| `configs/train_act_proprio_400/800.yaml` | ACT + proprio，Cosmos-only 基线 |
| `configs/train_act_dual_vggt.yaml` | ACT + proprio + dual VGGT，800 step 从零 |

参考实验目录：

- `experiments/phi0_act_proprio_800step/` — Cosmos-only 基线  
- `experiments/phi0_act_dual_vggt_800step/` — dual VGGT  
- `experiments/loss_comparison/` — loss 曲线对比  

Eval / 可视化：

```bash
python scripts/eval_action.py --checkpoint experiments/.../..._latest.pt --config-name train_act_dual_vggt
python scripts/benchmark_deploy.py --checkpoint ... --config-name train_act_dual_vggt
python scripts/visualize_skeleton.py --predictions experiments/.../benchmark_deploy.json --output-dir ...
python scripts/plot_training_loss.py
```

---

## VGGT-Omega（dual 模式）

```bash
# 权重（ModelScope / HF，见 vggt-omega/checkpoints/download_vggt_omega.sh）
pip install -e /path/to/vggt-omega
# 默认: vggt-omega/checkpoints/vggt_omega_1b_512.pt
```

**纯推理 + detach**：`VGGTOmegaTower.extract_register_context` 在 `@torch.no_grad()` 下跑 frozen aggregator（`freeze=True`、`eval()`、权重 `requires_grad=False`）。Scene register **不对 VGGT 反传**；可训练部分仅为 action 头里的 `vggt_embedding`。`_resolve_vggt_context` 对预缓存 register 也会 `.detach()`。

预处理与官方 `load_fn` 一致：宽高比 crop + **balanced** resize（640×480 → 592×448，非 512 拉伸）。

---

## Action chunk：DiT4DiT vs Phi_0 ACT

二者都不是 33 步自回归 teacher forcing，而是 **proprio prefix + future horizon 一次前向**；loss 只监督 future 段（Phi_0 为后 29 步 @ `past_action_window_size=4`）。

| | DiT4DiT (`FlowmatchingActionHead`) | Phi_0 ACT |
|--|-----------------------------------|-----------|
| Future 槽输入 | 对 GT future 加噪：`noisy = (1-t)*action + t*noise` | `zeros_like(future)` 占位 |
| DiT 条件 | AdaLN timestep `t` | 零 timestep（无扩散） |
| 训练目标 | 预测 velocity `noise - action`，MSE | 直接回归 normalized future，MSE + bone 辅助 |
| 推理 | 从 `randn` 多步 Euler 去噪 | 单次前向，`output_proj` 取 future 段 |
| State/proprio | `state_encoder` prepend 到 action tokens | `proprio_encoder` prepend（4 步 history） |

DiT4DiT 的 noisy-action 编码比 zero placeholder 信息更丰富；Phi_0 默认 **ACT**（`action_head: act`）。FM 路径（`action_head: fm`）与 DiT4DiT 更接近，保留作对照、默认不用。

---

## 环境与权重

### Conda `Phi-0-wpy`

```bash
conda create -n Phi-0-wpy python=3.10 -y && conda activate Phi-0-wpy
pip install -e /path/to/FastWAM
pip install -e /path/to/Phi_0[train,viz]
# GPU: torch 2.x + cu128，见 FastWAM README
pip install -e /path/to/vggt-omega   # dual 模式
```

Smoke：

```bash
PYTHONPATH=src:/path/to/FastWAM/src python scripts/smoke_test.py
```

### Cosmos 权重（必需）

```bash
bash scripts/download_cosmos_weights.sh
python scripts/verify_weights.py
# 或 export COSMOS25_BASE_MODEL=/path/to/Cosmos-Predict2.5-2B
```

Revision：`diffusers/base/post-trained`；运行时 **local_files_only**。

---

## 训练

```bash
# 基线 ACT+proprio 800 step（Cosmos hook only）
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/train.py --config-name train_act_proprio_800 device=cuda mixed_precision=bf16

# Dual VGGT 800 step（从零，无 resume）
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/train.py --config-name train_act_dual_vggt device=cuda mixed_precision=bf16

# CPU smoke
python scripts/train.py --config-name train_overfit max_steps=20 device=cpu smoke_action_only=true
```

Checkpoint：`experiments/<name>/<name>_latest.pt`（action expert + step；VGGT/Cosmos 不写入）。

---

## Deploy

```bash
PYTHONPATH=src:/path/to/FastWAM/src \
  python scripts/deploy_g1.py \
  --config-name train_act_dual_vggt \
  --checkpoint experiments/phi0_act_dual_vggt_800step/phi0_act_dual_vggt_800step_latest.pt \
  --input-video .../stereo_left.mp4 \
  --device cuda
```

- 与训练对齐：33 control 步 clip、17 帧 video、GT proprio 前缀、chunk=29。  
- 输出 JSONL `d_raw[256]`；**仅用 `d_raw[0:156]`** 做 skeleton / 控制。  
- Legacy `fastwam_ckpt` 仅加载 action expert，不改 Cosmos 权重。

---

## 目录

```
Phi_0/
├── configs/
│   ├── model/phi0_full.yaml          # 基线
│   ├── model/phi0_dual_vggt.yaml     # + VGGT
│   └── train_act_*.yaml              # 实验 schedule
├── scripts/                          # train, eval, deploy, viz
├── src/phi0/
│   ├── models/
│   │   ├── phi0.py                   # train/deploy 入口
│   │   ├── action_act_dit.py         # ACT 头
│   │   ├── action_fm_dit.py          # FM 头
│   │   ├── action_cross_attn.py      # cross-attn 模式
│   │   ├── cosmos/                   # Video tower
│   │   └── vggt/                     # VGGT tower + preprocess
│   ├── data/                         # xperience, egodex, sequence, video_pad
│   ├── inference/                    # session, deploy_align
│   ├── losses/                       # bone / hand
│   └── schema/                       # action_schema, draw_schema
└── experiments/
```

---

## 状态

| 项 | 状态 |
|----|------|
| Keypoints D_raw 256 + dim mask | ✅ |
| Cosmos hook + ACT/FM action 头 | ✅ |
| Proprio 前缀 + 17 帧 video clip | ✅ |
| Bone / hand 辅助 loss | ✅ |
| VGGT dual cross-attn 消融 | ✅ |
| Deploy 与训练 clip 对齐 | ✅ |
| `lambda_video>0` joint Cosmos FT | 配置支持，默认关闭 |
| SMPL-H mesh 可视化 | 可选（需 MANO 许可） |
