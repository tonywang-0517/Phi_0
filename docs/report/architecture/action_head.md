# Action 输出头

## 1. 概述

Action 输出头将视觉-语言（及可选空间）上下文 + proprio/历史动作，解码为未来 **action chunk**（多步联合预测）。是 Phi0 中 **唯一默认可训练** 的模块。

支持多种 action 维度：

| 维度 | 场景 | 配置 |
|------|------|------|
| 256-d | Legacy Xperience keypoints | `phi0_full.yaml` |
| 512-d | pick-tissue unified（SMPL+机器人+SONIC） | `phi0_xperience_unified.yaml` |
| 7-d / 36-d / 100-d | LIBERO / G1 / SONIC 线 | 对应 baseline 配置 |

## 2. 实现类

| 类 | 文件 | 模式 |
|----|------|------|
| `ActionACTDiT` | `src/phi0/models/action_act_dit.py` | **ACT**：直接回归（pick-tissue 默认） |
| `ActionFMDiT` | `src/phi0/models/action_fm_dit.py` | **FM**：rectified flow matching |
| `ActionFlowMatching` | `src/phi0/models/action_fm_scheduler.py` | FM 训练/推理调度 |
| `build_action_expert()` | `src/phi0/models/phi0.py` | 工厂函数 |

底层 DiT block 来自 FastWAM：`fastwam.models.wan22.wan_video_dit.DiTBlock`。

## 3. 架构细节

### 3.1 默认配置（`phi0_full.yaml`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 2048 | 与 VLM/VGGT 对齐时可省投影 |
| `ffn_dim` | 8192 | FFN 宽度 |
| `num_layers` | 6 | DiT 层数 |
| `num_heads` / `attn_head_dim` | 16 / 128 | 注意力 |
| `raw_action_dim` | 256 | 输出维度 |

### 3.2 pick-tissue 512-d 变体（`phi0_xperience_unified.yaml`）

| 参数 | 值 |
|------|-----|
| `hidden_dim` | 1024 |
| `num_layers` | 4 |
| `num_heads` | 4 |
| `raw_action_dim` | 512 |

VLM 仍输出 2048-d，经 `text_embedding: Linear(2048→1024)` 投影。

### 3.3 核心子模块

| 模块 | 作用 |
|------|------|
| `action_encoder` / `proprio_encoder` | `Linear(D→hidden)`，proprio 前缀不叠加 position embed |
| `output_proj` | `Linear(hidden→D)`，零初始化（VLA-Adapter 风格） |
| `text_embedding` / `vggt_embedding` | 塔输出维 ≠ `hidden_dim` 时的线性投影 |
| `position_embedding` | 可选 `nn.Embedding(max_seq_len, hidden)` |
| `future_placeholder_perturbation` | ACT 训练时对零 future slot 加可学习扰动 |
| `dit4dit_prefix_query` | `prefix_encoder` + `Dit4DiTActionEncoder` |

## 4. Cross-attention 调度

由 `action_cross_attn_mode` 控制（`src/phi0/models/action_cross_attn.py`）：

| 模式 | 偶数层 (0,2,4…) | 奇数层 (1,3,5…) | 配置 |
|------|-----------------|-----------------|------|
| `interleave_vlm` | cross → VLM | 仅 self-attn + FFN | `phi0_full` |
| `dual_vlm_vggt` | cross → VLM | cross → VGGT registers | `phi0_dual_vggt` |
| `all_vlm` | cross → VLM | cross → VLM | — |
| `self_only` | 无 cross-attn | 无 cross-attn | — |

## 5. ACT 前向流程

```
pre_dit（编码 action/proprio + 拼 context）
  → N 层 DiTBlock（self-attn + 条件 cross-attn + FFN，t_mod 全零）
  → post_dit（取 future 段 output_proj）
  → pred [B, T_future, D]
```

**损失**：`λ_action · MSE(pred, target)`，带 `action_is_pad` / `action_dim_is_pad` 掩码。

## 6. FM 前向流程

```
Dit4DiTActionEncoder 编码 noisy action + 正弦 timestep embedding
  → time_embedding + time_projection 生成 AdaLN 调制
  → 训练：预测 velocity v = source - clean，MSE on masked dimensions
  → 推理：Euler 积分（num_inference_timesteps，默认 4 步）
```

## 7. Proprio 拼接

`src/phi0/models/action_proprio.py`：

- `past_action_window_size` 帧 proprio 作为前缀 token
- legacy：4 帧；pick-tissue：1 帧
- 与 future horizon 拼接后送入 DiT

## 8. 训练 vs 推理

| 维度 | 训练 | 推理 |
|------|------|------|
| **模式** | ACT→MSE；FM→velocity MSE + Beta 采样 t | ACT→单次前向；FM→Euler 去噪 |
| **Checkpoint** | `save_action_expert_only=true` | 可只加载 `action_expert` |
| **Proprio** | 从 GT action 序列切分 | deploy 用历史预测或 GT hold；滚动更新 |
| **输出后处理** | — | `zero_unsupervised_action_dims` 清零未监督维 |

**推理入口**：`ActionInferenceSession` → `src/phi0/inference/session.py`

```
prefill_from_video_clip(video, instruction) → VLM extract + embed contexts
predict(num_frames) → Action DiT 出 normalized chunk
denormalize → 经 processor 反归一化
```

## 9. 配置变体对照

| 配置 | Action 头 | Cross-attn | VGGT | action 维 |
|------|-----------|------------|------|-----------|
| `phi0_full.yaml` | ACT (6层, 2048 hidden) | `interleave_vlm` | 关 | 256 |
| `phi0_dual_vggt.yaml` | 同上 | `dual_vlm_vggt` | 开 | 256 |
| `phi0_xperience_unified.yaml` | ACT (4层, 1024 hidden) | `interleave_vlm` | 关 | 512 |
