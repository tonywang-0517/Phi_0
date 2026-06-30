# 视觉空间加强塔

## 1. 概述

对应代码中的 **VGGT-Omega**，README 称「三塔可选」。

从 RGB 视频提取 **3D 场景 register tokens**，增强 Action DiT 的空间几何感知。在 `dual_vlm_vggt` 模式下，Action DiT **奇数层** cross-attend 到这些 token。

**默认关闭**（双塔基线 `phi0_full`）；开启后变为三塔结构（`phi0_dual_vggt`）。

## 2. 实现类

| 类 | 文件 |
|----|------|
| `VGGTOmegaTower` | `src/phi0/models/vggt/tower.py` |
| `VGGTSmokeTower` | 同上（单测用） |
| 预处理 | `src/phi0/models/vggt/preprocess.py` |

## 3. 架构细节

### 3.1 骨干网络

- **模型**：`vggt_omega.models.VGGTOmega`
- **配置**：`enable_camera/depth/alignment=False`，仅 aggregator
- **权重**：`vggt_omega_1b_512.pt`
- **输入分辨率**：`image_resolution=512`

### 3.2 输出规格

| 参数 | 值 | 说明 |
|------|-----|------|
| Register 维 | `VGGT_REGISTER_DIM = 2048` | frame + inter token concat |
| 每帧 register 数 | `VGGT_NUM_REGISTERS = 16` | |
| 输出形状 | `[B, S×16, 2048]` | S = 输入帧数 |

### 3.3 核心 API

```python
extract_register_context() → [B, S×16, 2048] + mask
```

### 3.4 输入帧策略

| 模式 | 输入 | 配置 |
|------|------|------|
| 默认 | 仅 **最后一帧** | `vggt_use_full_video=false` |
| 三塔全 clip | 完整视频 | `vggt_use_full_video=true` |

## 4. 与 Action Head 的连接

```
VGGT registers [B, S×16, 2048]
  →（可选 vggt_embedding: Linear(2048→hidden)）
  → Action DiT 奇数层 cross-attn target "vggt"
```

### Cross-attn 调度（三塔模式）

| 层 | Cross-attn 目标 |
|----|----------------|
| 偶数层 (0,2,4…) | VLM hidden |
| 奇数层 (1,3,5…) | VGGT registers |

## 5. 训练策略

| 组件 | 梯度 | 说明 |
|------|------|------|
| VGGT aggregator | **完全冻结** | 仅作特征提取 |
| `action_expert.vggt_embedding` | **可训练** | 投影层适配 Action DiT |

## 6. 训练 vs 推理

| 维度 | 训练 | 推理 |
|------|------|------|
| **梯度** | aggregator 冻结，`inference_mode()` 抽 context | 同左 |
| **输入帧** | 通常末帧；可开 `vggt_use_full_video` | 同左 |
| **Context 缓存** | 可预计算 `vggt_ctx` | `ActionInferenceSession` 缓存 |
| **必需输入** | dual 模式需提供 `vggt_video` 或预计算 `vggt_ctx` | 同左 |

## 7. 配置

```yaml
# configs/model/phi0_dual_vggt.yaml
vggt:
  enabled: true
  model_path: checkpoints/vggt/vggt_omega_1b_512.pt
  freeze: true
  image_resolution: 512
  vggt_use_full_video: false

action_expert:
  action_cross_attn_mode: dual_vlm_vggt
```

## 8. 塔间交互总览

```
RGB 帧 + 任务指令
    │
    ├─► [视觉-语言塔] Qwen3VLTower ──► action_ctx [B,S,2048]
    │
    ├─► [视觉-空间塔] VGGTOmegaTower ──► vggt_ctx [B,T×16,2048]  ← 可选
    │
    └─► [Action 输出头] ActionACTDiT
            ↑ 偶数层 cross-attn → VLM
            ↑ 奇数层 cross-attn → VGGT（三塔模式）
            └──► action chunk [B,T,D]
```

## 9. 使用建议

- **双塔基线**（`phi0_full`）：无 VGGT，适合快速迭代与 pick-tissue 主实验
- **三塔**（`phi0_dual_vggt`）：需要更强 3D 空间感知时启用；增加 VGGT 前向开销
- pick-tissue 当前主路径使用 **双塔**（`phi0_xperience_unified.yaml`，VGGT 关闭）
