# 视觉语言塔

## 1. 概述

冻结的 **Qwen3-VL-2B**（Psi0 HE 微调权重）作为 **纯 encoder**：把 RGB + 语言指令编码为 token 级 hidden states，供 Action DiT cross-attention。

**不承担** action 生成，**不承担** 对话（对话由外挂 LangChain Agent + 官方 Qwen3-VL 权重完成）。

## 2. 实现类

| 类 | 文件 |
|----|------|
| `Qwen3VLTower` | `src/phi0/models/vlm/tower.py` |
| `SmokeVLMTower` | 同上（CPU smoke，无 HF 权重） |
| `load_agent_speech_tower()` | eval 专用 AR 塔（可与 action 塔权重分离） |
| 预处理 | `src/phi0/models/vlm/preprocess.py` |

## 3. 架构细节

### 3.1 骨干网络

- **模型**：`transformers.Qwen3VLForConditionalGeneration` + `AutoProcessor`
- **默认权重**：`configs/model/phi0_full.yaml` → `vlm.model_path`（Psi0 bundle）
- **隐藏维**：`QWEN3VL_HIDDEN_DIM = 2048`（`action_context_dim`）

### 3.2 输入规格

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 180×320（H×W） | Resize NEAREST + CenterCrop（Psi0 对齐） |
| 视角 | ego + left_wrist | pick-tissue 双视角 |
| 语言 | task instruction | 如 `pick tissue` |

### 3.3 核心 API

```python
extract_action_context() → hidden_states[-1]  # [B, S, 2048] + attention_mask
```

取最后一层 hidden states 作为 Action DiT 的 cross-attention context。

### 3.4 可选 AR 生成

- `generate_text()` / `generate_text_from_vlm_batch()`
- 仅 eval 使用，默认 `suppress_mm_tokens=True` 屏蔽视觉 token
- **训练时从不调用**

## 4. 与 Action Head 的连接

```
VLM hidden [B, S, 2048]
  →（可选 text_embedding: Linear(2048→1024)）
  → Action DiT context
  → 偶数层 cross-attend（interleave_vlm 模式）
```

## 5. 权重来源

VLM 权重来自 **Psi0 预训练**（外部项目）：

| 阶段 | 数据集 | 产出 |
|------|--------|------|
| EgoDex 预训练 | EgoDex 200K | `pre.fast.1by1.2601091803.ckpt.ego200k` |
| HE 微调 | Humanoid Everyday 30K | `pre.fast.1by1.2601091803.ckpt.ego200k.he30k` |

Phi0 直接加载并 **冻结**（`freeze_vlm: true`），不重复预训练。

本地路径：`checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/`

## 6. 训练 vs 推理

| 维度 | 训练 | 推理 |
|------|------|------|
| **梯度** | `freeze=True`，`torch.inference_mode()` 抽 context，**无梯度** | `prefill` 时单次 forward |
| **模式** | `set_frozen_towers_eval()` 保持 eval 模式 | 同左 |
| **Context 缓存** | 可在 batch 中预计算 `action_ctx` 跳过塔 forward | `VLMContextCache` 按 instruction 复用 |
| **AR 生成** | **从不调用** | 仅 eval 显式开启 `enable_agent_speech_for_eval` |
| **Agent 语言塔** | 不参与训练 | 可与 Psi0 VLM 分离（`agent_speech_model_path` → 官方 Instruct 权重） |

## 7. 向后兼容

代码中 `video_tower` 别名仍指向 **Qwen3-VL**（非世界模型）：

- 旧名 `interleave_cosmos` → `interleave_vlm`
- 旧名 `dual_cosmos_vggt` → `dual_vlm_vggt`

架构已从 Cosmos Video2World hook 迁移至 Qwen3-VL encoder；不再依赖 Cosmos 权重或视频生成路径。

## 8. 配置

```yaml
# configs/model/phi0_full.yaml
vlm:
  enabled: true
  model_path: checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
  freeze: true
```

pick-tissue 变体（`phi0_xperience_unified.yaml`）使用相同 VLM 配置，Action Head hidden_dim 降为 1024 并加投影层。
