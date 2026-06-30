# 世界模型塔

## 1. 当前状态

**已移除，不再参与主路径。**

历史上 Phi_0 使用 **Cosmos Video2World** 作为「视频塔 / 世界模型塔」，通过 hook hidden states 做 cross-attn，并可做视频生成监督（`lambda_video`）。

迁移后，世界模型功能由 **Qwen3-VL observation encoder** 取代，不再预测未来视频 latent。

## 2. 历史架构（已废弃）

```
RGB 帧
  → Cosmos Video2World
  → hook hidden states
  → Action DiT cross-attention
  → 可选：视频生成监督 (lambda_video)
```

| 组件 | 作用 |
|------|------|
| Cosmos Video2World | 视频世界模型，预测未来帧 latent |
| `lambda_video` | 视频重建损失权重 |
| `predict_video()` | 推理时生成未来视频 |

## 3. 迁移变更

| 变更项 | 迁移前 | 迁移后 |
|--------|--------|--------|
| 视觉 encoder | Cosmos Video2World | Qwen3-VL-2B（Psi0） |
| 视频生成 | `predict_video()` 可用 | `NotImplementedError` |
| `loss_lambda_video` | 可配置 | 固定为 **0** |
| `video_tower` 别名 | Cosmos | 指向 **Qwen3-VL** |
| Cross-attn 模式名 | `interleave_cosmos` | `interleave_vlm`（别名仍兼容） |
| Checkpoint 策略 | 可能含 video tower | `save_action_expert_only=true` |

### 代码证据

- `Phi0.predict_video()` 直接 `NotImplementedError`（`phi0.py`）
- `loss_lambda_video` 固定为 0
- README 明确：「架构已从 Cosmos Video2World hook 迁移至 Qwen3-VL encoder；不再依赖 Cosmos 权重或视频生成路径」

## 4. 当前架构中的「第四塔」

若按「四塔」理解，第 4 塔在当前代码库中 **不存在独立实现**：

```
当前 Phi0 结构（双塔 / 三塔）：

RGB 帧 + 语言指令
    │
    ├─► [视觉-语言塔] Qwen3VLTower ──► action_ctx
    │
    ├─► [视觉-空间塔] VGGTOmegaTower ──► vggt_ctx  （可选）
    │
    └─► [Action 输出头] ActionACTDiT ──► action chunk

[世界模型塔] — 已移除，无独立实现
```

**功能替代**：Qwen3-VL 作为 observation encoder 提供多模态 context，Action DiT 直接预测未来 action，不再经过视频 latent 中间表示。

## 5. 遗留代码与配置

以下仅为向后兼容，与当前主模型无关：

| 遗留项 | 说明 |
|--------|------|
| `interleave_cosmos` / `dual_cosmos_vggt` | 别名映射到 VLM 路径（`action_cross_attn.py`） |
| `cosmos_video_size` 等 | LIBERO legacy eval 工具 |
| `video_tower` 属性 | 指向 `vlm_tower`（Qwen3-VL） |

## 6. 设计取舍

### 移除原因

1. **简化架构**：Cosmos 视频生成路径增加训练与推理复杂度
2. **Psi0 对齐**：Qwen3-VL 已在 EgoDex + HE 上预训练，直接作 encoder 更高效
3. **部署导向**：pick-tissue 等任务只需 action 预测，不需视频生成

### 影响

| 方面 | 影响 |
|------|------|
| 训练 | 无 `lambda_video` 损失；训练更快 |
| 推理 | 无 `predict_video()`；推理更轻 |
| Checkpoint | 仅保存 `action_expert`；VLM 从 `vlm.model_path` 单独加载 |
| 可解释性 | 失去「预测未来画面」的可视化能力 |

## 7. 未来可能方向

README 标注 `RL / MoE Action Head 🔜 预留`，世界模型塔未列入当前路线图。若未来需要世界模型能力，可能方向包括：

- 轻量级 future frame prediction 作为辅助 loss
- 与 RL 结合的 imagination model
- 复用 Qwen3-VL 的多帧理解能力而非独立视频塔

当前无具体实现计划。
