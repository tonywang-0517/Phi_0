# Phi0 系统开发报告

本目录为 Phi0 系统开发报告的技术文档集，基于项目源码、配置、实验日志与部署脚本整理而成。

## 文档索引

### 一、仿真 / 真机部署

| 文档 | 说明 | 成熟度 |
|------|------|--------|
| [Agent + Phi0 部署方案](deploy/agent_phi0.md) | LangChain Agent 语言决策 + Phi0 动作推理 + SONIC 执行 | ✅ 已测通 |
| [Phi0 + SONIC 部署方案](deploy/phi0_sonic.md) | 推荐默认路径：512-d unified → ZMQ v4 → TensorRT deploy | ✅ 已测通 |
| [Phi0 + Humanoid-GPT 部署方案](deploy/phi0_humanoid_gpt.md) | 备选全身跟踪：SMPL/qpos → ZMQ → HGPT tracker | ✅ sim 已测通 |
| [Phi0 + GMT 部署方案](deploy/phi0_gmt.md) | General Motion Tracking 集成 | ❌ 未实现 |

### 二、结构设计

| 文档 | 说明 |
|------|------|
| [Action 输出头](architecture/action_head.md) | Action DiT（ACT / FM）解码未来 action chunk |
| [视觉语言塔](architecture/vlm_tower.md) | Qwen3-VL-2B（Psi0）多模态 encoder |
| [视觉空间加强塔](architecture/vggt_tower.md) | VGGT-Omega 3D scene registers（可选三塔） |
| [世界模型塔](architecture/world_model_tower.md) | 历史 Cosmos 塔已移除；当前架构说明 |

### 三、训练方案

| 文档 | 说明 |
|------|------|
| [预训练](training/pretraining.md) | Psi0 Qwen3-VL 预训练（外部完成，Phi0 冻结加载） |
| [Action 头后训练](training/action_head_posttraining.md) | 冻结 VLM，训练 Action DiT；pick-tissue 主实验 |
| [后期强化学习训练](training/reinforcement_learning.md) | RL 预留与现状 |

## 系统总览

```
                    ┌─────────────────────────────────────┐
                    │  LangChain Agent (官方 Qwen3-VL)     │  可选上层
                    └──────────────┬──────────────────────┘
                                   │ skill routing
                    ┌──────────────▼──────────────────────┐
                    │  Phi0 (Qwen3-VL Psi0 + Action DiT) │
                    │  512-d unified @ 50 Hz              │
                    └──────┬─────────────────┬────────────┘
                           │                 │
              [396:460]+   │                 │  [3:346] or [360:396]
              [346:360]    │                 │
                           ▼                 ▼
              ┌────────────────────┐  ┌──────────────────────┐
              │ gear_sonic_deploy    │  │ Humanoid-GPT tracker │
              │ ZMQ v4 @ 5556        │  │ ZMQ qpos @ 5560      │
              │ + Dex3 夹爪          │  │ 无夹爪               │
              └─────────┬──────────┘  └──────────┬───────────┘
                        │                        │
                        ▼                        ▼
              MuJoCo sim / G1 真机        MuJoCo sim only
```

## 相关文档

- [Unified Action 设计](../unified_action_design.md) — 256-d / 512-d action 布局与 deploy 切片
- [项目 README](../../README.md) — 快速上手、eval 复现命令
- [HGPT ZMQ eval](../../experiments/phi0_hgpt_zmq/README.md) — Humanoid-GPT 集成细节

## 实验基准

横向对比实验统一使用：

- **数据集 clip**：`episode_index=447`（manifest ep2，~831 帧 @ 50 Hz，task=`pick tissue`）
- **Checkpoint**：`experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt`
- **数据根目录**：`Isaac-GR00T/data/pick_tissue_xperience_unified`
