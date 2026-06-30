# 后期强化学习训练

## 1. 现状

**Phi0 与当前 GR00T finetune 配置中均未启用 RL 训练。**

README 标注 `RL / MoE Action Head 🔜 预留`。PPO 相关代码存在于 `GR00T-WholeBodyControl/gear_sonic/trl/`，属于独立/预留能力，与 pick-tissue Phi0 训练管线 **未串联**。

## 2. 三阶段训练体系中的位置

```mermaid
flowchart LR
    A[阶段1: VLM 预训练<br/>Psi0 ✅] --> B[阶段2: Action 头后训练<br/>Phi0 ✅]
    B --> C[阶段3: 强化学习<br/>🔜 未实现]
```

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1. 预训练 | Psi0 Qwen3-VL | ✅ 外部完成 |
| 2. Action 头后训练 | 冻结 VLM + 训 Action DiT | ✅ 主路径已跑通 |
| 3. 后期 RL | 策略优化、在线适应 | ❌ 未落地 |

## 3. 代码库中的 RL 相关痕迹

### 3.1 Phi0 项目

| 位置 | 内容 | 状态 |
|------|------|------|
| `README.md` | `RL / MoE Action Head 🔜 预留` | 规划性 |
| 训练配置 | 无 RL 相关 loss 或 callback | 未实现 |
| `scripts/train.py` | 仅监督学习 | 未接入 RL |

### 3.2 GR00T 项目

| 位置 | 内容 | 状态 |
|------|------|------|
| 训练配置 | `add_rl_callback: false` | 配置项存在但未启用 |
| — | 无端到端 RL finetune 脚本 | 未落地 |

### 3.3 GR00T-WholeBodyControl

| 位置 | 内容 | 状态 |
|------|------|------|
| `gear_sonic/trl/trainer/ppo_trainer.py` | PPO 训练器 | 基础设施 |
| `gear_sonic/eval_agent_trl.py` | PPO eval agent | 独立工具 |
| pick-tissue Phi0 管线 | — | **未串联** |

## 4. 当前训练闭环

Phi0 当前采用 **监督学习 + 开环 eval** 闭环：

```
数据采集 (Isaac-GR00T teleop)
  → 数据转换 (unified 512-d)
  → Action DiT 监督训练 (MSE loss)
  → 开环 eval (dataset clip → SONIC sim)
  → 可选：真机开环 deploy
```

**缺失环节**：在线交互、奖励信号、策略梯度更新。

## 5. 潜在 RL 接入方向

### 5.1 基于 SONIC deploy 的 sim RL

```
Phi0 policy (frozen or fine-tuned)
  → SONIC deploy (MuJoCo sim)
  → 任务奖励（如抓取成功、轨迹跟踪误差）
  → PPO / SAC 更新 Action DiT
```

参考入口：`GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py`

### 5.2 基于真机数据的 offline RL

```
真机 teleop 数据 (已有)
  → 行为克隆 baseline (当前监督训练)
  → 可选：CQL / IQL 等 offline RL 改进
```

优势：无需在线探索风险；数据已有 pick-tissue 831 ep。

### 5.3 VLA + RL 混合

```
阶段2 监督预训练 (当前)
  → 阶段3 RL 微调 (小学习率)
  → 保持 VLM 冻结，仅更新 Action DiT
```

与当前 `save_action_expert_only` 策略兼容。

## 6. 技术挑战

| 挑战 | 说明 |
|------|------|
| **奖励设计** | pick-tissue 任务需定义抓取成功、轨迹质量等奖励 |
| **Sim-to-real gap** | MuJoCo sim 奖励与真机表现可能不一致 |
| **计算开销** | 在线 RL 需频繁 policy rollout + VLM 前向 |
| **安全** | 真机 RL 探索需急停、力矩限制等安全机制 |
| **Action 空间** | 512-d unified 空间大，RL 探索效率低 |

## 7. 与并行 baseline 的关系

| 项目 | RL 状态 |
|------|---------|
| Phi0 | 未实现 |
| GR00T N1.7 | `add_rl_callback: false` |
| Pi0.5 | 无 RL 集成 |
| Psi0 | 无 RL 集成 |
| GR00T-WholeBodyControl / gear_sonic | PPO 基础设施，未接入 Phi0 |

## 8. 建议实施路径（待决策）

若未来启用 RL，建议分步：

### Step 1：Sim RL 验证

1. 在 MuJoCo + SONIC deploy 环境定义任务奖励
2. 用 `gear_sonic/trl/` PPO 基础设施做 proof-of-concept
3. 从监督 checkpoint 初始化，小学习率微调 Action DiT

### Step 2：Offline RL 探索

1. 在现有 831 ep teleop 数据上尝试 offline RL 算法
2. 对比监督 baseline（当前 3k→23k loss ~0.024）

### Step 3：真机在线 RL（远期）

1. 完善安全机制（急停、力矩限制、人机协作）
2. 小批量真机 rollout + 离线更新
3. 参考 `G1_VISION_TO_GR00T.md` 闭环架构

## 9. 结论

后期强化学习训练在 Phi0 当前路线图中 **处于预留状态**。主路径为监督 Action 头后训练 + 开环 eval，已产出可用 checkpoint 与 SONIC deploy 演示。RL 接入需先明确奖励设计、仿真环境与安全策略，再复用 `GR00T-WholeBodyControl/gear_sonic/trl/` 中的 PPO 基础设施。

## 10. 相关文件

| 类别 | 路径 |
|------|------|
| PPO 训练器 | `GR00T-WholeBodyControl/gear_sonic/trl/trainer/ppo_trainer.py` |
| PPO eval | `GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py` |
| 真机闭环架构 | `GR00T-WholeBodyControl/G1_VISION_TO_GR00T.md` |
| Phi0 README 预留 | `README.md` — `RL / MoE Action Head 🔜` |
