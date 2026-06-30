# Agent + Phi0 部署方案

## 1. 概述

在 Phi0 动作推理之上叠加 **语言 Agent 层**：官方 `Qwen/Qwen3-VL-2B-Instruct`（LangChain + tool calling）理解用户中文指令与 ego/左腕图像，选出技能后路由到对应 Phi0 checkpoint，再交给 **SONIC latent 管线**执行。

### 设计动机

- Psi0 内嵌 VLM 仅作 **action encoder**，**无对话能力**
- Agent 与 action 权重 **完全分离**：语言理解用官方 Instruct 权重，动作预测用 Psi0 微调权重
- 执行段与纯 Phi0+SONIC 方案相同，Agent 只影响 **技能选择**

## 2. 架构与数据流

```
用户指令 + ego/左腕图
  → LangChain RobotAgent（官方 Qwen3-VL）
  → @tool: pick_tissues | throw_rubbish | stay
  → Phi0SkillRouter（按 skill 懒加载 checkpoint）
  → run_pick_tissue_sonic_latent_eval.sh（与纯 SONIC eval 相同）
  → phi0_sonic_latent_zmq_publisher.py
  → ZMQ v4 (64-d token + 左右手 7 维) @ 5556
  → g1_deploy_onnx_ref (TensorRT) + MuJoCo sim → mp4
```

### 技能映射

| Tool | Phi0 instruction | 默认 checkpoint |
|------|------------------|-----------------|
| `pick_tissues` | `pick tissue` | `experiments/pick_tissue_xperience_unified_3k_ddp4_fast/..._latest.pt` |
| `throw_rubbish` | `throw rubbish` | `experiments/throw_rubbish_xperience_unified/...`（缺失则 fallback pick） |
| `stay` | — | 不加载 Phi0、不推 ZMQ |

## 3. 关键文件

| 路径 | 作用 |
|------|------|
| `src/phi0/agent/robot_agent.py` | LangChain Agent 组装 |
| `src/phi0/agent/tools.py` | 三技能 tool 定义 |
| `src/phi0/agent/executor.py` | Phi0SkillRouter |
| `src/phi0/agent/checkpoints.py` | 每 skill checkpoint 注册表 |
| `src/phi0/agent/prompts.py` | 系统提示与 tool call 格式 |
| `scripts/phi0_agent_zmq_sim_demo.py` | Python 主入口 |
| `scripts/run_phi0_agent_zmq_sim_demo.sh` | 一键 wrapper |
| `scripts/phi0_langchain_agent_demo.py` | 仅 Agent（`--dry-run`） |

## 4. 环境与依赖

```bash
pip install -e ".[agent]"   # langchain + langchain-core
```

| 组件 | Conda 环境 |
|------|-----------|
| Phi0 推理 / Agent | `Phi-0-wpy` |
| SONIC sim | `GR00T-WholeBodyControl/.venv_sim` |
| gear_sonic_deploy | TensorRT 编译产物 |

## 5. 运行命令

### 全流程（Agent 自行选技能）

```bash
CUDA_VISIBLE_DEVICES=4 \
GT_PANEL_LAYOUT=top ENABLE_G1_DEBUG_OVERLAY=0 \
bash scripts/run_phi0_agent_zmq_sim_demo.sh \
  --user-instruction '你可以把沙发上的纸巾拿起来么？' \
  --episode-idx 447 \
  --motion-seconds 8 \
  --out-dir logs/agent_sonic_sim_demo
```

### 仅测 Agent（不启 SONIC）

```bash
PYTHONPATH=src python scripts/phi0_langchain_agent_demo.py --dry-run
```

### 跳过 Agent，直接测 SONIC 执行

```bash
bash scripts/run_phi0_agent_zmq_sim_demo.sh --force-skill pick_tissues
```

## 6. 实验结果

**测试场景**：ep447，`MOTION_SECONDS=8`，用户指令「你可以把沙发上的纸巾拿起来么？」

| 产物 | 路径 | 说明 |
|------|------|------|
| Agent 决策 JSON | `logs/agent_sonic_sim_demo/agent_result.json` | 含 `tool_steps`、`selected_skill` |
| 录屏 mp4 | `logs/agent_sonic_sim_demo/agent_pick_tissues_ep447_sonic_latent_model.mp4` | SONIC sim 录屏 |
| 演示 GIF | `assets/agent_pick_tissues_ep447_demo.gif` | README 展示 |

**成功标志**：

- `agent_result.json` 中 `tool_steps` 非空且含 `pick_tissues`
- Agent 回复「可以的，我来帮你拿。」
- Phi0 predict shape `(8, 512)`

**性能**：首轮约 3 分钟（Agent Qwen3-VL + SONIC publisher 各加载一次 VLM）；precompute 日志显示 ep447 推理 400 帧 × 512-d，耗时约 12s（VLM 加载后）。

## 7. 仿真 vs 真机

| 模式 | 状态 | 说明 |
|------|------|------|
| **仿真** | ✅ 已测通 | `run_phi0_agent_zmq_sim_demo.sh` 一键完成 |
| **真机** | 无一键脚本 | 可先 Agent 选 skill，再手动用 precompute npz 推流到 `gear_sonic_deploy` |

真机开环参考 [Phi0 + SONIC 部署方案](phi0_sonic.md) §真机部署。

## 8. 故障排查

| 现象 | 原因 | 处理 |
|------|------|------|
| `tool_steps` 为空 | 模型未输出 `<tool_call>` | 检查 `prompts.py`；有自动重试逻辑 |
| `throw_rubbish` fallback | checkpoint 未训练 | 在 `checkpoints.py` 更新路径 |
| 首轮很慢 | 双 VLM 加载 | 可用 `--precompute-in` 复用 npz 加速录屏 |
