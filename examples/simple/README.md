# Phi_0 SIMPLE G1 whole-body benchmark

参考 Psi0 `examples/simple/`，在 Phi_0 原生栈上完成 SIMPLE G1 训练与评测。

## 1. 依赖与子模块

```bash
# 已下载 SIMPLE.zip 时（解压到 third_party/SIMPLE + 链接 data/simple）
./scripts/setup_simple_data.sh

# Python 依赖（lerobot + HTTP serve）
./scripts/setup_simple_env.sh

# 训练数据（需外网）
DOWNLOAD_DATA=1 TASK=G1WholebodyBendPick-v0-psi0 ./scripts/setup_simple_data.sh
```

SIMPLE 仿真闭环（可选，需 Docker / Isaac Sim）：

```bash
cd third_party/SIMPLE
# .env 中 DATA_DIR 已指向 Phi_0/data/simple
docker compose build isaac-sim
```

## 2. 数据

Psi0 发布的 LeRobot 数据（36-d action/state，`meta/stats_psi0.json`）：

```bash
mkdir -p data/simple
# 示例：从 HuggingFace 下载 G1WholebodyBendPick-v0-psi0
# huggingface-cli download USC-PSI-Lab/psi-data simple/G1WholebodyBendPick-v0-psi0.zip --local-dir data/simple
```

目录结构：`data/simple/G1WholebodyBendPick-v0-psi0/`（含 `meta/stats_psi0.json`）。

## 3. 训练

```bash
chmod +x scripts/run_train_simple_g1.sh
TASK=G1WholebodyBendPick-v0-psi0 EXP=experiments/simple_g1_act \
  ./scripts/run_train_simple_g1.sh

# 快速 smoke（50 step）
python scripts/train.py --config-name train_simple_g1_smoke \
  data.simple_root=./data/simple data.simple_repo_id=G1WholebodyBendPick-v0-psi0
```

关键配置见 `configs/train_simple_g1_act.yaml`：
- `robot_action_dim: 36`，`action_head: act`，`future_action_steps: 30`
- VLM 冻结，`img_aug: true`，图像 180×320（与 Psi0 对齐）

## 4. Open-loop 评测

```bash
python examples/simple/openloop_eval.py \
  --checkpoint experiments/simple_g1_act/simple_g1_act.pt \
  --data-root ./data/simple \
  --repo-id G1WholebodyBendPick-v0-psi0
```

## 5. SIMPLE 闭环仿真评测

先启动 policy server（或与 `simple_eval.py` 一键启动）：

```bash
CHECKPOINT=experiments/simple_g1_act/simple_g1_act.pt \
  ./scripts/serve_simple_g1.sh

# 或一键 server + eval
CHECKPOINT=experiments/simple_g1_act/simple_g1_act.pt \
  DATA_DIR=./data/simple/G1WholebodyBendPick-v0-psi0 \
  ./scripts/run_simple_eval.sh --num-episodes 1 --save-video
```

> SIMPLE 客户端默认 `policy=psi0`（HTTP `/act` 协议与 Psi0 相同）。评测视频通常在 `third_party/SIMPLE/data/evals/psi0/`。

## 36-d 动作分解

| 索引 | 语义 |
|------|------|
| [0:14] | 手部关节 |
| [14:28] | 手臂关节 |
| [28:32] | 躯干 + 高度 |
| [32:36] | vx, vy, vyaw, target_yaw |
