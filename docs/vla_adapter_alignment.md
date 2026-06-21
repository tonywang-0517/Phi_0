# Phi_0 × VLA-Adapter 对齐（第一版）

本次接入目标是让 `Phi_0` 具备和 VLA-Adapter 类似的 eval I/O：

- 输入：`(full_image, wrist_image, state, instruction)`
- 输出：`action chunk [T,7]`，每步格式为  
  `dx, dy, dz, droll, dpitch, dyaw, gripper`

## 新增模块

- `src/phi0/benchmark/adapters.py`  
  - `libero_obs_to_vla()`：LIBERO obs -> VLAObservation
  - `calvin_obs_to_vla()`：CALVIN obs -> VLAObservation
  - `make_vla_prompt()`：VLA 风格 prompt
- `src/phi0/benchmark/policy.py`  
  - `Phi0VLAPolicy`：暴露 `reset()/step()`，可被 benchmark 循环调用
- `src/phi0/benchmark/action_projection.py`  
  - `KeypointToArmActionProjector`：把 Phi_0 的 keypoints 预测投影到 7D arm action
- `scripts/eval_vla_benchmark.py`  
  - 统一 eval 入口：`--benchmark libero|calvin|cavin`

## 运行方式

```bash
cd Phi_0
PYTHONPATH=src python scripts/eval_vla_benchmark.py \
  --benchmark libero \
  --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt \
  --config-name train_act_proprio_800 \
  --libero-suite libero_spatial \
  --output experiments/benchmarks/libero_report.json
```

```bash
cd Phi_0
PYTHONPATH=src python scripts/eval_vla_benchmark.py \
  --benchmark calvin \
  --checkpoint experiments/phi0_act_proprio_800step/phi0_act_proprio_800step_latest.pt \
  --config-name train_act_proprio_800 \
  --calvin-root /path/to/calvin \
  --output experiments/benchmarks/calvin_report.json
```

## 重要说明

- 目前的 7D 控制投影是 **bootstrap heuristic**（便于快速打通评测流程）。
- 若要达到稳定操控效果，建议替换为：
  - 机器人专用 IK / 控制器映射；
  - 或额外训练 keypoints -> arm action 的桥接头。
- `cavin` 参数作为 `calvin` 的别名支持，避免命令拼写差异。

