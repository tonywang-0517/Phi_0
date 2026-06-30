# Phi0 + GMT 部署方案

## 1. 现状

**工作区内无 Phi0+GMT 集成实现。**

仅在 `README.md` 设计优势一节提及「控制器兼容：SONIC deploy（推荐）、Humanoid-GPT tracker（备选）、**GMT 等**」，属规划性表述。

独立仓库 `humanoid-general-motion-tracking-master/` 是 UCSD 的 **GMT（General Motion Tracking）** 项目，与 Phi0 之间**没有** ZMQ publisher、数据格式转换或 eval 脚本。

## 2. GMT 项目简介

GMT（General Motion Tracking）是人形机器人通用运动跟踪框架，支持 G1 等机型。

```bash
# GMT 独立运行（与 Phi0 无关）
conda create -n gmt python=3.8
python sim2sim.py --robot g1 --motion walk_stand.pkl
```

## 3. 与其他部署方案对比

| 方案 | 下游执行器 | 传输协议 | 仿真/真机 | 三指夹爪 | 成熟度 |
|------|-----------|---------|----------|---------|--------|
| **Phi0 + SONIC** | gear_sonic_deploy (TensorRT) | ZMQ v4 latent @ 5556 | sim + 真机 | ✅ | ✅ 推荐 |
| **Phi0 + Humanoid-GPT** | HGPT ONNX tracker | ZMQ qpos @ 5560 | MuJoCo sim | ❌ | ✅ 备选 |
| **Phi0 + GMT** | GMT policy | — | GMT 自带 sim2sim | — | ❌ 未集成 |

## 4. 潜在集成路径（待实现）

若要将 Phi0 与 GMT 对接，需自行完成以下工作：

### 4.1 数据格式转换

Phi0 unified 512-d 中可用于 GMT 的切片：

| 切片 | 语义 | GMT 可能用途 |
|------|------|-------------|
| `[3:346]` | SMPL-H body | 人体运动参考 |
| `[360:396]` | G1 body qpos 36 维 | 直接关节角参考 |
| `[396:460]` | SONIC motion_token | 需确认 GMT 是否接受 latent |

### 4.2 建议集成架构

```
Phi0 推理
  → unified 512-d denorm
  → 格式转换层（待开发）
      smpl: [3:346] → GMT motion format
      qpos: [360:396] → GMT joint reference
  → GMT policy
  → G1 sim / 真机
```

### 4.3 参考实现

可参考已实现的部署路径：

- **SONIC 路径**：`src/phi0/deploy/sonic_zmq_io.py` — unified 切片 + ZMQ 推流
- **HGPT 路径**：`src/phi0/deploy/ref_traj_builder.py` — SMPL/qpos → deploy 格式

## 5. 结论

当前 pick-tissue 场景下，**Phi0 + SONIC** 是唯一覆盖夹爪、sim 与真机开环的完整方案。Phi0 + GMT 仅有 README 规划性提及，需从零集成。若未来需要 GMT 作为备选 tracker，建议先明确 GMT 期望的 motion 输入格式（GMT 数据处理代码标注为待发布），再设计 unified → GMT 转换层。
