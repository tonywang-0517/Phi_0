# Unified Action 设计（256-d / 512-d）

Phi-0 有两套 action 表示：**legacy D_raw 256**（Xperience keypoints）与 **unified 512**（SMPL-H + G1 机器人 + SONIC latent）。模型 I/O 宽度由配置 `action_token_dim` 决定；pick-tissue 走 512。

源码：`src/phi0/schema/unified_action_schema.py`、`src/phi0/schema/action_schema.py`。

---

## 1. Legacy D_raw（256-d）

| 切片 | 索引 | 维数 | Loss | 说明 |
|------|------|------|------|------|
| `keypoints_52` | 0:156 | 156 | ✅ | 52 关节 × (x,y,z) |
| `legacy_buffer_gap` | 156:211 | 55 | ❌ | 对齐 legacy buffer |
| `betas_storage` | 211:227 | 16 | ❌ | 元数据槽 |
| `tactile_storage` | 227:237 | 10 | ❌ | 触觉预留 |
| `reserved` | 237:256 | 19 | ❌ | padding |

Deploy / skeleton 可视化只用 `d_raw[0:156]`。

---

## 2. Unified 512-d 布局

```
[0:346)   SMPL-H 语义（root delta、51×rot6d、contact、tactile）
[346:360) G1 Dex3 夹爪 14 维（WBC 顺序）
[360:396) G1 body qpos 36 维（root xyz + quat wxyz + 29 dof）
[396:460) SONIC motion_token 64 维
[460:512) reserved padding（置零，不参与 loss）
```

### 2.1 SMPL-H 语义 `[0:346)`

| 字段 | 索引 | 维数 | 说明 |
|------|------|------|------|
| `root_trans_local` | 0:3 | 3 | 骨盆平移 **delta**（相对 proprio `State_t`） |
| `root_rot6d` | 3:9 | 6 | 骨盆全局朝向（6D rot） |
| `joint_rot6d_local_51` | 9:315 | 306 | 关节 1–51 的 parent-local 6D rot |
| `contacts_body21` | 315:336 | 21 | 身体接触 |
| `tactile_fingertips_10` | 336:346 | 10 | 指尖触觉（Xperience 有，pick-tissue 通常无） |

**不在 buffer 内、需 FK / 数据集 GT 推导：** `joints_world_52`、`root_trans_world`、`smpl_pose_aa`、`betas`。

### 2.2 机器人尾 `[346:396)`

**夹爪 `[346:360)` — WBC / unified 顺序（每手 7 维）：**

```
index×2, middle×2, thumb×3
```

来源：Isaac-GR00T `action.wbc` 手关节段。  
Deploy / MuJoCo **执行顺序不同**（thumb×3, index×2, middle×2），出口必须重排，见 `src/phi0/deploy/dex3_gripper.py` → `wbc_hand7_to_deploy()`。

**Body qpos `[360:396)`：**

```
[root_xyz(3), root_quat_wxyz(4), body_dof29(29)]
```

Pick-tissue **标签来自采集**，不是 GMR 反解：

- 29 dof：`action.wbc` body 段
- root 四元数：`observation.root_orientation`
- root xyz：`observation.base_trans`（v2.8+）；旧 parquet 回退 smpl 骨盆 proxy

实现：`src/phi0/data/g1_qpos_from_wbc.py`、`g1_qpos_teacher.py`。

### 2.3 SONIC latent `[396:460)`

64-d `motion_token`，与 gear_sonic deploy encoder 输出一致。Rebuild 时用 deploy encoder 离线写入（`attach_sonic_motion_token_to_parquet_rows`）。

### 2.4 Reserved `[460:512)`

52 维 padding，恒零，`action_dim_is_pad=True`。

---

## 3. 数据集 dim mask（训练 loss）

由 `dim_mask_for_dataset(name)` 决定哪些维度参与 MSE；loader 设 `action_dim_is_pad = ~mask`。

### `g1_sonic`（pick-tissue unified）

| 字段 | 监督 |
|------|------|
| `root_trans_local` | ❌（无可靠 mocap 骨盆里程计） |
| `root_rot6d` + `joint_rot6d_local_51` + `contacts_body21` | ✅ |
| `tactile_fingertips_10` | ❌ |
| `g1_gripper_joints_14` | ✅ |
| `g1_body_qpos_36` | ✅ |
| `sonic_motion_token_64` | ✅ |
| `reserved_52` | ❌ |

等价于：**`[3:346]` SMPL 体 + `[346:360]` 夹爪 + `[360:396]` qpos + `[396:460]` sonic token**；`[0:3]` root delta 与 `[460:512]` reserved 不监督。

Eval 时 `zero_unsupervised_unified_action_dims()` 会把未监督维清零，**不会**误清 sonic/gripper 切片。

### `xperience`（HDF5 unified）

- 监督 SMPL 语义 + tactile；**不**监督 gripper / qpos / sonic token。

---

## 4. 数据构建（pick-tissue）

### 4.1 目录结构（Isaac-GR00T/data）

| 路径 | 说明 |
|------|------|
| `2026-06-23-*` … | 原始 teleop session（LeRobot v2.1） |
| `data.json` | valid episode 索引 manifest |
| `pick_tissue_valid` | 合并后的有效 episode（607 ep / ~442k frames） |
| **`pick_tissue_xperience_unified`** | **512-d Phi-0 训练集**（本仓库主路径） |
| `pick_tissue_sonic_unified` | 43-d state / 100-d action（Pi0.5 / GR00T SONIC 格式，另一条线） |

### 4.2 512-d unified 转换

```bash
cd Phi_0
/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python scripts/data/isaac_groot_to_xperience_unified_lerobot.py \
  --data-root /mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid \
  --out-dir /mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified \
  --num-workers 8
```

- `CODE_VERSION=v2.8`（当前）
- 输出 `meta/stats_pick_tissue_unified.json`（z-score norm）
- 视频：`observation.images.ego_view`、`observation.images.left_wrist`

### 4.3 从 raw 一键 rebuild + eval smoke

```bash
bash scripts/run_pick_tissue_from_raw_rebuild_eval.sh
```

### 4.4 视频 predecode（训练加速）

```bash
bash scripts/data/run_predecode_pick_tissue_4gpu.sh
# 或单进程：
PYTHONPATH=src python scripts/data/predecode_lerobot_videos.py \
  --dataset-root /path/to/pick_tissue_xperience_unified \
  --backend cv2
```

### 4.5 Episode 索引

`episode_index`（unified LeRobot）与 manifest `(session, ep)` 的映射：

```python
from phi0.data.pick_tissue_episode_map import manifest_ep_to_unified_episode_index
# manifest ep2 -> unified episode_index 447
```

---

## 5. 训练

配置：`configs/train_pick_tissue_xperience_unified_ddp4_{8k,3k,23k}.yaml`

| 参数 | 典型值 |
|------|--------|
| `control_fps` | 50 Hz |
| `seq_len` | 33（~0.66s 窗口） |
| `action_token_dim` | 512 |
| mask | `g1_sonic` |
| stats | `stats_pick_tissue_unified.json` |

```bash
# 4×GPU DDP，8k steps
bash scripts/run_train_pick_tissue_xperience_unified_ddp4_8k.sh

# 快速 3k（experiments/pick_tissue_xperience_unified_3k_ddp4_fast）
bash scripts/run_train_pick_tissue_xperience_unified_ddp4_3k.sh

# 续训到 23k
bash scripts/run_train_pick_tissue_xperience_unified_ddp4_23k.sh
```

Checkpoint：`experiments/<name>/pick_tissue_xperience_unified_act_latest.pt`（默认 **action_expert only**）。

---

## 6. 部署路径

### 6.0 默认 eval clip：`episode_index=447`

Pick-tissue 回归 / 开环 eval / Agent 说话 demo **默认使用同一条 clip**：

| 字段 | 值 |
|------|-----|
| unified `episode_index` | **447**（manifest session **ep2**） |
| dataset clip row | **318688**（shuffle 后；`clip_dataset_index_for_episode()` 映射） |
| 帧数 | ~831 @ 50 Hz |
| task | `pick tissue` |
| 参考录屏 | `logs/pick_tissue_finetune/sonic_latent_model_20260628_122146/pick_tissue_ep447_sonic_latent_model.mp4` |

脚本默认值：`UNIFIED_EP=447`（`run_pick_tissue_sonic_latent_eval.sh`）、`--episode-idx 447`（`vlm_agent_speech_demo.py`）。

### 6.1 SONIC latent（推荐 pick-tissue 全链路 eval）

模型输出 unified → 取 `[396:460]` token + `[346:360]` 夹爪 → ZMQ v4 → `g1_deploy_onnx_ref` → MuJoCo。

```
scripts/phi0_sonic_latent_zmq_publisher.py   # 推理 / precompute / ZMQ 流
scripts/run_pick_tissue_sonic_latent_eval.sh   # sim + deploy + 录 mp4
src/phi0/deploy/sonic_zmq_io.py                # unified → (tokens, left7, right7)
src/phi0/deploy/dex3_gripper.py                # WBC → deploy 夹爪重排
```

**两阶段 eval（默认）：**

1. Phase 1：离线 `--precompute-out`（Lazy GT proprio LUT，无 sim）
2. Phase 2：sim + deploy + `--precompute-in` 秒级起流

**推荐可视化配置**（top panel + 无 marker，全 episode）：

```bash
CHECKPOINT=/path/to/pick_tissue_xperience_unified_act_latest.pt \
CONFIG_NAME=train_pick_tissue_xperience_unified_ddp4_3k \
UNIFIED_EP=447 \
GT_PANEL_LAYOUT=top \
ENABLE_G1_DEBUG_OVERLAY=0 \
MOTION_SECONDS=20 \
CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_pick_tissue_sonic_latent_eval.sh
```

| 变量 | 说明 |
|------|------|
| `CHECKPOINT` | 设则 model eval；不设则 GT replay |
| `UNIFIED_EP` | unified `episode_index`（447 = manifest ep2） |
| `GT_PANEL_LAYOUT` | `top`（ego+wrist 上 / sim 下）或 `inset` |
| `ENABLE_G1_DEBUG_OVERLAY` | `0` = 无 cyan/purple marker；`1` = inset 调试 marker |
| `MOTION_SECONDS` | 推理时长；长于 episode 时自动 clamp |
| `SKIP_PRECOMPUTE=1` | 跳过 phase 1（需已有 `sonic_latent_precompute.npz`） |
| `FORCE_PRECOMPUTE=1` | 强制重跑 precompute |

**仅离线推理 + npz：**

```bash
python scripts/phi0_sonic_latent_zmq_publisher.py \
  --checkpoint /path/to.ckpt \
  --config-name train_pick_tissue_xperience_unified_ddp4_3k \
  --episode-idx 447 \
  --motion-seconds 20 \
  --precompute-out logs/ep447_precompute.npz \
  --device cuda
```

**从 npz 重放 ZMQ（不加载 VLM）：**

```bash
python scripts/phi0_sonic_latent_zmq_publisher.py \
  --precompute-in logs/ep447_precompute.npz \
  --episode-idx 447 \
  --zmq-port 5556
```

### 6.3 VLM Agent 说话（eval 可选，与 action 解耦）

- **默认关闭**：训练、`predict()`、开环 publisher **均不**调用 `generate`
- **显式开启**：`enable_agent_speech_for_eval(True)`（首次 `prefill` 之前）
- **只做一次**：`run_agent_speech_once()` 在**首帧** VLM 输入快照上 AR；`refresh_*` / 多 chunk `predict` **不会**再次生成

```python
session.enable_agent_speech_for_eval(True)
session.prefill_from_video_clip(video_bcthw, instruction)
agent_text = session.run_agent_speech_once(gen_cfg=GenerateTextConfig(max_new_tokens=128))
action = session.predict(num_frames=8)
```

实现：`src/phi0/models/vlm/tower.py`。Demo（**默认 ep447 真实 ego/wrist 帧**）：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/vlm_agent_speech_demo.py \
  --enable-agent-speech --episode-idx 447 --skip-action
```

输出：`logs/pick_tissue_finetune/agent_speech_ep447_<ts>.txt`

**说明**：当前 Psi0 HE 微调 VLM 在 `generate` 时会倾向输出 vision token；`GenerateTextConfig(suppress_mm_tokens=True)`（默认）会屏蔽 MM 词表，使 AR 走纯文本。语言质量取决于 VLM 微调程度，与 action 路径独立。

**Action 与 Agent 权重分离**（可选）：`model.vlm.model_path` 始终用于 action train/infer（Psi0）；eval agent 可单独设：

```yaml
vlm:
  model_path: ./checkpoints/psi0/...   # action（默认）
  agent_speech_model_path: Qwen/Qwen3-VL-2B-Instruct  # eval AR only；null = 复用 model_path
```

或 demo / session：`--agent-speech-model-path Qwen/Qwen3-VL-2B-Instruct`。对照测试：`pytest tests/unit/test_vlm_official_weights_restore_speech.py -s`

### 6.2 Humanoid-GPT ZMQ（tracker sim，无 Dex3 手模）

Publisher 发 **36-d body qpos @ 50Hz**；夹爪 `[346:360]` 在 HGPT sim 阶段丢弃。

| `DEPLOY_MODE` | 路径 |
|---------------|------|
| `smpl`（默认） | unified SMPL → GMR → qpos |
| `qpos` | 直读 `[360:396]` |

详见 [`experiments/phi0_hgpt_zmq/README.md`](../experiments/phi0_hgpt_zmq/README.md)。

```bash
CHECKPOINT=/path/to.ckpt \
EPISODE_IDX=447 \
USE_GT=0 \
DEPLOY_MODE=smpl \
CUDA_VISIBLE_DEVICES=4 \
bash scripts/run_pick_tissue_hgpt_zmq_eval.sh
```

---

## 7. 单元测试

```bash
cd Phi_0
PYTHONPATH=src pytest tests/unit/test_pick_tissue_sonic_latent_pipeline.py -q
PYTHONPATH=src pytest tests/unit/test_unified_action_schema.py \
  tests/unit/test_deploy_pipeline.py tests/unit/test_phi0_hgpt_zmq_gt.py -q
```

覆盖：dim mask、WBC→deploy 夹爪、GT replay、ZMQ 协议、lazy GT LUT、ep447 集成。

---

## 8. 常见坑

1. **Checkpoint 路径**：eval 脚本请用**绝对路径**，否则 cwd 可能导致 `Phi_0/Phi_0/...` 找不到文件。
2. **夹爪视觉错误**：多为 WBC 顺序当 deploy 顺序发出；SONIC 路径必须经过 `sonic_zmq_io` / `dex3_gripper`。
3. **`episode_index` vs clip row**：LeRobot `episode_index=447` 对应 dataset row `318688`（shuffle 后）；publisher 内部用 `clip_dataset_index_for_episode()` 映射。
4. **Eval 默认 inset+marker**：与 top panel 无 marker 不同；按 §6.1 显式设 env。
5. **HGPT vs SONIC**：HGPT 看不到三指夹爪；要看夹爪用 SONIC latent eval。
