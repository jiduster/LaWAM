# Sharpa LaWAM H30 训练

2026-10-29 校准。配置以之前实际完成的原始 LAM 对照实验为准：

`results/Checkpoints/dexjoco_bimanual_photograph_A_h30/20261015_081821+dexjoco_bimanual_photograph_A_h30_sft/config.yaml`

同时核对了 C_h30 photograph 和 hanoi 保存配置。不能用目录中仍保留的早期 H4 YAML 替代这些实际运行记录，也不能照搬 contact/pose 续训实验的学习率和初始化 checkpoint。

## 参数对照

| 参数 | 之前的 DexJoCo H30 | 当前 Sharpa H30 |
| --- | --- | --- |
| action_horizon / future / past | 30 / 29 / 0 | 30 / 29 / 0 |
| 数据频率 | 30 Hz | 20 Hz，保留真实频率 |
| sec_chunk / flow.horizon_sec | 1.0 / 1.0 秒 | 1.5 / 1.5 秒 |
| 视觉端点偏移 | [0,29] | [0,29] |
| 非边界样本的视觉端点时间差 | 29/30 ≈ 0.967 秒 | 29/20 = 1.45 秒 |
| 主视角帧数 / 分辨率 | 2 / 256 | 2 / 256 |
| 单卡 batch / 梯度累积 | 4 / 1 | 4 / 1 |
| 卡数 / 全局 batch（下方命令） | 8 / 32 | 8 / 32 |
| steps / warmup / save | 40000 / 1000 / 5000 | 40000 / 1000 / 5000 |
| eval_interval / eval_batches | 500 / 20 | 500 / 20 |
| base / flow / decoder / VLM 学习率 | 5e-5 / 1e-4 / 5e-5 / 2e-5 | 相同 |
| perceptual / distill 权重 | 0.1 / 0.1 | 0.1 / 0.1 |
| seed / val_tail_ratio | 2026 / 0.01 | 2026 / 0.01 |
| 冻结策略 | LAM encoder 冻结、decoder 训练；VLM 保留前 16 层 | 相同 |
| VLM / LAM / 策略初始化 | 本地 Qwen / release LAM / lawam_pretrain | 相同 |
| load_pretrained_policy_lam | false | false，防止策略 checkpoint 覆盖单独指定的 LAM |
| use_pretrained_dataset_statistics | false | false，使用 Sharpa 自身统计 |
| action_dim / state_dim | 44 / 46 | 单臂 29 / 29，双臂 58 / 58 |

H4 并非框架不支持的值，但不符合本次沿用既有 H30 训练配方的要求。原 Sharpa 配置的 H4、0.2 秒、10k steps、2k save 已修正。数据字段 `action_horizon` 也显式设置为 30，梯度累积和 DDP 设置显式保留为 1 / false。

**H30 与 LAM 预训练的 1.6 秒是两个不同概念。**此次按既有 H30 配方修正，没有额外增加独立视觉采样参数。若要求 20 Hz 下视觉严格相隔 1.6 秒，需要 `[0,32]`；在当前首尾绑定采样下对应 33 个连续动作，或后续实现视觉端点独立采样。不能将采样频率伪装成 30 Hz 来迁就旧配置。

## 启动

在本机从仓库根目录运行，使用 8 卡、全局 batch 32。配置里保留 wandb，与原实验一致；使用者需有可用的 wandb 配置，也可通过 `WANDB_MODE=offline` 离线记录。

```bash
conda activate lawam
cd /data/home/zyh/LaWAM

HF_HOME=/tmp/lawam_hf_cache \
HF_HUB_CACHE=/tmp/lawam_hf_cache/hub \
HF_DATASETS_CACHE=/tmp/lawam_hf_cache/datasets \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_sharpa_door_open.yaml
```

双臂 package opening 使用同一条命令，将配置换成 `starVLA/config/training/starvla_train_sharpa_bimanual.yaml`。两个 run ID 均带 `h30`，按时间戳创建新输出目录。

## 验证

```bash
PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 \
MPLCONFIGDIR=/tmp/lawam_matplotlib \
HF_HOME=/tmp/sharpa_audit_hf HF_DATASETS_CACHE=/tmp/sharpa_audit_hf/datasets \
HF_HUB_OFFLINE=1 \
python scripts/validate_sharpa_training.py \
  --config starVLA/config/training/starvla_train_sharpa_door_open.yaml \
  --report docs/validation/sharpa_h30_door_open.json
```

双臂配置同样验证，报告为 `docs/validation/sharpa_h30_bimanual.json`。脚本使用真实 train/val 数据、生产配置构建器、真实本地 Qwen processor 和生产 collator。检查 99/1 的 episode 划分、动作 `[B,30,29]` 或 `[B,30,58]`、主视角端点 `[0,29]`、腕部当前图、语言、状态维度、归一化动作未被 collator 重采样/截断，以及轨迹末端不会跨到下一条。还将真实动作/状态送进缩小的生产 flow head 做一次 CPU forward/backward，检查时间网格与动作 mask 一致。

末端沿用既有 loader 行为：不足 H30 时重复该 episode 最后一个绝对关节目标和最后一幅主视角图；因此末端样本实际视觉间隔会短于 1.45 秒。此次验证没有把它误报为固定间隔样本。

这是配置和数据链路验证，缩小 flow 的视觉/VLM 特征是合成张量。没有运行完整 VLM/LAM 的 GPU 训练，也没有执行仿真成功率评估；本次不据此判断 Stage 1 适配的收益。

## 仿真评测

开门训练产物：

`results/Checkpoints/sharpa_cabinet_pour_door_open/20261029_191743+sharpa_cabinet_pour_door_open_h30_sft/final_model/pytorch_model.pt`

推理服务跑在 `lawam` 环境。评测客户端要能 import DexArena 和 ManiSkill，这台机器上的 `lawam` 环境没有 ManiSkill，README 里的 `ms3` 环境也还没建。两个进程不要塞进同一个环境。

服务端按 checkpoint 里的 `min/max` 做 state 归一化和动作反归一化，并要求请求带上训练时的腕部视角：单臂 1 路，双臂先左后右。评测端直接使用 ManiSkill 环境本身，按 controller 顺序读关节，再把 `base_camera` 和腕部相机发给服务。成功与否读环境 `info["success"]`。开门任务的上限是 600 步，动作块是 30 步（20 Hz 下 1.5 秒）。

```bash
# terminal 1
conda activate lawam
cd /data/home/zyh/LaWAM
export PYTHONPATH=/data/home/zyh/LaWAM
export HF_HOME=/tmp/lawam_hf_cache
CUDA_VISIBLE_DEVICES=0 python examples/DexArena/eval_files/lawam_sharpa_server.py \
  --checkpoint results/Checkpoints/sharpa_cabinet_pour_door_open/20261029_191743+sharpa_cabinet_pour_door_open_h30_sft/final_model/pytorch_model.pt \
  --unnorm-key sharpa_left \
  --port 10093

# terminal 2, in the DexArena / ManiSkill environment
cd /data/home/zyh/LaWAM
export PYTHONPATH=/data/home/zyh/LaWAM
python examples/DexArena/eval_files/eval_lawam_sharpa.py \
  --env-id CabinetPourDoorOpen-v1 \
  --task-module task/cabinet_pour_door_open.py \
  --dexarena-root /data/home/zyh/DexArena \
  --unnorm-key sharpa_left \
  --camera-profile robodex_fixed \
  --port 10093 \
  --episodes 1 \
  --steps 60 \
  --video-dir /tmp/sharpa_door_eval \
  --output /tmp/sharpa_door_eval/summary.json
```

先用上面的 1 条、60 步确认环境和协议能对上，再去掉 `--steps 60`、把 `--episodes` 调到要统计的次数。正式统计沿用任务自己的 600 步上限。
