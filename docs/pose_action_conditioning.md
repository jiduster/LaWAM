# 当前物体姿态 → Action DiT：实现与运行记录

2026-10-22。已实现交接文档的橙色分支，默认关闭。pose 不进入 VLM、latent head、LaWM 或 endpoint head。本文的短程训练使用生产 flow／DiT 和小型替代 VLM／LAM；**尚未完成原始 7 GB checkpoint 的微调或策略成功率评估**，当前会话 `torch.cuda.is_available() == False`，`nvidia-smi` 无法连接驱动。

## 接入位置与数据

`object_pose_condition` 单条为 float32 `[12]`，batch 为 `[B,12]`。前三维位置用训练子集 mean/std 标准化；旋转六维和目标方向三维保持原范围。三个独立 MLP（中间宽度 128）加类型嵌入，生成 `[B,3,1024]` token。每个 DiT block 的原 attention／FFN 残差之后，增加独立的 `PoseCrossAttention`；当前 16 个 block 全部插入，16 heads。Q 是 action expert 当前 hidden（含原 state／可选 future 前缀），K/V 是三个 pose token。h30、use_state=true、无 future query token 时 hidden 为 `[B,31,1024]`，原 action decoder 仍只输出最后 30 个动作 token、每步 44 维。

attention 输出投影 weight/bias 零初始化；新增模块构造保留原参数的 RNG 序列。CFG conditional/unconditional 使用同一份当前 pose token，未来条件的原 dropout／guidance 规则不变。字段缺失、维度不符、非有限值、目标方向非单位长度会明确报错。pose encoder 的 mean/std 是持久 checkpoint buffers，启用配置与超参数同时保存在训练运行目录的 `config.yaml`；推理加载会检查 pose 参数／配置是否匹配。

已发现并接入设计侧后来补齐的数据：

```text
/data/home/zyh/dexjoco/outputs/photograph_object_pose_20261022/lerobot/bimanual_photograph
```

来源审计见该输出目录上两级的 `report.md`。这是 `mujoco_raw_restore_reconstruction`，并非原始录制时独立测量的物体真值。100 条候选轨迹中 87 条通过审计；13 条隔离。baseline/orange 都在 chunk 采样前按 `pose_train_episode_indices.json` 过滤，保留原 episode/frame ID，然后按原 loader 的 tail split 划分：86 条训练、28,664 帧，episode 99 验证、328 帧。不会使用数据副本中包含全部候选行的 pose 统计。

实际训练位置 mean = `[-0.1354795775, 0.0023239882, 1.4801980067]`，std = `[0.1177022336, 0.1093336337, 0.1152409072]` 米。配置生成器重算并保存训练／验证 episode 列表。loader 读取**当前 base_index** 的独立 pose 列，不读取 chunk 终点，也不将 pose 拼入 state；遇到 `object_pose_valid=False` 会拒绝。

## 文件与职责

| 文件 | 改动 |
|---|---|
| `starVLA/object_pose.py` | 固定 back 光学坐标变换、同帧 recorder、独立 sidecar store |
| `starVLA/model/framework/vlas/pose_conditioning.py` | 三组编码器、校验、独立 pose residual attention |
| `starVLA/model/framework/vlas/{flowmatching_expert,cross_attention_dit}.py` | 开关／超参数、两种 DiT 的训练与 CFG 采样接入 |
| `starVLA/model/framework/vlas/lawam.py` | 重复 diffusion batch 传递、action expert 专用冻结／loss 模式 |
| `starVLA/model/framework/lawam_framework.py` | 通用 freeze 策略执行后再次落实 action-only 冻结 |
| `starVLA/model/framework/base_framework.py` | 推理 checkpoint pose key 一致性检查 |
| `starVLA/model/framework/latent_world/{types,batch_builder}.py` | example／batch schema 与 processor 后的独立 float32 pose 字段 |
| `starVLA/model/framework/latent_world/runtime/{runner,offline_diagnostics}.py` | 验证／离线采样转发 pose |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | 采样前 episode 过滤、当前帧列读取与有效性检查 |
| `starVLA/dataloader/latent_world_train_collator.py` | raw sample／sidecar → batch 字段、action-only 时跳过 endpoint 标签要求 |
| `scripts/configure_pose_action.py`, `scripts/train_pose_action.sh` | 同源初始化的配对配置与训练入口 |
| `starVLA/config/training/starvla_train_dexjoco_photograph_pose_action_{baseline,orange}.yaml` | 已用实际数据生成的独立 10k 配置 |
| `scripts/{serve,eval}_lawam_pose_action.py` | 当前 websocket 链路的适配器，复用原 infer／eval 实现 |
| `scripts/record_photograph_object_pose_from_qpos.py`, `scripts/prepare_photograph_object_pose.py` | 其他数据源的完整状态回放、state／timestamp／frame 对齐校验及 sidecar 发布 |
| `scripts/test_pose_action.py`, `scripts/test_pose_action_server.py`, `scripts/test_photograph_object_pose_mujoco.py`, `scripts/validate_real_pose_dataset.py` | 模型、服务、真实 XML、真实数据验证 |
| `scripts/test_external_latent_action_head.py` | tiny test fixture 增加 pose／action-only 参数，原测试保留 |

`action_expert_only=true` 冻结所有非 flow 参数（包括 act／flow query、VLM、latent head、DINO／LaWM、endpoint head），并让冻结模块保持 eval；只有原 action expert 和新增 pose 参数可训练。endpoint head 仍按基配置构造和加载；原 `fingertip_loss_weight` 保留只是为保持 head 结构，实际总目标严格为 `loss_flow`，不要求 endpoint 标签。原 state/action 统计从基 checkpoint 继承。

## 实际验证结果

日志在 `results/pose_action_validation/`。

- 24 项 unittest 全部通过，包含既有 contact／external latent 回归。两种 DiT 在 CFG=1、1.5 下，零初始化 orange 与 baseline 的动作和训练 loss 均逐值相等，`atol=rtol=0`。
- 另临时加载 git HEAD 的原始 flow／DiT 源码，与新代码关闭分支比较：两种 DiT 的参数、训练 loss、CFG=1／1.5 采样均逐值相等，`atol=rtol=0`。这是 action expert 级验证，没有声称完整原始 checkpoint 在 GPU 上做过对比。
- 5 步 tiny backend 训练，两种 DiT 首步输出投影梯度绝对和为约 `0.0879945`／`0.107325`，第二步 encoder 获得非零梯度。训练后只改变输入位置 0.2 米，固定采样种子的最大动作差为 `0.000116900`／`0.000126958`。零输出初始化下，第一步内部层梯度为零是预期行为。
- 冻结模块参数与 `h_t`、`h_t1_pred`、`h_vlm`、latent 在更新前后逐值相等；endpoint 参数不变。训练 batch 重复两次、CFG 两支保留同 pose、checkpoint 严格权重往返、baseline→orange 微调加载和 serving 的 config/权重入口均已验证。完整 backbone 在这些模型测试中使用替代小模块。
- 真实数据 loader/collator 解码了当前图像和动作，验证两组同 seed 抽样 `(episode 9, frame 262)`、`(episode 24, frame 109)` 的图像／state／action 逐值一致；所有 87 条有效轨迹首末 pose 可读，隔离 episode 15 被拒绝。真实样本在 tiny CPU backend 上完成 3 步训练，固定 flow 随机条件时 loss 为 `1.538788 → 1.529149 → 1.513765`。
- 真实 photograph XML 的位移、旋转、光学轴、观测快照拷贝通过；11 个朝向下，`acos(goal_dir_O.z)` 与环境 `_compute_success_metrics` 的最大角度差约 `1.65e-6°`。
- 服务适配器经实际 msgpack codec 验证：pose 正确进入 example，机器人归一化正常，动作 `[30,44]` 和原 endpoint `[8,3]` 响应保留。服务与评估 CLI `--help` 可运行。未实际启动完整策略 websocket rollout，未报告成功率提升。

复现：

```bash
cd /data/home/zyh/LaWAM
export PYTHONPATH=/data/home/zyh/dexjoco/dexjoco:$PWD
export MPLCONFIGDIR=/tmp/lawam_pose_mpl
export HF_DATASETS_CACHE=/tmp/lawam_pose_hf

/data/home/zyh/miniconda3/envs/lawam/bin/python -m unittest \
  scripts.test_pose_action scripts.test_oracle_contact scripts.test_external_latent_action_head -v
/data/home/zyh/miniconda3/envs/lawam/bin/python scripts/test_pose_action_server.py
/data/home/zyh/miniconda3/envs/lawam/bin/python scripts/validate_real_pose_dataset.py
/data/home/zyh/miniconda3/envs/dexjoco/bin/python scripts/test_photograph_object_pose_mujoco.py
```

## 训练命令

基模型保持为交接中指定的 `.../dexjoco_bimanual_photograph_C_h30_tip_endpoint/20261015_142505+.../final_model/pytorch_model.pt`。新输出分到 `results/Checkpoints/pose_action_baseline` 与 `pose_action_orange`，均从同一基模型初始化，只优化 flow，默认 10k steps、seed 2026、lr 1e-5。

```bash
conda activate lawam
cd /data/home/zyh/LaWAM

# 两组完整训练，分别执行；NUM_PROCESSES 根据可用 GPU 设置。
NUM_PROCESSES=1 POSE_SEED=2026 POSE_STEPS=10000 bash scripts/train_pose_action.sh baseline
NUM_PROCESSES=1 POSE_SEED=2026 POSE_STEPS=10000 bash scripts/train_pose_action.sh orange

# 有 GPU 后先用完整基模型短程确认（本会话未执行）。
NUM_PROCESSES=1 POSE_STEPS=10 bash scripts/train_pose_action.sh baseline \
  --datasets.vla_data.per_device_batch_size 2
NUM_PROCESSES=1 POSE_STEPS=10 bash scripts/train_pose_action.sh orange \
  --datasets.vla_data.per_device_batch_size 2
```

脚本每次生成独立配置，重新校验所选训练集并计算统计。`POSE_DATA_ROOT` 可指定另一个 `bimanual_photograph` 目录；若不用带 pose 列的数据副本，需提供由本仓库工具校验发布的 `POSE_SIDECAR_ROOT`。baseline/orange 必须使用同一数据及有效 episode 集，不能只给 orange 过滤轨迹。更改 split 时重新运行配置生成器，不能沿用旧 pose 统计。

也可直接使用已生成配置：

```bash
NUM_PROCESSES=1 bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_photograph_pose_action_baseline.yaml
NUM_PROCESSES=1 bash train_lawam.sh \
  --config_yaml starVLA/config/training/starvla_train_dexjoco_photograph_pose_action_orange.yaml
```

## 服务、评估及直接调用

无需修改外部 DexJoCo checkout，必须使用本仓库新适配器；原服务会丢弃额外 observation 字段。新服务对 baseline/orange 都可用，baseline 忽略额外 pose。客户端在 `_process_obs` 所在的仿真线程拷贝当前 site／geom／camera 数据，通过原观测队列和 websocket 传递；服务继承原 `infer`，仅包装 `model.predict_action` 补充独立字段，因此原 endpoint 解码、动作返回和视频／成功记录保持原实现。

```bash
# lawam 环境；分别将 CHECKPOINT 指向两组“已训练”输出。
export PYTHONPATH=/data/home/zyh/dexjoco/dexjoco:/data/home/zyh/LaWAM
CHECKPOINT=/absolute/path/to/trained_run/final_model/pytorch_model.pt
python scripts/serve_lawam_pose_action.py \
  --checkpoint "$CHECKPOINT" --lawam-root /data/home/zyh/LaWAM \
  --host 127.0.0.1 --port 8000 --device cuda --robot-type dual_arm \
  --single-arm-stats-key dexjoco_bimanual --dual-arm-stats-key dexjoco_bimanual \
  --action-hz 30 --seed 2026

# dexjoco 环境，另一个终端；两组配对使用相同 seed，输出目录分开。
cd /data/home/zyh/LaWAM
export PYTHONPATH=/data/home/zyh/dexjoco/dexjoco:$PWD
for seed in 9 10 11 12 13 14; do
  python scripts/eval_lawam_pose_action.py \
    --config /data/home/zyh/dexjoco/configs/multi_task/bimanual_photograph.yaml \
    --host 127.0.0.1 --port 8000 --seed "$seed" --episodes 1 \
    --action-horizon 30 --wait-for-actions \
    --output "results/pose_action_eval/orange/seed${seed}"
done
# baseline：重启服务指向 matched baseline checkpoint，将输出中的 orange 改为 baseline。
# 正式成功率：两组使用同一个 seed 和 --episodes 50 或更多，再用其他配对 seed 复测。
```

评估器第一版限制 `bimanual_photograph`、`rand_full=false`、`randomize_dynamics=false`。`--wait-for-actions` 仍按原逻辑执行整个 chunk，不改变 horizon 或重规划节奏。6 个诊断 seed 不足以证明成功率收益。正式结果需保留总体成功率，并结合原视频／物理诊断分开统计持续抓持时的朝向变化、掉落率和掉落后重新抓取率；本次没有生成这些策略指标，也没有新增未经验证的抓持判定器。

LaWAM 直接调用（图像、机器人 state 沿用原流程）：

```python
from starVLA.object_pose import extract_photograph_object_pose

# 与 RGB、raw_robot_state 同一帧，在仿真线程中执行。
pose = extract_photograph_object_pose(raw_env._model, raw_env._data)  # float32[12]
example = {
    "primary_image": [ego_rgb],
    "wrist_image": [left_wrist_rgb, right_wrist_rgb],
    "lang": prompt,
    "embodiment_id": 28,
    "action_hz": 30.0,
    "state": normalized_robot_state,  # 原 state 统计，仅机器人 46 维
    "object_pose_condition": pose,
}
result = model.predict_action([example], guidance_scale=1.0, num_inference_steps=10)
actions_normalized = result["normalized_actions"]
```

仍需完成的实验是：在可用 GPU 上加载完整基模型做短程微调，再完成配对 10k 训练与 rollout。真实数据已经可供使用，但标签性质仍是通过审计的仿真重建条件；结果应按这一性质解释。本次没有修改原 checkpoint、原数据或外部 DexJoCo 仓库，也保留了原有未提交的 `scripts/visualize_fingertip_subgoal.py`。
