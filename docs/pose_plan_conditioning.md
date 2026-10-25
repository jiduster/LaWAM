# 绿色分支：object pose → latent planning → future subgoal

2026-10-23。按原设计，在 latent head 生成 latent 之前，用当前 object pose 修正规划特征。默认关闭；绿色训练开启 `plan_pose_conditioning`，关闭橙色 `flow_cfg.pose_conditioning`。

## 模型和梯度

```text
current pose[12] → 3 pose tokens[128] ──────────────┐
VLM hidden → h_act 副本[8,2048] → pose adapter ←─┘
                              → 原 latent head → z[1,32]
current DINO h_t ──────────────────────────────┐
z ────────────────────────────────────────────┴→ LaWM decoder
                                                  → future DINO
                                                    ├→ endpoint head → 8-tip subgoal
                                                    └→ Action DiT → 30×44 actions
```

`plan_pose_encoder` 将 position / rotation / goal-direction 分别编码为 3 个 128 维 token，位置只按训练子集统计归一化。`vlm_to_lam.pose_adapter` 以 `h_act` 为 Q、pose 为 K/V 做残差 cross-attention，注意力宽度 128、8 heads；最后的 128→2048 输出投影 weight/bias 零初始化。它不原地修改共享 VLM hidden，也不往 VLM 序列插入 pose。生产配置新增 **648,704** 个可训练参数，另有 6 个统计 buffer 元素。

`external_qformer` 模式同样支持：只修正该 head 的私有 latent query，在 final norm 之前融合，不修改共享 VLM context。绿色和橙色可独立开关。绿色不改变 flow 的 CFG 规则：两支之前生成同一预测未来特征，再沿用原未来条件替换。

完整 Stage 2 保留 `detach_future_feature=true`。flow loss 不通过未来特征反传到绿色模块；绿色主要由原 latent teacher、未来 DINO MSE 和 endpoint loss 学习，三项权重均为 0.1。修正后的 latent 仍受原 LAM teacher 约束，保留原 latent 空间。`action_expert_only=true` 与绿色分支组合会报错。

## 训练

`scripts/train_pose_plan.sh` 先生成完整配置，再通过原 `train_lawam.sh` 启动。不要直接用模板 YAML 训练，数据路径和 pose 统计需要由生成器填写。

| 项目 | 设置 |
|---|---|
| 初始化 | LaWAM pretrain + photograph-excluded Stage 1 LAM；新绿色模块初始化 |
| 模式 | 完整 Stage 2；绿色 true、橙色 false |
| horizon / future / past | 30 / 29 / 0 |
| 数据/模型频率字段 / horizon_sec / sec_chunk | 30 / 1 / 1 |
| action / state / pose | 44 / 46 / 12 维 |
| 视觉帧偏移 / endpoint | [0,29] / t+29 |
| 数据 | 审计有效 87 条；86 train、1 val，统计只用 train |
| GPU / 每卡 batch / 累积 | 8 / 4 / 1，全局 batch 32 |
| 优化步 / warmup | 40000 / 1000 |
| 学习率 | flow 1e-4、VLM 2e-5、LaWM decoder 5e-5、其他（含绿色模块）5e-5 |
| 损失 | flow + 0.1 perceptual + 0.1 latent distill + 0.1 endpoint |
| 验证 / 保存 | 每 500 / 5000 步 |
| 输出 | 仓库下 results/pose_stage2_green_local，避开旧 Checkpoints 软链接 |

原 Stage 2 的 embedding/末层冻结规则及 DINO/LAM encoder 冻结规则保留。首次初始化 `load_pretrained_policy_lam=false`，保留 photograph-excluded Stage 1 LAM；从中途 checkpoint 续训时应恢复已训练的 LAM。

本机可直接启动：

```bash
conda activate lawam
cd /data/home/zyh/LaWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
POSE_DATA_ROOT=/data/home/zyh/dexjoco/outputs/photograph_object_pose_20261022/lerobot/bimanual_photograph \
FINGERTIP_ROOT=/mnt/ceph2/DexJoCo-Datasets-LeRobot/dexjoco_fingertip_sidecars \
bash scripts/train_pose_plan.sh
```

末尾加 `--dry-run` 可仅生成配置、检查迁移资源。生成 YAML 位于 `results/pose_plan_configs/`；日志在每次 run 的 `logs/train.log`。启动应显示 8 processes、batch 4、accumulation 1、global batch 32。

可用 `LAWAM_ASSET_ROOT` 改模型资源根目录（默认仓库根目录），`POSE_OUTPUT_ROOT` 改输出位置，`POSE_STEPS` / `POSE_SEED` 改步数/seed。生成器会检查剩余空间；40k 按约 76 GB 模型文件预留，实际还需缓存和日志空间。完整启动 YAML 会由原脚本复制到 run 的 logs。

## 迁移机器

迁移完整代码和训练环境。当前验证环境为 Python 3.10、PyTorch 2.7.0+cu128；没有验证其他依赖组合。可从现有 lawam 环境导出重建或用 conda-pack 迁移，GPU 驱动须支持对应 CUDA runtime。

模型资源根目录（`LAWAM_ASSET_ROOT`）下须有：

```text
results/Checkpoints/qwen3_weights/                         # 完整模型、processor/tokenizer
results/Checkpoints/dinov3-vitb16-pretrain-lvd1689m/        # 完整 DINO 权重和配置
results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt
latent_action_model/logs/dino_large_vae_dexjoco_bimanual_exclude_photograph_fingertip_stage1/pytorch_model.pt
latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml
```

本机 `results/Checkpoints` 指向外部 Ceph，复制这个软链接本身不等于复制模型。对上述选定资源使用如 `rsync -aL` 的方式解引用复制。生成器会把来源 LAM YAML 中写死的 DINO 路径改成新资源位置，另存 `.lam.yaml`，不改源配置。

数据需要完整的扩展 `bimanual_photograph/`（data、meta、videos、有效 episode 清单和 provenance）。视频也是软链接，复制时须解引用。`FINGERTIP_ROOT` 下要有 `bimanual_photograph/manifest.json` 和 `episodes/`，可只迁移该任务。

新机器将启动命令的 `cd`、`POSE_DATA_ROOT`、`FINGERTIP_ROOT` 改成新路径；模型资源就在项目下时无需设置 `LAWAM_ASSET_ROOT`。生成器会检查视频断链、87 条轨迹清单、pose 列、endpoint 文件、资源路径及输出空间，并重新计算训练集统计。训练不依赖外部 DexJoCo simulator 仓库；评测仍需 DexJoCo 的 pose 采集/转发接入。

训练完成后，迁移 run 应同时保留 `config.yaml`、`dataset_statistics.json` 和目标 checkpoint。模型构造仍依赖 Qwen/DINO/Stage 1 LAM 资源；换路径部署时更新 `base_vlm`、`lam_ckpt_path`、`lam_yaml_path` 及 LAM YAML 中的 DINO 路径，保留绿色开关和位置统计。推理入口会拒绝绿色参数与配置不匹配的 checkpoint。

## 验证

已通过 CPU 行为测试：两类 latent head、两类 DiT 的零初始化等价；pose 对 latent→未来特征→endpoint/actions 的作用；三项规划监督的梯度；训练/推理一致；绿色/橙色组合；checkpoint 保存恢复及缺失参数拒绝；迁移路径重写和断链检测。已有橙色/latent head 回归测试通过。

真实数据使用生产 loader/collator，train/val 的 actions=[2,30,44]、state=[2,46]、pose=[2,12]、endpoint=[2,8,3]，确认同帧 pose 和 t+29 endpoint 对齐。小型替代 VLM/LAM/flow 运行 3 次联合反向传播，总 loss 1.80848→1.78969→1.77536，绿色编码器及 adapter 有梯度。这是链路验证，未运行完整 7 GB 模型 GPU 训练或成功率评估。pose 仍为审计通过的动作重放重建标签，没有独立逐帧物体真值。

复核：

```bash
PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 MPLCONFIGDIR=/tmp/lawam_pose_mpl \
  python -m unittest scripts.test_pose_plan scripts.test_pose_action scripts.test_external_latent_action_head -v

# 先用 train_pose_plan.sh --dry-run 生成实际配置：
PYTHONPATH=. NO_ALBUMENTATIONS_UPDATE=1 MPLCONFIGDIR=/tmp/lawam_pose_mpl \
  python scripts/validate_real_pose_plan_dataset.py --config /path/to/generated_green.yaml
```

本机真实数据结果在 `results/pose_action_validation/green_real_data.json`。当前环境无法使用 CUDA，未启动完整训练。
