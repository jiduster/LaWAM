# LaWAM + DexJoCo 指尖关键点辅助损失交接文档

## 1. 目标

验证在 DexJoCo 灵巧手任务上，为 LaWAM 增加指尖关键点预测辅助任务，是否能让 latent representation 学到更有用的手部运动与接触几何先验，从而提升 VLA policy 的动作质量。

当前 LaWAM 主要通过 DINOv3 feature/LAM latent 预测未来视觉状态，并将未来 latent 注入 flow-matching action head。这个目标可能更偏向物体和整体视觉变化，对手指弯曲、接触建立、双手协调等细粒度运动约束不足。

辅助任务的原则是：训练时用未来指尖位置作为监督，推理时不提供未来指尖标签。建议的损失为：

```text
L_total = L_flow + lambda_latent * L_latent + lambda_tip * L_fingertip
```

当前实现已加入 fingertip sidecar loader、训练辅助 head/loss 和日志指标；sidecar 仍然不修改原始 LeRobot parquet schema。

## 2. 当前代码事实

当前 DexJoCo 双臂单任务实验使用：

- 配置：`starVLA/config/training/starvla_train_dexjoco_bimanual_photograph.yaml`
- mixture：`dexjoco_bimanual_photograph`
- robot type：`dexjoco_bimanual`
- action 维度：44
- state 维度：46
- action horizon：4
- 数据帧率：30 Hz，当前 `sec_chunk: 0.15`
- LAM：`latent_action_model/logs/dino_large_vae/lam_release`
- LaWAM 初始化：`results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt`
- 当前启用：`future_prediction: true`、`enable_loss_distill: true`
- 当前启用：`load_pretrained_policy_flow: true`

关键代码位置：

- LaWAM 后端：`starVLA/model/framework/vlas/lawam.py`
- flow expert：`starVLA/model/framework/vlas/flowmatching_expert.py`
- latent/world batch：`starVLA/model/framework/latent_world/batch_builder.py`
- LeRobot 数据读取：`starVLA/dataloader/lerobot_datasets.py`
- DexJoCo modality：`starVLA/dataloader/gr00t_lerobot/data_config.py`
- 训练入口：`starVLA/training/train_starvla.py`

## 3. 先确认是否已有指尖标签

不能假设 LeRobot 数据已经包含 fingertip keypoint。首先检查 `meta/info.json`、`meta/modality.json` 和 parquet schema：

```bash
DATASET=/path/to/dexjoco_lerobot_datasets/bimanual_photograph

python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/path/to/dexjoco_lerobot_datasets/bimanual_photograph")
for name in ["meta/info.json", "meta/modality.json"]:
    path = root / name
    print(f"\n--- {path} ---")
    if path.exists():
        print(json.dumps(json.loads(path.read_text()), indent=2)[:12000])
    else:
        print("missing")

files = sorted((root / "data").glob("*/*.parquet"))
if files:
    print("\n--- parquet columns ---")
    for field in pq.read_schema(files[0]):
        print(field.name, field.type)
PY
```

搜索字段：`fingertip`、`keypoint`、`landmark`、`site`、`tcp`、`ee_pos`、`qpos`、`joint`。

如果只有 action/state 和视频，推荐使用 DexJoCo/MuJoCo 状态做 forward kinematics。不要先用未经验证的 2D 视频 detector 代替 3D 标签，因为会引入相机标定、遮挡和深度误差。

## 4. 指尖标签定义

### 4.1 预测对象

DexJoCo Allegro 每只手有 4 个真实指尖，双臂共 8 个点：

```text
left:  ff, mf, rf, th
right: ff, mf, rf, th
```

推荐 target shape：

```text
[B, H, 8, 3]
```

其中 `H=4` 与 action horizon 保持一致。内部保留 `[8, 3]` 结构，不要一开始就丢失手和指尖维度。

### 4.2 位置还是位移

第一版推荐预测相对当前帧的位移：

```text
delta_tip[t] = fingertip_position[t] - fingertip_position[current]
```

相对位移对 episode 初始摆放更稳定，也更直接对应短时运动。绝对位置可以作为后续 ablation，但前提是坐标系稳定。

### 4.3 坐标系

推荐优先级：

1. robot base/world 坐标系：最容易从 FK 得到；
2. wrist/base 相对坐标系：更有运动不变性；
3. 物体坐标系：需要可靠物体 pose，第一版不建议。

第一版只能选一个主坐标系。所有 episode 必须统一轴定义、左右手定义和单位。位置单位统一为米。

### 4.4 时间对齐

指尖 target 必须和 action 使用同一个绝对 frame index：

```text
tip_target = tip[t + 0 : t + 4]
action_target = action[t + 0 : t + 4]
```

如果改成 `t+1..t+4`，必须在整个 loader、collator 和模型中保持一致。episode 边界处应复用 action chunk 的 padding/过滤规则，并额外提供：

```text
fingertip_valid: [B, H, 8]
```

loss 只在有效标签上计算。

## 5. 标签生成

### 5.1 推荐：MuJoCo FK

对每个 episode 的每一帧：

1. 读取与该帧对应的 robot joint state/qpos；
2. 使用 DexJoCo 完全相同的 XML、joint order 和 model；
3. 设置 `data.qpos` 并执行 forward；
4. 读取固定 fingertip site 或 body 的 `site_xpos/body_xpos`；
5. 按固定的左右手、四指顺序导出 8 个 3D 坐标；
6. 按 dataset frame index 与 parquet 行严格对齐。

如果 XML 没有 fingertip site，应使用明确的末端 body 名称或新增固定 site。不要让不同任务自行猜名称。

### 5.2 保存方式

第一版推荐 sidecar 文件，不立即修改原始 LeRobot schema，例如：

```text
meta/fingertip_positions/episode-000000.parquet
```

每个 sidecar 至少保存：

- `episode_index`、`frame_index`；
- `tip_position`；
- `valid_mask`；
- 坐标系名称和单位；
- fingertip 名称及顺序；
- 生成脚本版本和 XML 标识。

后续如果 sidecar 方案稳定，再考虑写入 parquet 列，例如 `observation.fingertip_position: float32[8, 3]`，并同步更新 LeRobot metadata。

### 5.3 标签检查

导出后检查：

- 坐标是否全为有限值；
- 单位是否为米而不是毫米；
- 每个 episode 帧数是否与 parquet 一致；
- 左右手及四指顺序是否稳定；
- 速度是否有异常尖峰；
- episode 边界是否串帧；
- 可视化时点是否确实位于指尖。

至少抽取 10 个 episode，把 fingertip 投影到 fixed front/wrist 视频或 MuJoCo 渲染图中人工核对。

## 6. 模型接入

### 6.1 保持 action contract 不变

指尖预测是 auxiliary head，不是控制输出。不要把 target 拼进 action，也不要把 DexJoCo action 维度从 44 改成 74。推理时 action 仍然是 44 维。

### 6.2 Prediction head

实现已在 `LatentWorldPolicyBackend` 中增加 MLP head。输入使用预测未来 LAM 表征和当前 state：

- `h_t1_pred`：预测未来 latent；
- 当前 state。

当前实现输出：

```python
tip_pred = fingertip_head(h_t1_pred, state)
# tip_pred: [B, H, 8, 3]
```

这样先验证当前 representation 是否包含指尖运动信息。后续再尝试让 `h_t1_pred` 解码整段未来指尖轨迹。

训练 forward 从 batch 读取：

```python
batch["fingertip_targets"]
batch["fingertip_valid"]
```

并返回 `loss_fingertip` 和 `fingertip_rmse`。在线 policy server 不需要 fingertip label，也不需要修改 action protocol。

## 7. Loss

推荐 masked Smooth L1：

```python
error = F.smooth_l1_loss(tip_pred, tip_target, reduction="none")
# error: [B, H, 8, 3]
mask = fingertip_valid[..., None].to(error.dtype)
loss_tip = (error * mask).sum() / mask.sum().clamp_min(1.0)
```

如果对位置或位移做标准化，训练集统计量要固定，评估 RMSE 时反归一化回米。

第一版所有指尖等权。不要一开始同时加入 thumb/index 加权、接触阶段加权和速度加权，否则难以定位收益来源。

建议 sweep：

```text
lambda_tip = 0.01, 0.05, 0.1
```

至少记录：

```text
train_loss_total
train_loss_flow
train_loss_fingertip
val_loss_total
val_loss_flow
val_loss_fingertip
fingertip_rmse_mm
```

`lambda_tip` 需要依据归一化后的 loss 标度调整，不能直接假设 `0.1` 合适。

## 8. 初始化与冻结

保持现有 SFT 初始化：

```yaml
trainer:
  pretrained_checkpoint: results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt
  load_pretrained_policy_flow: true
```

新增 fingertip head 没有预训练参数，随机初始化；其余模型从 LaWAM pretrain checkpoint 初始化。

第一版建议：

- flow head：训练；
- LAM decoder：沿用现有配置；
- VLM：沿用现有配置；
- DINOv3 encoder：冻结；
- fingertip head：训练。

不建议第一版重新训练 DINOv3 或重新预训练 LAM。这样能把实验变量限制为 fingertip supervision。

如果新 head 在 DDP 下触发 unused parameter 错误，可以暂时打开 `ddp_find_unused_parameters` 调试；稳定后再决定是否恢复为 false。

## 9. 推荐实验顺序

### 阶段 0：数据审计

- 确认是否已有标签；
- 确认 XML、FK 接口和 fingertip site/body；
- 生成 sidecar；
- 检查数量、范围、可视化和时间对齐。

### 阶段 1：只接入 loader

- 不改模型，读取一个 batch 的 target；
- 确认 shape 为 `[B, 4, 8, 3]`；
- 确认 train/val 不串 episode；
- 确认 target 与视频帧对应。

### 阶段 2：加入 head/loss

- 单卡运行 10 到 100 step smoke test；
- 检查 loss 是有限值；
- 检查 head 参数有梯度；
- 检查旧 checkpoint 可以加载；
- 确认新增 head 的 missing key 是预期的随机初始化。

### 阶段 3：A/B 消融

至少比较：

```text
A. 原始 LaWAM SFT，无 fingertip loss
B. 有 fingertip head，但 lambda_tip=0
C. fingertip loss，lambda_tip=0.01/0.05/0.1
```

实验必须保持相同数据 split、初始化 checkpoint、flow 初始化、global batch size、训练 steps、随机种子和评测 seed。

### 阶段 4：在线评测

先用 2 个 episode 做 smoke test，确认：

- action 不是全零；
- 归一化和范围正确；
- 双臂 action 仍为 44 维；
- server 不需要 fingertip label；
- 没有抖动、提前闭合等明显管线问题。

之后再用正式 episode 数量比较成功率。少量 episode 只能发现管线错误，不能作为最终结论。

## 10. 必做诊断与后续方向

除了总成功率，还应记录：

- arm action error；
- hand joint/action error；
- left-right coordination error；
- fingertip position RMSE；
- 接触前后的行为和动作抖动。

如果 fingertip RMSE 下降但成功率不变，说明辅助监督没有有效传递到 flow action head。下一步可尝试：

- 增强 `h_t1_pred` 到 flow head 的 cross-attention；
- 对 hand action group 增加单独 loss weight；
- 将 wrist-relative fingertip feature 作为 action head conditioning；
- 增加接触状态、指尖速度或指尖间相对距离任务。

## 11. 风险清单

### 未来信息泄漏

未来 fingertip、未来 state 或由未来 state 得到的特征只能用于 loss，不能输入 policy。

### 坐标系不一致

front camera、wrist camera、world frame 和 robot base frame 不能混用。必须在 sidecar metadata 中写明坐标系。

### 时间错位

视频、state、action、fingertip 必须按同一绝对 frame index 对齐，重点检查 LeRobot episode 最后几帧的 chunk padding。

### 标签质量问题

错误的 site、joint order 或单位会带来比没有辅助任务更差的监督。标签未通过可视化审计前不要启动大规模训练。

### 辅助 loss 过强

如果 `lambda_tip` 太大，模型可能牺牲 flow action loss 去拟合指尖。必须同时看 flow loss、tip RMSE 和在线成功率。

## 12. 后续 agent 的最小任务清单

1. 检查 `bimanual_photograph` 是否已有 fingertip/keypoint 字段。
2. 如果没有，定位 DexJoCo FK 接口和 fingertip site/body 名称。
3. 写独立导出脚本，生成并可视化 sidecar 标签。
4. 为 dataset wrapper 增加 fingertip target 和 valid mask。
5. 在 `lawam.py` 增加 fingertip auxiliary head。
6. 在训练 forward 中加入 masked loss 和日志。
7. 运行单卡 smoke test，确认 checkpoint 加载和梯度。
8. 进行 lambda_tip 对照实验。
9. 用相同 seed 和 episode 数量做 DexJoCo rollout 对比。
10. 只有确认有稳定收益后，再扩展接触和相对几何任务。

## 13. 结论

这个方案的核心不是让 LaWAM 在推理时获得指尖真值，而是把指尖运动作为训练阶段的结构化监督，迫使 latent world representation 保留灵巧手局部动力学信息。

第一版应保持简单：固定坐标系、固定 8 个指尖、固定 horizon、sidecar 标签、一个 masked Smooth L1 loss、一个小 prediction head 和严格 A/B 对照。先验证指尖监督是否提升 DexJoCo 成功率，再考虑接触、速度、相对距离或 hand-specific action weighting。
