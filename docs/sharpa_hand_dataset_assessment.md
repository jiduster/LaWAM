# Sharpa Hand 数据用于 LaWAM 的兼容性评估

检查日期：2026-10-27。数据根目录：`/mnt/ceph2/DexArena/data/sharpa_hand`。

结论：数据可以用于 LaWAM 的策略微调以及 LAM 的领域适配，但当前仓库不能只修改数据路径就直接训练。无需重新采集或重渲染整批视频；主要工作是补充机器人/相机/语言映射，整理本体状态，并设置匹配的动作与时间窗口。此次未修改训练实现、未启动训练、未写入源数据目录。

## 实际检查结果

- 30 个任务，每任务 100 条轨迹，共 3,000 条、658,212 个 transition，20 Hz，约 9.14 小时。
- 9 个左手单臂任务、20 个右手单臂任务、1 个双臂任务。单臂动作 29 维，双臂动作 58 维。
- 61 个 RGB 视频，均为 AV1、256×256、20 Hz；主视角 `observation.images.base_camera`，另有对应左右腕相机。
- 逐任务扫描了全部 Parquet 的当前状态、下一状态、动作、索引、时间戳和成功标记。状态/动作未发现 NaN/Inf；索引、episode 长度和行范围一致；同一 episode 内 `next.observation.state[t]` 与 `observation.state[t+1]` 完全一致；时间戳最大误差约 1.53 微秒。
- 3,000 条轨迹均出现 `next.success=True`，且结束帧均成功。这是数据标签检查，不代表此次重新执行仿真验证成功。
- 逐视频用 ffprobe 检查，61 个视频帧数均与对应数据行数相同。通过仓库实际 `get_frames_by_timestamps` 函数，在每个视频的第 0、50、99 条轨迹分别抽取起始、未来和末尾帧，共 183 组 / 549 帧，均成功解码。
- 30 份 `recording_integrity.json` 均标记 valid，包含生产端 roundtrip / 视频对齐检查记录。这些记录作为附加证据；此次没有重新校验所有 SHA256，也没有逐帧完整解码视频。
- `video_manifest.json` 中每任务帧数比现存训练数据多 100，属于元数据口径差异；现存 MP4 帧数与 episode offset / Parquet 匹配，不能用该 manifest 的帧数重建索引。

## 不能直接开训的具体原因

### 1. 相机自动推断会报错

所有任务均无 `meta/modality.json`。现有 [fallback](../starVLA/dataloader/gr00t_lerobot/datasets.py#L100) 只要腕相机名含 left/right 就判断为双腕，导致只有一个 `left_wrist_camera` 或 `right_wrist_camera` 的 29 个单臂任务进入需要两个不同腕相机的分支。

已分别用真实左臂、右臂数据调用该 helper，均复现 `ValueError: Could not infer distinct left/right wrist cameras`；双臂 package 数据可以通过此推断。

适配方式：提供明确的 modality 映射，单臂映射成 `video.primary_view` + `video.wrist_view`，双臂映射成 `video.primary_view` + `video.wrist_left` + `video.wrist_right`；或者修正 fallback 的判别条件。

### 2. 原始 observation.state 不是统一的关节状态

状态维数随任务变化，为 74–390 维，名字只有 `obs_0` 等通用占位名。回放元数据表明采集时为 `obs_mode=state`。本机 DexArena 任务源码也确认该模式包含任务额外状态，例如门位姿、柜体位置、阶段标记、键盘目标/按键进度等；不能将完整向量当成纯本体状态。

建议从环境状态或 replay 中提取控制器顺序的关节 qpos，单臂为 29 维，双臂为 58 维，并核对控制器顺序。本机已有参考实现 `/data/home/zyh/DexArena/utils/image_observation.py::extract_controller_qpos`。不要仅凭位置截取 `observation.state[:29]`。

如果目标是 RGB + 本体状态的 VLA，应明确排除评测时不提供的物体真值/阶段信息；保留确实需要的目标条件时，应定义为明确的输入并在评测端同样提供。

现有 [维度对齐函数](../starVLA/model/framework/latent_world/batch_utils.py#L15) 会直接截断超长向量。例如沿用 state_dim=46 会把 74–390 维状态静默截断，不能以训练能跑作为映射正确的证据。

### 3. 需要注册 Sharpa 数据与 embodiment

`ROBOT_TYPE_CONFIG_MAP`、`ROBOT_TYPE_TO_EMBODIMENT_TAG` 和 `DATASET_NAMED_MIXTURES` 目前没有 Sharpa 配置。建议定义单臂左/右及双臂的配置与明确的 embodiment ID，避免复用 DexJoCo 的物理语义。这里需要的是适配注册和配置，不意味着必须重写训练框架。

动作的 replay 元数据及 DexArena 控制契约均为 `pd_joint_pos`：单臂 Panda 7 + Sharpa 22 个关节的绝对目标角度，双臂共 58 维。训练与推理的关节顺序、反归一化和控制模式必须一致。

现有 DexJoCo 示例为 action_dim=44、state_dim=46，不适用。单臂纯 qpos 方案应设置 action_dim=29、state_dim=29；双臂纯 qpos 方案为 58/58。混合训练可使用 58 维动作和掩码，但需要明确通道语义、embodiment，以及相机数量处理；建议先验证单任务，再合并。

### 4. 语言目前是环境 ID

`meta/tasks.parquet` 能被现有 `_get_tasks` 读取，其索引文字是 `...-v1__obj_000`，而非自然语言操作指令。这并不一定阻止单任务拟合，但与标准 VLA 的指令输入不一致。

本机 `/data/home/zyh/DexArena/utils/dex_tasks.py` 已有每个任务的 canonical 指令、改写列表、机器人类型，以及 `dataset_episode_instruction`。应复用这套任务协议，生成语言映射，并保持训练/评测一致。每个目录的 task_index 都从 0 开始；合并元数据时需重新映射或维持各子数据集的独立查表。

### 5. 需按 20 Hz 重新核对时间窗口

[加载器](../starVLA/dataloader/lerobot_datasets.py#L194) 使用 `int(sec_chunk * fps)` 决定真实动作片段长度，两帧视觉监督取片段首尾。以现有示例的 sec_chunk=0.15、action_horizon=4 为例，20 Hz 下实际只会取 3 个动作，再在 collator 中补到 4 个；两幅图间隔是 0.10 秒。

因此需要同时明确 sec_chunk、action_horizon、未来图像间隔与控制执行频率，不能直接照搬其他数据集的数值。若希望 4 个连续动作及首尾 0.15 秒的监督，应让加载器产生索引 `[0,1,2,3]`，例如将 sec_chunk 设为 0.20，并核对模型的 horizon_sec 语义。

## Stage 1 与 Stage 2 的区别

- **先做策略 SFT：**可先用已有 LAM 与 VLM 权重，完成上述数据适配后训练 Stage 2。当前 Stage 2 的 `_run_lam_teacher` 对 LAM 传入 `states=None`，因此 LAM 的 max_state_dim 不必跟 flow 的 29/58 维输入强行一致。还应核对所用 LAM checkpoint/config 与训练端视觉、时间条件。
- **做 LAM 领域适配：**这批连续 RGB 数据也适合 Stage 1。若使用关节状态辅助损失，应明确提取 qpos，设置一致的 model/data.max_state_dim，并处理 checkpoint 中状态相关权重的尺寸。默认配置为 32，已有 DexJoCo 适配配置为 14，不能直接塞入原始 74–390 维状态。单臂 29 维 qpos 可以采用 32 维上限加掩码，包含双臂则需至少覆盖 58 维，或另行定义有明确语义的状态目标。
- 现有数据中没有已按本仓库规范准备的 fingertip/contact/depth sidecar。基础 LaWAM 不依赖这些，可先用基础训练配置；若启用对应辅助分支，需另外构建并验证标签。

## 建议的最小落地流程

1. 选一个单臂任务，例如 `CabinetPourDoorOpen-v1`，先验证 100 条轨迹。
2. 在独立输出目录创建适配数据：复用现有视频，整理小体积 Parquet/metadata；补相机映射、关节 qpos 和自然语言，不必重新生成视频。
3. 注册 Sharpa 配置、数据 mixture 和 embodiment，设置匹配的动作/状态维度及时间窗口，保存 Sharpa 自身的归一化统计。
4. 按完整 episode 划分训练/验证。每任务仅 100 条，建议初期留 5–10 条验证；更正式的评估使用独立 rollout 和场景/物体变化。数据目录均为 obj_000，生产端还筛选了成功且较快的演示，不能据此推断跨物体泛化能力。
5. 先检查真实 batch 的图像、指令、关节顺序、动作反归一化与 episode 边界，再做少量 forward/backward 和仿真闭环验证，之后扩大到 29 个单臂任务，最后接入 58 维双臂任务。

## 各任务实测规模

下表中的状态维度是原始观测宽度，不是建议用于训练的 qpos 宽度。

| 任务 | 轨迹 | 帧数 | 原始状态维度 | 动作维度 |
| --- | ---: | ---: | ---: | ---: |
| AngledMobileBaseSharpaOpenCloseBox-v1 | 100 | 12,062 | 85 | 29 |
| AngledMobileBaseSharpaPickAndPlaceBread-v1 | 100 | 17,678 | 116 | 29 |
| AngledMobileBaseSharpaScoopTomato-v1 | 100 | 35,275 | 118 | 29 |
| CabinetPourBottlePickPlace-v1 | 100 | 22,590 | 97 | 29 |
| CabinetPourCapRemovePlace-v1 | 100 | 33,226 | 96 | 29 |
| CabinetPourDoorOpen-v1 | 100 | 13,121 | 79 | 29 |
| CabinetPourPour-v1 | 100 | 21,164 | 92 | 29 |
| DrawerPlugInsert-v1 | 100 | 25,947 | 96 | 29 |
| DrawerPlugOpen-v1 | 100 | 12,502 | 79 | 29 |
| DrawerPlugPickPlace-v1 | 100 | 22,735 | 92 | 29 |
| ScrewingBoltPickPlace-v1 | 100 | 14,159 | 82 | 29 |
| ScrewingNutPickPlace-v1 | 100 | 18,157 | 95 | 29 |
| ScrewingTurn-v1 | 100 | 19,630 | 97 | 29 |
| SharpaAngledClickMouse-v1 | 100 | 8,212 | 149 | 29 |
| SharpaAngledMobilePanelLayFlatLeft-v1 | 100 | 18,145 | 134 | 29 |
| SharpaAngledMobilePipetteAspirate-v1 | 100 | 36,192 | 83 | 29 |
| SharpaAngledMobilePipetteDispense-v1 | 100 | 50,217 | 90 | 29 |
| SharpaAngledMobileRailInsert-v1 | 100 | 27,319 | 153 | 29 |
| SharpaAngledMobileScrewFasten-v1 | 100 | 6,631 | 93 | 29 |
| SharpaAngledOpenLaptop-v1 | 100 | 11,983 | 123 | 29 |
| SharpaAngledTypeKeyboardAllKeys-v1 | 100 | 45,889 | 390 | 29 |
| SharpaCloseFridge-v1 | 100 | 8,189 | 74 | 29 |
| SharpaMobilePlaceBlenderLid-v1 | 100 | 18,231 | 121 | 29 |
| SharpaMobilePressBlenderButton-v1 | 100 | 9,832 | 111 | 29 |
| SharpaOpenFridge-v1 | 100 | 18,115 | 74 | 29 |
| SharpaPackageOpeningExtractAndPlaceApple-v1 | 100 | 27,439 | 85 | 29 |
| SharpaPackageOpeningOpenPackage-v1 | 100 | 19,921 | 128 | 58 |
| SharpaTakeAppleToBlender-v1 | 100 | 26,622 | 88 | 29 |
| TableCleanDumpDustpan-v1 | 100 | 16,760 | 90 | 29 |
| TableCleanPickDebris-v1 | 100 | 40,269 | 88 | 29 |

## 复核产物与范围

全量检查结果：[outputs/sharpa_dataset_audit.json](../outputs/sharpa_dataset_audit.json)。复核脚本：[outputs/audit_sharpa_lawam.py](../outputs/audit_sharpa_lawam.py)。脚本使用本机 lawam 环境的 NumPy/PyArrow/PyAV 及 ffprobe，输出路径固定在本仓库 outputs，可用以下命令复跑：

```bash
/data/home/zyh/miniconda3/envs/lawam/bin/python outputs/audit_sharpa_lawam.py
```

本次完成的是数据兼容性与完整性检查，以及现有相机 fallback 的错误复现；未完成完整 dataset/collator/model 联调、训练、checkpoint 迁移验证或仿真 rollout。训练效果须在上述适配完成后验证。
