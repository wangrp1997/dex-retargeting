# 示例 RGB 视频的 MANO 拟合与 GS S1 回放

输入为官方示例 `example/vector_retargeting/data/human_hand_video.mp4`。按 30 Hz 处理全部源帧，数组中的帧号从 0 开始，视频角标从 1 开始。左、中画面裁取相同区域。机器人掌部固定，MuJoCo 视角逐帧随人手方向转动，便于对比手指。黄色提示表示该帧的优化器回退并保留了上一帧关节指令。

- `observations.npz` 保存 MediaPipe 检测出的图像关键点与以手腕为原点的三维关键点。
- `mano.npz` 保存每帧 45 维 MANO 关节姿态、整体旋转、相机系下以手腕为原点的关节和 778 个顶点（米），以及全片共享的形状、尺度和三角面。相机主点取图像中心，焦距近似取图像长边。参数通过三维关键点、二维重投影、姿态先验和时间连续性联合拟合。相机平移、深度及遮挡表面随模型估计。
- `gs_s1.npz` 保存按名称排序的 16 路主动关节目标（弧度）、每帧求解状态与指腹误差。四个非拇指 J1 回放时均随 J2 联动。沿用掌部定位权重 0、拇指到四指相对向量倍率 1.0 的目标。
- `metrics.json` 汇总拟合与机器人运动学误差。重投影的参照是检测器关键点。指腹中心间距是两个选定表面点的距离。独立三维标注与触觉真值均缺失。

这次跟踪对每次回退额外记录了候选解的精确几何检查，见 `solver_diagnostics.csv`。297 个回退帧均尝试了三个步长，三次均未通过掌部 STL 间隙复查。保存的实际关节目标均通过间隙检查。

## 本机复跑

从仓库根目录运行。读取原版 MANO pickle 的步骤使用带 chumpy 的环境。后续使用现有 dex_retargeting 环境和本地 SMPL-X 的 LBS 实现。

```zsh
bench_dir=example/vector_retargeting/data/gs_s1_hand_baselines
input_clip=example/vector_retargeting/data/human_hand_video.mp4
demo_dir=$(mktemp -d /tmp/gs-example-mano.XXXXXXXX)
dex_python=~/miniconda3/envs/dex_retargeting/bin/python
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
export PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
~/miniconda3/envs/grabnet/bin/python "$bench_dir/fit_example_mano.py" prepare \
  ~/Downloads/mano_v1_2/models/MANO_RIGHT.pkl "$demo_dir/model.npz"
"$dex_python" "$bench_dir/fit_example_mano.py" detect "$input_clip" "$demo_dir/observations.npz"
"$dex_python" "$bench_dir/fit_example_mano.py" fit "$demo_dir/model.npz" "$demo_dir/observations.npz" "$demo_dir/mano.npz" \
  --smplx-root ~/miniconda3/envs/DGA/lib/python3.8/site-packages
"$dex_python" "$bench_dir/run_example_mano.py" "$demo_dir/mano.npz" "$demo_dir"
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" "$input_clip" "$demo_dir/mano.npz" \
  "$demo_dir/gs_s1.npz" "$demo_dir/example-mano-gs-s1.mp4" "$demo_dir/observations.npz"
```

## 博雅手停顿与小指对指复查

初版输出保存在本目录。修正后的轨迹和统计在 `revised/`，对应视频为 `~/Videos/example-mano-gs-s1-v2.mp4`。重跑使用同一个 MANO 文件和相同的 30 Hz 时间轴。

掌部和拇指近节间的曲面距离原先在上一次关节位置做一次线性近似，近似失效时整手一起退回。当前实现仍先走线性求解，掌部精确检查拒绝候选解时，重试会在 SLSQP 的每个候选点重新求实际 STL 间隙及梯度。末次检查仍使用原间隙容差。该变动单独消除了本视频的 297 帧回退。源帧 517（17.23 秒）的拇小指指腹中心距仍为 28.7 毫米。

在修正求解器的基础上，`--opposition` 为四对指间向量使用相同的连续权重 `1 + palm_width² / (target_gap² + length_tolerance²)`，并以该权重的平方根跟踪另外四个指腹的 MANO 法线。掌宽由食指和小指指根距离计算，长度容差沿用 10 mm。此选项保持四对目标向量的原长。视频右侧与中间的数值是指腹中心距，不能用来代替两个末节网格的最短距离。新选项只在这个片段进行了探索性评估。

以下命令使用上节的变量及已有的 MANO 结果。

```zsh
"$dex_python" "$bench_dir/run_example_mano.py" "$demo_dir/mano.npz" "$demo_dir/revised" --opposition
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" "$input_clip" "$demo_dir/mano.npz" \
  "$demo_dir/revised/gs_s1.npz" "$demo_dir/revised/comparison.mp4" "$demo_dir/observations.npz" --show-pinky-gap
```

## 第三版的连续对指辅助

离线回放使用同一份右手 MANO 拟合。每帧先独立求出第二版姿态，再从这个姿态微调。拇指和其余四指分别计算整个末节表面的最短距离，覆盖指侧、指端和指腹。人手距离来自每帧变形后的 MANO 三角网格，机器人距离来自 GS S1 末节 STL。单目拟合表面可能相交，此时人手间隙记为零。该版只对四组拇指对指施加闭合辅助。

设每对人手末节间隙经过 One Euro 滤波后为 `h`，距离单位为米。闭合权重连续变化，公式为 `w = closure_weight / (1 + (h / 0.010)²)²`。机器人末节间隙的目标取碰撞检查的 0.5 mm 安全余量；误差使用斜率有界的平滑损失。第二阶段同时惩罚相对第二版参考姿态的关节角偏移，按 12° 归一化，拇指 J1 偏移的平方代价乘四。两阶段各自沿用同一套关节限位、速度上限与逐帧碰撞检查。四个非拇指的被动 J1 均跟随各自的 J2。

以下命令使用上节在本机生成的拟合，以及随仓库提供的示例视频对指观察窗。默认 `--closure-weight 0.35` 为本示例的实验设置，设为零可逐帧复现第二版。新版拟合文件已内含末节面片标签。

```zsh
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
export PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bench_dir=example/vector_retargeting/data/gs_s1_hand_baselines
fitted_dir="$demo_dir"
mkdir -p "$fitted_dir/v3"
cp "$bench_dir/example_rgb_mano/v3/opposition_comparison.csv" "$fitted_dir/v3/"
dex_python=~/miniconda3/envs/dex_retargeting/bin/python
"$dex_python" "$bench_dir/run_example_mano.py" "$fitted_dir/mano.npz" "$fitted_dir/v3" --v3
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" \
  example/vector_retargeting/data/human_hand_video.mp4 "$fitted_dir/mano.npz" "$fitted_dir/v3/gs_s1.npz" \
  ~/Videos/example-mano-gs-s1-v3.mp4 "$fitted_dir/observations.npz" --compare-v2
```

对比视频并排显示原视频与骨架、MANO 投影、第二版和第三版。底部显示该帧四组末节表面间隙，单位为毫米。`v3/gs_s1.npz` 包含两条轨迹、逐帧间隙、实际使用的四组权重及手指弯曲角的变化。`v3/opposition_comparison.csv` 使用 `contact_review/原视频动作索引.csv` 中预先记录的四段对指观察窗，列出各自在窗口内的最小表面间隙和对应帧号。三列的最小值可能出现在不同帧。

在本片段的 621 帧上，关闭闭合辅助后逐帧复现了已保存的第二版关节轨迹。使用默认权重时，相对第二版，五指各两处骨段夹角的平均绝对变化为 0.018°，最大为 1.29°，拇指主动 J1 的最大角度变化为 0.30°。全部输出指令通过关节、速度、联动和碰撞检查，优化回退帧数为零。小指观察窗中，MANO 和第三版的最小间隙分别为 6.97 mm 和 14.76 mm。本片段没有独立接触标注。

## v3.1 的间隙跟踪

v3.1 只改变第二阶段的目标间隙。以食指到小指指根间的掌宽换算人手和机器人尺寸，比例为机器人中立位掌宽除以 MANO 片段掌宽中位数，本例为 0.996。每对指的目标为 `hypot(0.5 mm, 比例 × 滤波后的人手末节间隙)`。人手模型接触时目标为 0.5 mm，分开时连续增大。四组权重、速度上限、碰撞约束和对 v2 关节偏移的惩罚均沿用 v3。视频底部显示各自的实际模型间隙，MANO 栏显示未经滤波的测量值。

以下命令沿用上节环境和路径变量。

```zsh
"$dex_python" "$bench_dir/run_example_mano.py" "$fitted_dir/mano.npz" "$fitted_dir/v3_1" --v3.1
"$dex_python" "$bench_dir/compare_example_gaps.py" "$fitted_dir"
"$dex_python" "$bench_dir/pinky_gap_budget.py" "$fitted_dir" --frame 462
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" \
  example/vector_retargeting/data/human_hand_video.mp4 "$fitted_dir/mano.npz" "$fitted_dir/v3_1/gs_s1.npz" \
  ~/Videos/example-mano-gs-s1-v3.1.mp4 "$fitted_dir/observations.npz" --compare-v3 "$fitted_dir/v3/gs_s1.npz"
```

`v3_1/comparison.csv` 记录相同对指窗内的最小间隙。另从 MANO 中找出从 20 mm 外靠近并再次分开的完整过程，要求期间最小间隙在 3–15 mm 之间，至少含 3 个连续源帧。这些条件仅用于选取评估片段。标记 robot ≤2 mm 且 MANO ≥5 mm 的帧作为全片疑似过度闭合指标，每帧的四个指对分别计数。单目拟合和录像均未提供接触真值，且几何选出的片段可能与人工对指观察窗重叠。逐版本计数与间隙误差保存在 `v3_1/comparison.json`。

`v3_1/pinky_budget_frame_462.csv` 单独记录 15.40 秒处的静态可达性。以该帧 v2 为参考，固定其余三指，只改变拇指与小指。两类预算分别限制骨段间的弯曲角和朝向，后者同时限制近节骨段方向与末节指腹法线。各预算下用多起点局部搜索找最小间隙，逐候选复核关节限位、被动联动和碰撞。表中各行给出找到的可行姿态，静态求解不含逐帧速度约束，也不构成全局最优证明。

该帧 MANO 测量间隙为 7.54 mm，滤波换算后的目标为 7.74 mm。v2、v3 和 v3.1 的实际间隙分别为 15.74、14.76 和 15.01 mm。在静态实验中，固定朝向变化预算为 2° 时，把弯曲预算由 2° 提高到 5°，找到的间隙均为 13.87 mm。固定弯曲预算为 2°，允许朝向变化 5°，则找到 11.43 mm。两项预算设为 10° 时找到 7.45 mm，实际最大弯曲变化为 5.79°。

## v3 对指权重的数值对照

`v3_weight_sweep/comparison.csv` 汇总同一段 621 帧上的实验。脚本复用已保存的第一阶段参考姿态和连续权重，以 30 Hz 从中立位逐帧重跑第二阶段。输出报告分别给出四段对指窗内的间隙、达到 2 mm 以内的帧数、全片误合代理计数，以及相对 v2 的手型变化。标记 MANO ≥5 mm 而机器人 ≤2 mm 的指对帧用于评估，单目 MANO 的估计不能确认真实皮肤接触。

```zsh
"$dex_python" "$bench_dir/sweep_closure.py" "$fitted_dir" "$fitted_dir/v3_weight_sweep" \
  --weight 0.70 --base-cost 0.006 --carry-offset --mcp-flex --gap-target compressed
```

`--base-cost` 乘到各 J4 和拇指 J3 的关节偏移代价，`--mcp-flex` 将相同倍率用于其余四指 J3。拇指 J1 和四指的被动 J1 联动保持原有约束。`--carry-offset` 以当前 v2 参考姿态加上上一帧的修正量初始化优化器。`--gap-target compressed` 将掌宽换算、滤波后的 MANO 间隙 `h` 连续压缩到 `h³/(h² + (10 mm)²)`，再用 `hypot(0.5 mm, 压缩间隙)` 叠加安全余量。无此选项时沿用 v3 的 0.5 mm 目标。

上述命令对应 **v3.2**。下面的命令直接读取这组已验证的关节轨迹，与原版 v3 同帧回放。15.40 秒处可比较拇指与小指的末节间隙。

```zsh
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" \
  example/vector_retargeting/data/human_hand_video.mp4 "$fitted_dir/mano.npz" "$fitted_dir/v3/gs_s1.npz" \
  ~/Videos/example-mano-gs-s1-v3-vs-v3.2.mp4 "$fitted_dir/observations.npz" \
  --compare-v3.2 "$fitted_dir/v3_weight_sweep/w0.70_base0.006_compressed_carry_mcp.npz"
```

## 掌宽换算间隙与开合变化量对照

`v32_gap_motion_ablation/comparison.csv` 复用同一段 621 帧的输入和 v2 参考轨迹。四指掌指关节和拇指根部的偏移代价倍率取 0.006，另设倍率 1 的对照。各组的对指权重上限均为 0.70。 `scaled` 直接跟踪按掌宽换算、滤波后的人手间隙，并沿用 0.5 mm 安全余量。`compressed` 额外使用 v3.2 的连续压缩函数。

开合变化项比较当前间隙与 100 ms 前的间隙之差，机器人端采用该组实际执行的历史姿态，人手端采用同一段经过滤波和掌宽换算的人手间隙。变化项采用与绝对间隙相同的平滑损失，权重为两个时刻对指权重的均值乘以 0、0.35 或 1。轨迹起点和缺少连续历史的帧只使用绝对间隙项。

```zsh
ablation_dir="$fitted_dir/v32_gap_motion_ablation"
for target in compressed scaled; do
  for motion in 0 0.35 1; do
    "$dex_python" "$bench_dir/sweep_closure.py" "$fitted_dir" "$ablation_dir" --weight .70 \
      --base-cost .006 --carry-offset --mcp-flex --gap-target "$target" --motion-weight "$motion"
  done
  "$dex_python" "$bench_dir/sweep_closure.py" "$fitted_dir" "$ablation_dir" --weight .70 \
    --base-cost 1 --carry-offset --mcp-flex --gap-target "$target"
done
"$dex_python" "$bench_dir/score_gap_motion.py" "$fitted_dir" "$ablation_dir"
```

评分使用未经滤波的同帧 MANO 末节表面间隙。四段原视频对指窗共计 78 个目标指对帧，平均绝对误差按指对帧汇总。有符号误差为机器人间隙减人手间隙。松开阶段从各对指窗内 MANO 最小间隙的首帧，到该窗结束后 0.5 秒；取其中连续 100 ms 内人手净间隙增大的 70 个指对帧，分别评估增量的绝对误差和方向一致数。近接指标分母为 MANO 间隙 ≤2 mm 的 115 个指对帧，疑似误合指标分母为 MANO 间隙 ≥5 mm 的 2362 个指对帧。各项均使用单目拟合几何作参照。

**v3.3** 采用 `scaled`、指根代价倍率 0.006 和变化项权重 0。以下命令把 v3.2 和 v3.3 的已保存轨迹放在同一视角中回放。

```zsh
MUJOCO_GL=osmesa "$dex_python" "$bench_dir/render_example_mano.py" \
  example/vector_retargeting/data/human_hand_video.mp4 "$fitted_dir/mano.npz" "$fitted_dir/v3/gs_s1.npz" \
  ~/Videos/example-mano-gs-s1-v3.2-vs-v3.3.mp4 "$fitted_dir/observations.npz" --compare-v3.3 \
  "$ablation_dir/w0.70_base0.006_compressed_carry_mcp.npz" "$ablation_dir/w0.70_base0.006_scaled_carry_mcp.npz"
```

## 同一 MANO 输入上的论文基线与通用指标

复跑入口使用官方 `human_hand_video.mp4` 的 621 个连续右手帧（30 Hz）。此片参与过 Botyard 的开发和选参。五种方法从中立位出发，使用同一个固定腕部的 GS S1 V3.2 模型、16 个主动关节、四个随 J2 联动的非拇指 J1、关节速度和碰撞约束。所有人手输入来自同一次离线 MANO 拟合；只有 Botyard 使用同时拟合得到的表面。每种方法从头运行，启用项目默认的 One Euro 输入滤波。机器人评分使用逐帧指令的 MuJoCo 正向运动学，与同帧、未经 One Euro 滤波的 MANO 作比较。示例缺少独立三维和接触真值。

- DexPilot 采用 `dex-retargeting` 的 DexPilot 目标。AnyTeleop 向量基线取官方 Shadow 五指配置的十个掌部到指尖和中节的向量，适配至 GS S1 后使用尺度 1.0。清华方案使用 `3846d3f` 提供的五指完整目标的固定腕部适配。Wuji 为额外开源对照，采用 `531f6ed` 的解析优化目标。Botyard 使用实时入口相同的两阶段后端和每帧拟合掌宽。
- 清华 Xin 等人（IEEE Robotics and Automation Practice 2026，DOI `10.1109/RAP.2026.3656110`）评估指尖相对腕部、相对拇指的位置，以及末节方向。本表在各自掌坐标系内计算原始毫米欧氏距离，全片评五指；拇指相对误差另报四个预存对指窗中当时参与对指的那根手指。末节方向取从末节关节到指尖的向量夹角，单位为度。腕部固定使世界位置无法提供独立的手腕跟踪信息。
- ManipTrans（CVPR 2025）评估手部关节点和指尖的三维位置误差。本表的 `mpjpe21_wrist_mm` 是掌坐标系适配值，21 点为原点及每指四点，机器人取 proximal、middle、distal、tip 的模型原点。腕部误差按定义为零。人机解剖结构不同，这些绝对长度误差也反映尺寸和关节点定义差异。
- `tip_displacement_100ms_mm` 对相隔三帧的指尖位移向量求人机欧氏误差，再对所有时间窗和五指平均，单位为毫米。它覆盖这段录像中的动作。GeoRT（IROS 2025）在整个手指构型空间上采样方向和位置，评估运动方向保持与空间覆盖率，两者的采样范围不同。
- `bend_deg` 比较每指两处相邻骨段夹角。`thumb_pad_normal_deg` 测拇指内侧指腹法线夹角。`opposition_gap_mae_mm` 比较四个原视频对指窗内参与指对的末节表面最短距离。`matched_near_pairs` 的分母为人手间隙 ≤2 mm 的指对帧数，`false_near_pairs` 的分母为人手 ≥5 mm 的指对帧数，机器人近接以 ≤2 mm 计算。阈值仅用于评分。
- `max_joint_step_p95_deg` 为相邻帧中 16 个主动关节最大角变化的第 95 百分位。`cad_collision_frames` 逐指令检查非相邻 URDF 碰撞几何。`held_frames` 计任一阶段求解回退的帧数。`retarget_p50_ms` 和 `retarget_p95_ms` 计从已拟合 MANO 到关节指令的单帧墙钟时间，含该方法的特征处理与滤波，排除视频检测、MANO 拟合、显示和离线打分。本机 Intel Core i7-14650HX，方法依次运行，各数值库使用单线程配置。

使用本机已准备的右手 MANO 模板和本目录的离线拟合，从仓库根目录运行：

```zsh
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
export PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/miniconda3/envs/dex_retargeting/bin/python \
  example/vector_retargeting/data/gs_s1_hand_baselines/evaluate_example_objectives.py
```

[总体指标](../example_paper_metrics/summary.csv) · [逐指指标](../example_paper_metrics/per_digit.csv)
