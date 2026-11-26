# DexMV 芥末瓶搬运的 v3.2 与 v3.3 对照

两版共用源帧 11–122、25 Hz、原始 DexMV 右手 MANO 参数和瓶子标注、第一阶段手型、腕部轨迹、自由瓶初始姿态及仿真参数。第一阶段保留弯曲、指腹朝向和指间向量目标。第二阶段沿用对指权重 0.70、指根关节代价倍率 0.006、上一帧修正量初始化和四指被动 J1 联动。v3.2 额外压缩按掌宽换算的间隙，v3.3 直接跟踪换算后的间隙。两版均关闭开合变化项。

视频上排用同一标注瓶位比较 MANO 重建与机器人的关节指令，下排展示人手参考轨迹和两版在独立自由瓶仿真中的结果。机器人仿真只规定腕部运动，通过执行器跟踪指关节命令。瓶子初始状态之后的位置由重力、桌面与手部接触决定。桌面来自首段瓶姿估计，仿真瓶质量约 0.717 kg。稳定抓取要求在有手部支撑、无桌面支撑时瓶底离桌至少 50 mm 并连续保持至少 0.5 秒。

v3.3 在这条序列上未能抓起自由瓶。在下面的 `mustard_versions.py` 命令中加入 `--settle-bottle` 并指定新的输出目录，可启用多点接触和一秒放稳过程。关闭手瓶碰撞的对照中，瓶子水平漂移约 0.003 mm。开启碰撞后，源帧 53 拇指先顶到瓶子，源帧 61 瓶身倾斜超过 45°。[放稳测试指标](../mustard_v33_settling_check.json)

`comparison.csv` 分开记录运动指令误差和自由瓶结果。手指到瓶面的距离用 MANO 末节三角面或 GS S1 末节 CAD 到两片标注瓶体碰撞网格的最短无符号距离。表面相交时记为零。单帧姿态的表面间隙无法测出夹持力。抬升段由人手演示的瓶底高度确定，本序列为源帧 80–103。仿真中检测到手瓶软接触穿透，详细峰值与独立 CAD 自碰撞标记见两版 JSON。

从仓库根目录运行。下列命令使用已有 ROS Humble 和 dex_retargeting 环境；MANO 表面由本机授权模型重建到临时目录。

```zsh
bench=example/vector_retargeting/data/gs_s1_hand_baselines
dst="$bench/mustard_v32_v33"
scratch=$(mktemp -d /tmp/gs-mustard.XXXXXXXX)
~/miniconda3/envs/grabnet/bin/python "$bench/dexmv_mano_surface.py" ~/Downloads/mano_v1_2/models/MANO_RIGHT.pkl \
  ~/Projects/dexmv-sim/examples/retargeting_source/relocate_mustard_example_seq/hand_pose "$scratch/mano.npz"
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
export PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
dex_python=~/miniconda3/envs/dex_retargeting/bin/python
"$dex_python" "$bench/mustard_versions.py" "$scratch/mano.npz" "$dst" --with-control
MUJOCO_GL=osmesa "$dex_python" "$bench/render_mustard_versions.py" "$scratch/mano.npz" "$dst" \
  ~/Videos/dexmv-mustard-v3.2-vs-v3.3.mp4
```
