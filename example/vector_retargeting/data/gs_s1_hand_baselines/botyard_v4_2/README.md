# Botyard v4.2

算法名称保持 `botyard`。相机入口使用 `objective:=botyard`，当前默认模式为 `botyard_shape:=v4_2`。

v4.1 指上一轮姿态保护验证版，对指阶段仍把拇指朝向场误差增加限制在 2°以内。v4.2 把这项硬限制改成弱偏好，让拇指可以为食指对指调整。当前求解目标、源姿态保护和完整控制器均已进入项目，不再依赖临时实验 Python 模块。

## 优化过程

1. 用机器人骨长下的单位骨段方向和内在弯曲角拟合 MANO；四指 J4 随观测变化。
2. 固定四指与拇指 J1/J2，优化拇指 J3/J4 的整体指向和朝向掌内、其他手指的偏好。
3. 在每指源姿态误差、拇指整体指向、屈曲平面、关节限位、速度、联动和碰撞约束下，优化真实末节 STL 的最短间隙。拇指朝向场作为弱偏好，而非 2°硬限制。

使用 NLopt SLSQP 和解析雅可比。连续权重为 `exp(-(MANO_gap/(0.15*MANO_palm_width))**2)`；距离目标停在 0.5 mm 碰撞余量，长度单位为 10 mm。相对主阶段的屈伸范围为 ±10°，拇指整体指向范围为 15°，四指屈曲平面允许额外 1°的平方正弦残差。每指弯曲 RMS 与四指近节方向的绝对源误差容差为 10°；当主阶段误差高于容差时，附加任务不能撤销其改善。朝向软代价系数为 `0.02*sum(weights)`，另有系数 0.03 的主姿态偏移代价，角度单位均为 20°。

上一实际指令只有满足当前帧源姿态和物理约束，才可作为更优候选与初值。数值求解失败时保留已评估的最优可行候选；主阶段失败时保持上一实际指令，并记录异常。

## 当前验证与边界

同输入验证覆盖 example 全部 621 帧、用户录像 395 帧（成功跟踪 330 帧）以及张手横扫 149 帧。用户录像两段食指对指的间隙中位数从约 11.9 / 9.7 mm 降到约 1.6 / 1.5 mm。所有输出通过几何、关节范围、速度与源姿态保护检查；原有握拳退化没有重现。全套测试在本次版本更名前为 86 通过。

接近效果并非各指都已解决：example 无名指、小指窗口的最小间隙仍约 2.3 / 17.9 mm；小指比上一轮朝向硬限制验证版的约 9.6 mm 更大。当前结果不能证明这些间隙不可达，也不能证明局部求解为全局最优。example 的两次握拳四指平均方向误差约 23.8° / 35.5°；无对指参考约为 19.0° / 28.5°。拇指定制朝向不计入该方向误差。

example 在第 259 帧有一次主阶段数值异常，用户录像在第 119 帧有一次附加阶段异常，均使用可行保持。MANO 间隙只是单目模型几何，不能确认真实皮肤接触；网格距离在最近三角面切换时分段可微。

MANO 拟合仍只对拇指末节做方向修正，不保证在近节拟合不准时整根拇指完全伸直。原有合成短拇指测试的整指弯曲断言属于此前已弃用的整拇指拟合设计；测试现改为检查末节方向改善、其他关节点保持和真实可见弯曲保留，没有调整源拟合参数。

## 使用与复跑

```zsh
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
ros2 launch "$PWD/example/vector_retargeting/gs_s1_preview.launch.py" \
  backend:=mujoco objective:=botyard botyard_shape:=v4_2 botyard_gradient:=analytic
```

同一录像会话可使用 `compare_recorded_shapes.py /path/to/session --shapes sweep v4_2 --gradient analytic --out /tmp/botyard-comparison.csv`，需要源码 `src`、`example/vector_retargeting` 和其 `data/gs_s1_hand_baselines` 在 PYTHONPATH。原始 RGB 视频也可用上述 launch 的 `video:=/path/to/human.mp4` 入口。

`validate_botyard.py --source /path/to/source.npz --scene /path/to/scene.xml --out /tmp/botyard-v4.2 --label example` 用同一个公开控制器回放缓存 MANO。缓存必需数组为 `joints`、`vertices`、`time_s`、`dt`，可选 `frame`、`valid`。使用 `--baseline-controls` 可加入同帧对照轨迹。

视频、逐帧轨迹、MANO 模板和大型诊断产物保留在本地并由 gitignore 排除。Python 包仍使用上游包版本；GS S1 V3.2 是 CAD 型号。
