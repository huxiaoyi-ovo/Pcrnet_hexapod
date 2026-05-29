## locomotion

__机器人规划控制的核心代码__
"curv_adapt"用于崎岖地形（变曲率表面）运动控制
"cross_plane"用于跨壁面，包含轨迹规划（gcopter）和控制

## interface

__机器人与仿真或实物交互__
接收locomotion的topic，调用仿真或实物控制接口发送电机控制指令，重置机器人等
发送本体传感信息，电机状态，角速度，身体位姿

## sim_env

__仿真环境配置__
model中存放机器人urdf模型和地形环境等信息
包括gazebo和Isaac gym

注：使用isaac gym加载urdf模型文件时，需要使用.obj格式的网格文件
1、使用SolidWorks导出urdf，模型.STL格式，使用网站："https://imagetostl.com/cn/convert/file/stl/to/obj#convert"转化为.obj格式。
2、由于isaac gym加载.obj文件会自动将坐标系绕x轴旋转-90°，所以把所有的点都绕原坐标系旋转90°，使用src/sim_env/gym/obj_axes_trans.py文件
3、使用xacro文件编写urdf文件，球关节+多个腿部连杆，会导致gym.get_asset_dof_names(hexa_asset)输出的关节名称错误，原因未知，采用多个连接的旋转关节近似

## utils

__通用工具__
C++中的彩色输出 color_cout等

## preception

__定位与环境感知__
发送机器人定位信息，线速度等

## python 中，单个程序中有多个Subscriber时，rospy.spin()会采用多线程同时处理这些回调函数，而不是依次运行

## isaacgym中，同一个actor下，设置碰撞filter后，部分刚体之间的碰撞检测正常，部分刚体之间的碰撞检测失效， 如果设置为不同的actor，全部正常，原因未知

---

# PCR 实机部署说明

## 目标

PCR 实机模式用于：

```text
D435i 相机
-> 人类目标相对坐标 + 障碍物 local_map_2ch
-> PCR 高层网络
-> /usr/command
-> run_agent2.py
-> /sita_des
-> 电机
```

原则是最小兼容：不改已经能跑真机的 `run_agent2.py`、`joy_command.msg` 和电机控制链路。PCR 只作为新的速度命令来源接入 `/usr/command`。

## 新增文件位置

PCR 实机相关文件放在：

```text
interface/scripts/pcr_real/
```

文件作用：

```text
real_pcr_input_check.py
    D435i + YOLO，生成 PCR 观测，并可发布 ROS topic。发布内容包括目标相对坐标、local_map_2ch 和 actor_difficulty。

pcr_realplay.py
    读取 PCR 观测，加载 PCR/avoid 网络，输出 /usr/command。

run_pcr_real_compat.sh
    PCR 实机专用启动脚本。不会启动 joy_ctrl。

keyboard_usr_command.py
    低速键盘控制脚本，按住按键时发布 /usr/command，松开后自动停，用于验证 run_agent2.py 到电机的运动链路。

high_level_planner.py
expert_s0_follow.py
    PCR 推理所需的最小网络/专家代码。

yolov8n.pt
    YOLO person 检测模型。

checkpoints/avoid_best.pt
    Avoid expert checkpoint。

checkpoints/moe_teacher_best_learnedw.pt
    当前 PCR 主策略 checkpoint。

checkpoints/moe_teacher_best_yonly.pt
checkpoints/moe_teacher_best_w0.15.pt
    PCR 对照/备用 checkpoint。

checkpoints/low_level_best.pt
    仿真 PCR 低层策略记录；当前实机链路默认不直接加载它。
```

底层实机策略文件已放在：

```text
agent/base_line200.pt
agent/fast_2000.pt
agent/EGPO_2000.pt
agent/EGPO_3000.pt
```

其中当前 PCR 实机默认使用：

```text
agent/base_line200.pt
```

注意：`run_agent2.py` 当前从 `/home/nvidia/agents/` 加载底层策略，所以部署到上位机后需要执行一次：

```bash
mkdir -p /home/nvidia/agents
cp agent/base_line200.pt /home/nvidia/agents/base_line200.pt
```

如果切换其他底层策略，例如 `fast_2000.pt`，也要先复制到 `/home/nvidia/agents/`，然后启动时加：

```bash
--lowlevel_agent fast_2000.pt
```

当前 `src_real` 已包含 PCR 实机所需 checkpoint，不需要再手动补放。若以后换新的 PCR 主策略，放到 `interface/scripts/pcr_real/checkpoints/`，启动时用 `--pcr_ckpt` 指定即可。

## 环境要求

上位机需要能正常运行原 `src_real` 工作空间，并额外具备：

```text
ROS1 / rospy
torch
numpy
opencv-python
pyrealsense2
ultralytics
```

如果缺少 Python 包，先在上位机对应 Python 环境里安装。不要在机器人上临时改底层代码。

## 编译与 source

在上位机上进入本工作空间根目录，也就是包含 `interface/ locomotion/ perception/ real_env/ sim_env/ utils/` 的目录。

常规 ROS 编译后 source：

```bash
catkin_make
source devel/setup.bash
```

确认 `interface` 包可见：

```bash
rospack find interface
```

## 重要安全规则

PCR 模式下不要启动默认 `manage.launch`，因为它会启动 `joy_ctrl`，而 `joy_ctrl` 也会发布 `/usr/command`。

禁止同时存在：

```text
joy_ctrl -> /usr/command
PCR      -> /usr/command
```

PCR 专用脚本会检测 `/joy_ctrl`，如果它已经运行，会拒绝启动。

手柄模式和 PCR 模式分开使用：

```text
手柄模式：使用原 manage.launch
PCR 模式：使用 interface/scripts/pcr_real/run_pcr_real_compat.sh
```

如果已经在 ROS 工作空间内完成 `catkin_make` 和 `source devel/setup.bash`，也可以使用 PCR 专用 launch。它不会启动 `joy_ctrl`，默认同样不发布运动命令：

```bash
roslaunch interface pcr_real_compat.launch
```

## 干跑检查

先不要让机器人动，只检查相机输入和 PCR 输出。

```bash
bash interface/scripts/pcr_real/run_pcr_real_compat.sh \
  --show
```

此时不会发布运动命令。需要确认：

```text
target_valid 大部分时间为 true
goal_buf 不是 NaN
local_map_2ch 为 2x32x32
cmd_safe 有限
target_lost / depth_invalid 时 cmd_safe 为 0
没有 joy_ctrl 同时运行
```

## 低速发命令

干跑正常后，再允许 PCR 发布 `/usr/command`：

```bash
bash interface/scripts/pcr_real/run_pcr_real_compat.sh \
  --show \
  --publish_cmd
```

对应的 launch 写法是：

```bash
roslaunch interface pcr_real_compat.launch show:=true publish_cmd:=true
```

默认限幅很低：

```text
max_cmd_x   = 0.06
max_cmd_y   = 0.10
max_cmd_yaw = 0.20
```

PCR 输出到 `run_agent2.py` 的换算关系是：

```text
x_vec   = cmd_x_right   / 1.6
y_vec   = cmd_y_forward / 2.4
w_twist = cmd_yaw       / 0.375
```

这对应 `run_agent2.py` 里的命令缩放。

## 键盘低速控制

如果只想验证底层运动链路，不启动相机和 PCR，可以用键盘直接发布 `/usr/command`：

```bash
python3 interface/scripts/pcr_real/keyboard_usr_command.py
```

按键：

```text
按住 W/S/A/D/Q/E/R/F 才会持续运动，松开后自动停。
W/S: 前进 / 后退
A/D: 左移 / 右移
Q/E: 左转 / 右转
R/F: z_vec 正/负
Space 或 X: 立即清零
1: 发送一次 set_init，相当于手柄 B
2: 发送一次 moving init，相当于手柄 Y after B
3: 发送一次 stop，相当于手柄 A after moving init
4: 发送一次 change_mode，相当于右侧轮盘按钮
5: 发送一次 disable_pump
6: 发送一次 disable_torque
7: 发送一次 action_valve
Ctrl+C: 清零并退出
```

默认速度很低：

```text
x_speed   = 0.06 m/s
y_speed   = 0.10 m/s
z_speed   = 0.06
yaw_speed = 0.20 rad/s
```

脚本默认优先使用真实按下/松开事件。如果上位机缺少对应 Python 包，会自动退回终端按键重复模式；这种模式下也会在松开后短时间自动停。

键盘控制、手柄控制、PCR 发命令三者同一时间只能开一个。若 `joy_ctrl` 已运行，键盘脚本默认会拒绝启动。

## 单独启动相机观测

如果只想看 D435i 输入，不启动 PCR：

```bash
python3 interface/scripts/pcr_real/real_pcr_input_check.py \
  --yolo_model interface/scripts/pcr_real/yolov8n.pt \
  --width 640 \
  --height 480 \
  --fps 30 \
  --map_size 32 \
  --map_extent_m 3.0 \
  --camera_height_m 0.37 \
  --camera_pitch_down_deg 15 \
  --yolo_conf 0.35 \
  --target_depth_mode roi \
  --ground_remove_height_m 0.04 \
  --debug_map_px 260 \
  --publish_ros \
  --show
```

## 单独启动 PCR 推理

如果相机 topic 已经在发布，可单独启动 PCR：

```bash
python3 interface/scripts/pcr_real/pcr_realplay.py \
  --pcr_ckpt interface/scripts/pcr_real/checkpoints/moe_teacher_best_learnedw.pt \
  --avoid_ckpt interface/scripts/pcr_real/checkpoints/avoid_best.pt \
  --cmd_backend usr_command \
  --device cpu \
  --rate_hz 10 \
  --risk_memory
```

加 `--publish_cmd` 才会真正发布 `/usr/command`。

注意：`real_pcr_input_check.py` 发布的 `/pcr/target_state` 当前字段为：

```text
[x_right, y_forward, v_right, v_forward, valid, target_too_close, depth_invalid, actor_difficulty]
```

`pcr_realplay.py` 会优先使用这里的 `actor_difficulty`，保证显示出来的 difficulty 和 PCR 实际收到的 difficulty 一致。

## 方向验收顺序

正式跟人前，先做最小方向验收：

```text
1. 小正 y_forward：机器人应向前。
2. 小正 x_right：机器人应向右。
3. 小正 yaw：确认实机转向符号，再决定是否需要在 PCR 输出端改符号。
```

如果方向不对，先停机，不要通过调大速度来观察。

## 常见问题

### 找不到 interface.msg.joy_command

说明没有 source 工作空间：

```bash
source devel/setup.bash
```

### 找不到 /home/nvidia/agents/base_line200.pt

复制底层策略：

```bash
mkdir -p /home/nvidia/agents
cp agent/base_line200.pt /home/nvidia/agents/base_line200.pt
```

### PCR checkpoint 不存在

当前默认 PCR 主策略应存在于：

```text
interface/scripts/pcr_real/checkpoints/moe_teacher_best_learnedw.pt
```

如果后续换新策略，把新的 `.pt` 放到 `interface/scripts/pcr_real/checkpoints/`，然后启动时指定实际路径：

```bash
--pcr_ckpt /path/to/your_pcr_checkpoint.pt
```

### joy_ctrl 已运行

PCR 启动脚本会拒绝继续。停止 `joy_ctrl` 后再运行 PCR 模式。

### 目标丢失

`target_valid=false` 时 PCR 会发零速度。先修 YOLO/深度目标链路，不要绕过这个安全判断。
