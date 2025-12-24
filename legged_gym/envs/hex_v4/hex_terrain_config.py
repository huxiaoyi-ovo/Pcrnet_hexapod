from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexTerrainCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8 #环境数量(降低以减少GPU内存占用)
        
        # 【GPT建议】Debug模式开关：启用Phase 2数据链和curriculum统计的断言检查
        debug_mode = True  # 开发阶段建议开启，生产训练可关闭（提升性能）
        
       
        # 【EGPO Encoder 观测空间架构】
        # 本体观测 (可部署，无需特权信息)
        num_observations = 67  # [quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), torque(18), cmd(3)]
        
        # 特权观测 (训练时可用，部署时需估计)
        # EGPO中特权观测 = obs_vgf(30) + obs_terrain(143) = 173维
        # 但这个值在Runner中未使用，Runner硬编码了实际维度
        num_privileged_obs = 173  # obs_vgf(30) + obs_terrain(143)
        
        # 观测处理流程:
        # 1. obs_buf (67)      → Actor/Critic MLP输入的一部分
        # 2. obs_vgf_buf (30)  → 拼接到obs_buf: [obs + vgf] = 97维
        #    [base_lin_vel(3) + projected_gravity(3) + rb_contact_force(24)]
        # 3. obs_terrain_buf (143) → CNN Encoder: 143 → 32维latent
        #    [Raycast高度图: 11×13点]
        # 
        # 最终网络输入: [obs(67) + vgf(30) + terrain_latent(32)] = 129维
        # 
        # Runner硬编码维度 (expert_guided_encoder_runner.py L51):
        #   actor_obs_shape = [num_obs + 30] = [97]  # obs拼接层
        #   critic_obs_shape = [11*13] = [143]       # terrain编码器输入
        
        
        num_actions = 18
        episode_length_s=10
        env_spacing=2.0
        termination_height_threshold = 0.045#0.035可能低了，导致大量膝盖触地
        termination_max_tilt_deg = 60.0

    class sensor:
        '''传感器相关配置'''
        class depth_camera:
            """深度相机配置
            Phase 1 (EGPO): 使用Raycast高度图，不需要深度相机
            Phase 2/3 (Teacher-Student): 需要深度相机用于导航决策
            """
            enable = False  # Phase 1: 关闭深度相机以节省GPU资源
            width = 128    #    图像宽度
            height = 128   #
            horizontal_fov = 87.0  # 水平视场角（度）
            # 垂直FOV会根据分辨率自动计算
            near_clip = 0.05   # 最近5cm（原来0.1太大）
            far_clip = 5.0    # 最远5m
            #坐标系：x左右，y前后，z上（body link坐标系）
            # 机身尺寸：长0.2m×宽0.44m×高0.05m
            position = [0.00, 0.22, 0.08]  
            pitch_deg = 0.0    # 滚转（验证正确）    
            roll_deg = 20.0   # 俯仰
            yaw_deg = 90.0      # 偏航（0.0默认朝向机器人正右）
            capture_interval = 5  # 每5步采集一次
            # 50Hz控制频率 / 5 = 10Hz深度图更新
            #输出分辨率
            #如果网络需要不同尺寸，设置output_size
            output_size = 128  # 与width/height一致
            # 噪声模型
            add_noise = True
            noise_level = 0.02  # 2%噪声
            # 黑洞比例（模拟反光/透明表面）
            hole_ratio = 0.05  # 5%像素失效
            # 边缘模糊强度
            edge_blur_strength = 0.05
            
    class navigation:
        """导航任务配置"""
        
        # ==================== Phase 控制 ====================
        # Phase 1: EGPO 运动训练 (enable_nav_reward=False)
        # Phase 2: Teacher 导航训练 (use_gt_affordance=True, freeze_student_policy=True)
        # Phase 3: Student 蒸馏训练 (use_gt_affordance=False, freeze_student_policy=True)
        
        # 是否启用导航奖励（启用后会覆盖底层 locomotion reward）
        enable_nav_reward = False
        
        # Phase 2/3: 是否冻结底层运动策略（或使用极小学习率微调）
        freeze_student_policy = False  # Phase 1 设为 False，Phase 2/3 设为 True
        student_fine_tune_lr = 1e-6    # 如果不完全冻结，使用极小学习率
        
        # Phase 2: 使用 GT affordance，Phase 3: 使用估计 affordance
        use_gt_affordance = True       # Phase 2=True, Phase 3=False
        
        # ==================== LocomotionAdapter ====================
        # 是否使用 LocomotionAdapter 解耦高层导航与底层运动
        use_adapter = False            # Phase 1=False, Phase 2/3=True
        
        # Adapter 参数：将 (subgoal, intensity) 转换为 (vx, vy, omega)
        adapter_distance_scale = 2.0   # 距离缩放因子
        adapter_max_ang_vel = 1.0      # 最大角速度（rad/s）
        adapter_heading_gain = 2.0     # 朝向增益
        
        # ==================== 指令计算安全参数 ====================
        min_command_distance = 0.1     # 最小指令计算距离（防止除零）
        max_lin_vel_command = 0.8      # 最大线速度指令 (m/s)
        max_ang_vel_command = 1.5      # 最大角速度指令 (rad/s)
        goal_slowdown_distance = 1.0   # 开始减速的距离 (m)
        goal_min_speed_ratio = 0.2     # 最小速度比例
        
        # ==================== 目标生成 ====================
        # 目标生成模式 
        goal_mode = 'velocity_based'  
        #  'velocity_based', 'fixed', 'random', 'waypoints'
        
        # velocity_based模式：基于速度指令设置目标
        goal_distance = 5.0  # 目标在速度方向5米处
        goal_update_interval = 50  # 每50步更新一次目标
        
        # fixed模式：固定目标位置 
        fixed_goal = [10.0, 0.0]  # [x, y]全局坐标
        
        # random模式：随机采样目标 
        goal_range_x = [3.0, 15.0]   # x方向范围
        goal_range_y = [-8.0, 8.0]   # y方向范围
        goal_min_distance = 3.0      # 最小目标距离
        
        # waypoints模式：预定义路径点
        waypoints = [
            [5.0, 0.0],
            [10.0, 5.0],
            [15.0, 0.0],
            [10.0, -5.0]
        ]
        waypoint_radius = 1.0  # 到达判定半径
        
        #  奖励相关 
        goal_reached_reward = 10.0
        goal_reached_threshold = 0.5  # 到达判定距离（米）
        
        #  Affordance相关
        affordance_grid_size = 16  # 16×16网格
        affordance_cell_size = 0.3125  # 每格0.3125m（5m/16）
        
        # Intensity计算相关 (Step A-P1: 修复作弊)
        max_speed_for_intensity = 0.7  # 用于将速度归一化到[0,1]，适配机器人最大速度

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        border_size = 4.0
        terrain_length = 8.0
        terrain_width = 8.0
        horizontal_scale = 0.1
        vertical_scale = 0.005


        # 训练默认：课程学习开启，selected 关闭
        curriculum = True
        selected = False
        terrain_kwargs = None

        # Curriculum grid
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 1
        
        # === P0.4: collision阈值拆分 ===
        # collision_force_threshold: 终止/硬碰撞阈值
        # collision_penalty_threshold: 惩罚/统计阈值（更敏感）
        collision_force_threshold = 1.0  # Newton（终止判据）
        collision_penalty_threshold = 0.5
        
        # === P1.2: Curriculum质量门槛（基于raw统计） ===
        curriculum_stability_threshold = 0.10  # camera_stability范围[0,1]
        curriculum_height_threshold = 0.15     # base_height范围[0,1]
        curriculum_collision_threshold = 20.0  # 碰撞事件数（使用统一阈值）
        curriculum_quality_score = 2.0         # 4项中至少2项达标
        curriculum_consecutive_passes = 2      # 连续2次才升级（软升级）
        curriculum_distance_threshold = 1.0   # episode累计距离阈值(米)
        curriculum_expert_level_cap_start = 0
        curriculum_expert_level_cap_end = 4
        curriculum_post_expert_freeze_iters = 100
        
        # === 底噪渐进参数 ===
        noise_amplitude_min = 0.005  # 0.5cm
        noise_amplitude_max = 0.020  # 2cm
        noise_downsampled_scale = 0.3  # 低频

        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        slope_treshold = 0.6#0.4会让更多的斜坡地形变成墙面

        #  robot envelope 
        robot_body_width = 0.25
        robot_body_length = 0.40
        robot_swing_abduction = 0.15
        robot_envelope_width = robot_body_width + 2.0 * robot_swing_abduction

        nominal_speed = 0.5
        reaction_time = 0.4

        #  Gate 
        gate_margin_max = 0.50
        gate_margin_min = 0.10# warm-start; later anneal to 0.05 for final eval
        gate_wall_height = 0.60
        gate_wall_thickness = 0.20
        gate_x_frac = 0.65
        gate_door_offset_max = 0.60

        #  Slalom 
        slalom_wall_height = 0.60
        slalom_wall_thickness = 0.20
        slalom_corridor_width_scale = 2.8
        slalom_pillar_size_x = 0.45
        slalom_pillar_size_y = 0.35
        slalom_num_pillars = 6

        #  Gate-on-Slope 
        gate_on_slope_angle_deg = 20.0

        # 10 items: up to slalom; remaining prob -> gate_on_slope (make_terrain else)
        # Phase 1: Focus on basic locomotion (slopes, stairs, obstacles)
        # Phase 2/3: Add navigation terrains (gate, slalom)
        terrain_proportions = [
            0.20,  # smooth slope        - 增加基础地形比例
            0.20,  # rough slope         - 增加基础地形比例
            0.15,  # stairs A            - 增加台阶训练
            0.15,  # stairs B            - 增加台阶训练
            0.30,  # discrete obstacles  - 增加障碍物训练
            0.00,  # stepping stones     - Phase 2/3 启用
            0.00,  # gap                 - Phase 2/3 启用
            0.00,  # pit                 - Phase 2/3 启用
            0.00,  # gate                - Phase 2/3 启用（需要导航能力）
            0.00,  # slalom              - Phase 2/3 启用（需要路径规划）
        ]



    class commands(LeggedRobotCfg.commands):
        max_curriculum = 1.
        num_commands = 3 # lin x y  ang_yaw
        heading_command = False
        resampling_time=10.0
        #越障模式
        curriculum = False
        class ranges:
            lin_vel_x=[-0.6,0.6]
            lin_vel_y=[-0.8,0.8]
            ang_vel_yaw=[-1.0,1.0]
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.1]
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle']
        _joint=0.0
        _swing_joint=0.0
        default_joint_angles ={}
        default_swing_init_angles={}
        angles=[0.5,0.67,-2.2]
        for t in _tao:
            for qn in _q_name:
                if qn == 'thigh':
                    if t == 'rf' or t == 'lb':
                        _joint=angles[0]
                        _swing_joint=0.36
                    elif t == 'lf' or t == 'rb':
                        _joint=-angles[0]
                        _swing_joint=-0.36
                    else:
                        _joint=0.0
                        _swing_joint=0.0
                elif qn == 'knee':
                    _joint=angles[1]
                    if t=='lm' or t=='rm':
                        _swing_joint=1.4
                    else:
                        _swing_joint=1.46
                elif qn == 'ankle':
                    _joint=angles[2]
                    if t=='lm' or t=='rm':
                        _swing_joint=-2.26
                    else:
                        _swing_joint=-2.32
                else:
                    _joint=0.0
                default_joint_angles['j_'+t+'_' + qn]=_joint
                default_swing_init_angles['j_'+t+'_'+qn]=_swing_joint
    class control(LeggedRobotCfg.control):
        # use_actuator_net = False
        use_actuator_net = True
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1.pth"
        # actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0924_1.pth" #目前效果最好
        actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_0929.pth"
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle']
        stiffness={}
        damping={}
        for t in _tao:
            for qn in _q_name:
                stiffness['j_'+t+'_' + qn]=70.0#70.0
                damping['j_'+t+'_'+qn] = 3.0#2.0
        action_scale=0.5
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hex_v4/urdf/hex_ground.urdf"
        name="hex_v4"
        foot_name="toe"
        penalize_contacts_on=["knee","knee","thigh"]
        terminate_after_contacts_on=["body"]
        # terminate_after_contacts_on=[]
        collapse_fixed_joints=False #ankle 和 toe 之间是固定关节，toe接触地面，不能被折叠
        thickness=0.01
        class links: #连杆长度
            l1 = 0.072
            l2 = 0.13
            l3 = 0.17
        class body_shape: #身体形状
            x = 0.1
            y = 0.22
        class depth: #深度相机相关参数
            resolution = [848,480]
            horizontal_fov = 87
            clip_range = [0.2,3.0]

            pass
    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots=False
        friction_range = [0.8,1.2]
        # randomize_base_mass = True
        # added_mass_range = [-1., 1.]

        
    class rewards(LeggedRobotCfg.rewards):
        # ==================== 奖励安全参数 ====================
        # 奖励截断（防止梯度爆炸）
        min_reward_clip = -10.0
        max_reward_clip = 10.0
        low_height_penalty_threshold = 0.05
        low_height_penalty_value = -1.0

        # ==================== 姿态门控（防刷分/异常形态） ====================
        # projected_gravity[:,2] = cos(tilt)；0.7≈45°，0.85≈31.8°
        upright_cos_min = 0.75

        # ==================== 非足端接触惩罚滞后（防噪声误触发） ====================
        # 连续 N 个 control step 检测到非足端接触才触发 collision 惩罚
        nonfoot_contact_hysteresis_steps = 2
        
        # ==================== 相机稳定性细分权重 ====================
        # 这些参数被 _reward_camera_stability() 使用
        camera_jitter_weight = 0.05   # 抖动权重（角加速度）
        camera_wobble_weight = 0.05   # 晃动权重（俯仰/横滚角速度）- 下降以避免抬腿时“过度求稳”
        camera_bobbing_weight = 0.01   # 颠簸权重（垂直加速度）
        
        # 防止偶发尖峰污染curriculum的鲁棒性阈值
        # 这些cap基于正常六足运动的物理约束（不同dt/动力学参数需调整）
        camera_jitter_cap = 50.0     # (rad/s²)² 上限，对应~7rad/s² 角加速度
        camera_wobble_cap = 4.0      # (rad/s)² 上限，对应~2rad/s 角速度
        camera_bobbing_cap = 100.0   # (g 偏差)² 上限：_reward_camera_stability 使用 (a_z - 1g)^2（base_lin_acc单位:g）
        
        # Phase 2/3: 导航时的稳定性保持权重
        nav_stability_weight = 0.3  # 导航时额外添加 camera_stability 的权重

        # ==================== 步态 shaping 超参（容忍偏离） ====================
        # 仅对 swing 腿生效；deadzone 内不惩罚，deadzone 外逐渐衰减
        foot_xy_deadzone_base = 0.03            # [m]
        foot_xy_deadzone_per_difficulty = 0.03  # [m] 难度越高越宽松
        foot_xy_sigma_base = 0.10               # [m]
        foot_xy_sigma_per_difficulty = 0.08     # [m]
        foot_xy_reward_min = 0.0
        foot_xy_reward_max = 1.0

        # ==================== 腾空时间 shaping（避免超长刷分） ====================
        feet_air_time_target_s = 0.1
        feet_air_time_max_s = 0.3
        feet_air_time_long_penalty = 1.0     # 超过max后的惩罚斜率（相对奖励）
        feet_air_time_over_cap_s = 0.60      # 超长惩罚裁剪，避免极端尖峰
        feet_air_time_cmd_threshold = 0.2    # 低指令时不计入

        # ==================== 三角步态（Tripod gait） ====================
        # 基于 expert.py 的 A/B 三角分组：鼓励 3 腿支撑 + 3 腿摆动，减少小踏步
        tripod_cmd_threshold = 0.2
        tripod_stance_weight = 0.4
        tripod_swing_weight = 0.6
        
        class scales(LeggedRobotCfg.rewards.scales):
            # === 核心运动奖励 ===
            tracking_lin_vel = 3.5      # 跟踪线速度指令
            tracking_ang_vel = 2.5      # 跟踪角速度指令
            
            # === 相机稳定性（Sim-to-Real关键）===
            camera_stability = 1.5      # 新增：惩罚机身抖动以提升视觉质量
            lin_vel_z = -1.5            # 惩罚垂直颠簸
            ang_vel_xy = -0.05          # 惩罚俯仰/横滚角速度
            
            # === 姿态与步态 ===
            base_height = 0.5           # 保持目标高度
            orientation = -0.5          # 软姿态约束：防止异常姿态刷分
            feet_air_time = 0.8         # 鼓励合理的摆动相位
            tripod_gait = 1.0           # 鼓励三角步态（基于 expert.py A/B 分组）
            
            # === 惩罚项 ===
            collision = -10.0           # 非足端接触：强惩罚（注意：总奖励仍会被 min/max_reward_clip 截断）
            action_rate = -0.05         # 平滑动作变化
            dof_acc = -1.5e-7           # 关节加速度惩罚
            stand_still = -3.0          # 零指令时保持静止 (修复: 增强惩罚)
            
            # === 能耗（可选）===
            CoT = 0.0                   # 运输成本（已禁用）
            
            # === 六足特定（实验性）===
            tracking_dof = -0.0       # 关节位置跟踪
            footend_pos_xy = 1.0      # 足端位置奖励            
            dof_pos_limits = -0.2     # 关节接近软限位惩罚（防反折/僵直）

        # 软关节限位：避免反折/僵直，但不锁死越障
        soft_dof_pos_limit = 0.95
        only_positive_rewards = False
        tracking_sigma = 0.12
        # tracking_sigma = 0.04
        base_height_target = 0.1
        max_contact_force = 60.0
    
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            actions = 0.5
            quat = 1.0
            ang_vel = 0.25
            lin_acc = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            dof_torque = 0.1
            command = 1.0
            lin_vel = 2.0
            gravity = 1.0
            contact_force = 0.01
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        # add_noise = False
        
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            quat = 0.05
            ang_vel = 0.2
            lin_acc = 0.2
            dof_pos = 0.01
            dof_vel = 1.5
            dof_torque = 1.0
            lin_vel = 0.1
            gravity = 0.05
            contact_force = 10.0
            height_measurements = 0.02
            camera_depth = 0.02
    
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [3.5,0,4]
        lookat = [3.5,5,0]
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        class physx(LeggedRobotCfg.sim.physx):
            num_threads=20
            num_position_iterations=4.0




class HexTerrainCfgPPO(LeggedRobotCfgPPO):

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.8
        
        # actor_hidden_dims = [512,256,256,128]
        # critic_hidden_dims = [512,256,256,128]
        # activation = 'relu'
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        
        # learning_rate = 1.e-4
        # schedule = 'fixed' 
        expert_interface_iter=300 #专家干预的时间
        expert_alpha_min=0.0
        expert_alpha_schedule="cosine"

        pass
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticEncoder'
        algorithm_class_name = 'EGPOEncoder'
        save_interval = 200
        # algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000
        run_name=''
        experiment_name="hex_terrain"
        
        # [Optimization] Lower learning rate for complex EGPO+Encoder architecture
        # Reduced from default 1e-3 to 5e-4 for improved training stability
        learning_rate = 5e-4
        
        load_run=-1
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/resources/expert_data/bc_actor2.pth"
