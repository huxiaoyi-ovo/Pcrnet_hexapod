from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexTerrainCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8 #环境数量(降低以减少GPU内存占用)
        #[quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)] 67
        num_observations = 67
        # ============================================================
        # 【EGPO 特权观测维度说明】
        # EGPO Runner (expert_guided_encoder_runner.py) 硬编码了维度:
        #   - actor_obs_shape = [num_obs + 30] = [97]
        #   - critic_obs_shape = [11*13] = [143] (高度图)
        # 
        # 特权观测结构:
        #   obs_vgf_buf = [lin_vel(3) + gravity(3) + contact_force(24)] = 30维
        #   obs_terrain_buf = [11×13 Raycast高度图] = 143维
        #
        # 注意: 此值在 EGPO Runner 中实际未被使用（已被硬编码覆盖）
        #       设为30是为了与 obs_vgf_buf 维度保持一致，便于理解
        # ============================================================
        num_privileged_obs = 30  # obs_vgf_buf 维度 (EGPO Runner中未直接使用)
        # num_privileged_obs = None
        num_actions = 18
        episode_length_s=10
        env_spacing=2.0

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
        terrain_proportions = [
            0.10,  # smooth slope
            0.10,  # rough slope
            0.10,  # stairs A
            0.10,  # stairs B
            0.20,  # discrete obstacles
            0.00,  # stepping stones
            0.00,  # gap
            0.00,  # pit
            0.20,  # gate
            0.10,  # slalom
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
                stiffness['j_'+t+'_' + qn]=100.0
                damping['j_'+t+'_'+qn] = 0.8
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
        
        # ==================== 相机稳定性细分权重 ====================
        # 这些参数被 _reward_camera_stability() 使用
        camera_jitter_weight = 0.05   # 抖动权重（角加速度）
        camera_wobble_weight = 0.5    # 晃动权重（俯仰/横滚角速度）
        camera_bobbing_weight = 0.1   # 颠簸权重（垂直加速度）
        
        # Phase 2/3: 导航时的稳定性保持权重
        nav_stability_weight = 0.3  # 导航时额外添加 camera_stability 的权重
        
        class scales(LeggedRobotCfg.rewards.scales):
            # === 核心运动奖励 ===
            tracking_lin_vel = 2.0      # 跟踪线速度指令
            tracking_ang_vel = 1.5      # 跟踪角速度指令
            
            # === 相机稳定性（Sim-to-Real关键）===
            camera_stability = 2.5      # 新增：惩罚机身抖动以提升视觉质量
            lin_vel_z = -2.0            # 惩罚垂直颠簸
            ang_vel_xy = -0.05          # 惩罚俯仰/横滚角速度
            
            # === 姿态与步态 ===
            base_height = 0.5           # 保持目标高度
            orientation = 0.0           # 保持水平（由camera_stability覆盖）
            feet_air_time = 0.5         # 鼓励合理的摆动相位
            
            # === 惩罚项 ===
            collision = -1.0            # 非足端接触
            action_rate = -0.05         # 平滑动作变化
            dof_acc = -3.0e-7           # 关节加速度惩罚
            stand_still = -2.0          # 零指令时保持静止
            
            # === 能耗（可选）===
            CoT = 0.0                   # 运输成本（已禁用）
            
            # === 六足特定（实验性）===
            # tracking_dof = -0.1       # 关节位置跟踪
            # footend_pos_xy = 3.0      # 足端位置奖励            


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
        expert_interface_iter=200 #专家干预的时间

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
        load_run=-1
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/resources/expert_data/bc_actor2.pth"
