from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexTerrainCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8 #环境数量(降低以减少GPU内存占用)
        #[quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)] 67
        num_observations = 67
        #[lin_vel(3), gravity(3), contact_force(6) ,measured_heights(143)] 199 + above_obs 199+67=266 #contact_force noly contains the z axis
        num_privileged_obs = 240
        # num_privileged_obs = None
        num_actions = 18
        episode_length_s=10
        env_spacing=2.0

    class sensor:
        '''传感器相关配置'''
        class depth_camera:
            """深度相机配置"""
            enable = True  # 是否启用深度相机
            width = 128    #    图像宽度
            height = 128   #
            horizontal_fov = 60.0  # 水平视场角（度）
            # 垂直FOV会根据分辨率自动计算
            near_clip = 0.05   # 最近5cm（原来0.1太大）
            far_clip = 5.0    # 最远5m
            #坐标系：x左右，y前后，z上（body link坐标系）
            # 机身尺寸：长0.2m×宽0.44m×高0.05m
            position = [0.00, 0.22, 0.15]  # 
            pitch_deg = 0.0    # 
            roll_deg = 45.0  # 向下倾斜45度
            yaw_deg = 0.0      
            capture_interval = 10  # 每10步采集一次
            # 50Hz控制频率 / 10 = 5Hz深度图更新
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

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        # mesh_type = 'plane'
        border_size=1.0
        terrain_length=8.0
        terrain_width=8.0
        max_init_terrain_level=1 #这个必须比num_rows小，否则会超出索引边界
        num_rows=5 #等级
        num_cols=10 #不同地形种类的总数量，比例按照 terrain_proportions来
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #11
        measured_points_y = [ -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #13 1mx1.2m rectangle (without center line)        
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
        #开启了地形选择，就按照参数中的地形生成
        selected=True
        num_sub_terrains=1
        terrain_kwargs={"type":"terrain_utils.pyramid_stairs_terrain",
                        "step_width":0.31,
                        "step_height":-0.07,
                        "platform_size":3}
        slope_treshold=0.4


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
        class scales(LeggedRobotCfg.rewards.scales):
            action_rate = -0.05
            tracking_ang_vel = 1.5
            tracking_lin_vel = 2.0
            lin_vel_z = -3.0
            ang_vel_xy = -0.1
            base_height = 0.5
            orientation = -2.0
            feet_air_time = 2.0
            collision = -1.0

            dof_acc=-3.0e-7
            # dof_vel = -2.0e-5

            stand_still = -2.0
            # feet_contact_forces = -0.004

            CoT = -0.001
            pass
            #针对六足添加的奖励：
            # footend_pos_xy = 3.0 #距离swing_init_point的xy值越近，奖励越高
            tracking_dof = -0.1 #pos_des和dof_vel距离越大，惩罚越大
            #足端力的增加变化，增加的越快，惩罚越大
            # feet_contact_forces_increase = -0.0005 
            #
            # swing=1.0 #摆动时距离swing_init_point越近，奖励越高，靠近到一定阈值后，距离越远，奖励越高[并不好用]

            # 设计RF与RB；LF和LB的对角奖励
            # mirror=1.0

            action_rate = -0.05
            tracking_ang_vel = 1.5
            tracking_lin_vel = 2.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            base_height = 0.5
            orientation = 0.0
            feet_air_time = 0.5
            collision = -1.0

            dof_acc=-3.0e-7
            # dof_vel = -2.0e-5
            CoT=0.0

            stand_still = -2.0
            # feet_contact_forces = -0.004


            # CoT = -0.001
            pass
            #针对六足添加的奖励：
            # footend_pos_xy = 3.0 #距离swing_init_point的xy值越近，奖励越高
            # tracking_dof = -0.1 #pos_des和dof_vel距离越大，惩罚越大            


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
