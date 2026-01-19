import math

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexGroundCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 #环境数量
        #[quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)] 67
        # num_observations = 67
        num_observations = 75
        # num_observations = 218 # 75+11*13=75+143=218
        #[lin_vel(3), gravity(3), contact_force(6) ,measured_heights(187)] 199 + above_obs 199+67=266 #contact_force noly contains the z axis
        # num_privileged_obs = 222 #67+12+11*13
        # num_privileged_obs = 218
        num_privileged_obs = 230 #3+3+6+11*13=155 155+75=230
        # num_privileged_obs = None
        num_actions = 18
        episode_length_s=15
        env_spacing=2.0
        termination_height_threshold = 0.035
        termination_max_tilt_deg = 60.0
    class sensor:
        class depth_camera:
            enable = False  # play/调试时可打开
            width = 128
            height = 128
            horizontal_fov = 87.0
            near_clip = 0.05
            far_clip = 5.0
            # 坐标系：x左右，y前后，z上（body link坐标系）
            position = [0.00, 0.22, 0.08]
            pitch_deg = 0.0
            roll_deg = 20.0
            yaw_deg = 90.0
            capture_interval = 5
            output_size = 128
            add_noise = True
            noise_level = 0.02
            hole_ratio = 0.05
            edge_blur_strength = 0.05
    class navigation:
        """固定目标避障（Phase 2 基础任务）"""
        # 目标生成模式
        goal_mode = 'fixed'  # random / fixed
        # random 模式：相对 env origin 的范围
        goal_range_x = [2.0, 5.0]
        goal_range_y = [-3.0, 3.0]
        goal_min_distance = 2.0
        goal_reached_threshold = 0.5
        # 采样控制：确保直线路径被障碍物阻挡（需要绕行）
        goal_sample_max_tries = 20
        goal_line_samples = 16
        goal_obstacle_height_threshold = 0.2
        goal_allow_fallback = True
        # fixed 模式：相对 env origin 的固定目标
        fixed_goal = [0.0, 0.0]
        resample_on_reach = False
        # 出生点采样：边缘随机，朝向中心并加随机扰动
        spawn_edge_enable = True
        spawn_edge_margin = 0.3
        spawn_outside_margin = 0.2
        spawn_yaw_jitter_deg = 20.0
        heading_offset_rad = 0.5 * math.pi  # 标准: +pi/2, 机体 +Y 为前进
        crossable_height_max = 0.08
        crossable_width_margin = 0.0
        crossable_sector_deg = 60.0
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        #mesh_type = 'plane'
        curriculum = True
        border_size=1.0
        terrain_length=8.0
        terrain_width=8.0
        max_init_terrain_level=4 #这个必须比num_rows小，否则会超出索引边界
        num_rows=5 #等级
        num_cols=10 #不同地形种类的总数量，比例按照 terrain_proportions来
        measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] #11
        measured_points_y = [ -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #13 1mx1.2m rectangle (without center line)        
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, pit, gate, slalom]
        # 仅使用离散障碍地形，难度随 curriculum 行递增
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #开启了地形选择，就按照参数中的地形生成
        selected=False
        num_sub_terrains=1
        '''
        terrain_kwargs={"type":"terrain_utils.pyramid_stairs_terrain",
                        "step_width":0.31,
                        "step_height":-0.09,
                        "platform_size":2}
        '''
        terrain_kwargs={"type":"terrain_utils.discrete_obstacles_terrain",
                        "max_height":1.0,
                        "min_size":0.5,
                        "max_size":1.0,
                        "num_rects":5,
                        "platform_size":2.}
        # 关闭底噪，确保仅有离散障碍物
        noise_amplitude_min = 0.0
        noise_amplitude_max = 0.0
        noise_downsampled_scale = 0.3
        # 固定布局开关与参数
        fixed_layout_enable = True
        fixed_layout_ring_half_size = 2.2
        fixed_layout_gap_min = 0.45
        fixed_layout_gap_max = 1.0
        fixed_layout_gap_buffer = 0.1
        fixed_layout_robot_clearance = 0.27
        fixed_layout_wall_thickness = 0.25
        fixed_layout_center_clearance = 0.6
        fixed_layout_gap_center_offset_deg = 45.0
        fixed_layout_high_height_min = 0.25
        fixed_layout_high_height_max = 0.35
        fixed_layout_low_height_min = 0.08
        fixed_layout_low_height_max = 0.12
        fixed_layout_cyl_radius_min = 0.12
        fixed_layout_cyl_radius_max = 0.12
        fixed_layout_cyl_offset = 0.7
        fixed_layout_rand_cyl_num_min = 5
        fixed_layout_rand_cyl_num_max = 10
        fixed_layout_rand_cyl_r_min = 0.8
        fixed_layout_rand_cyl_r_max = 1.6
        fixed_layout_rand_shape_box_prob = 0.5
        fixed_layout_rand_box_size_min = 0.25
        fixed_layout_rand_box_size_max = 0.6
        # mixed terrain placeholders (disabled by default)
        mixed_enable = False
        mixed_roughness_enable = False
        mixed_roughness_scale = 0.02
        mixed_roughness_downsampled_scale = 0.3
        mixed_obstacle_enable = False
        mixed_obstacle_num_rects_min = 4
        mixed_obstacle_num_rects_max = 12
        mixed_obstacle_min_size = 0.3
        mixed_obstacle_max_size = 0.6
        mixed_obstacle_height_min = 0.12
        mixed_obstacle_height_max = 0.28
        # 离散障碍参数（用于 curriculum 地形）
        discrete_obstacles_height_min = 0.12
        discrete_obstacles_height_max = 0.28
        discrete_obstacles_min_size = 0.3
        discrete_obstacles_max_size = 0.8
        discrete_obstacles_num_rects_min = 6
        discrete_obstacles_num_rects_max = 14
        discrete_obstacles_platform_size = 2.0
        # slalom走廊加宽（相对默认2.8扩大1.5倍）
        slalom_corridor_width_scale = 4.2
        slope_treshold=0.1
        collision_force_threshold = 1.0
        collision_penalty_threshold = 0.5


    class commands(LeggedRobotCfg.commands):
        max_curriculum = 1.
        num_commands = 3 # lin x y  ang_yaw
        heading_command = False
        resampling_time=10.0
        #越障模式
        # curriculum = False
        # class ranges:
        #     lin_vel_x=[-0.6,0.6]
        #     lin_vel_y=[-0.7,0.7]
        #     ang_vel_yaw=[-1.0,1.0]
        #冲击速度模式
        curriculum = True
        class ranges:
            lin_vel_x=[-1.0,1.0]
            lin_vel_y=[-1.5,1.5]
            ang_vel_yaw=[-2.0,2.0]            
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.1]
        rot = [0.0, 0.0, 0.0, 1.0]
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
        penalize_contacts_on=["thigh","knee","ankle"]
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
        friction_range = [0.4,0.8]
        # randomize_base_mass = True
        # added_mass_range = [-1., 1.]

        
    class rewards(LeggedRobotCfg.rewards):
        low_height_penalty_threshold = 0.05
        low_height_penalty_value = -1.0
        feet_contact_force_threshold = 1.0
        zero_cmd_threshold = 0.1
        zero_cmd_contact_force_threshold = 1.0
        zero_cmd_dof_vel_eps = 0.05
        zero_cmd_action_rate_eps = 0.05
        zero_cmd_base_lin_vel_eps = 0.05
        zero_cmd_base_ang_vel_eps = 0.05
        class scales(LeggedRobotCfg.rewards.scales):
            action_rate = -0.043#-0.04
            tracking_ang_vel = 2.0
            tracking_lin_vel = 3.0
            lin_vel_z = -1.5
            ang_vel_xy = -0.18#-0.15
            camera_wobble_y = -0.05
            base_height = 1.0#0.8
            orientation = -8.8#-0.80
            feet_air_time = 0.5
            collision = -2.0

            dof_acc=-3.0e-7
            # dof_vel = -2.0e-5

            stand_still = -2.0
            # feet_contact_forces = -0.004
            zero_cmd_feet_contact = -5.0
            zero_cmd_dof_vel = -2.0
            zero_cmd_action_rate = -1.0
            zero_cmd_base_vel = -2.0

            CoT = -0.001
            pass
            #针对六足添加的奖励：
            footend_pos_xy = 0.5 #距离swing_init_point的xy值越近，奖励越高
            # tracking_dof = -0.1 #pos_des和dof_vel距离越大，惩罚越大
            #足端力的增加变化，增加的越快，惩罚越大
            # feet_contact_forces_increase = -0.0005 
            #
            # swing=1.0 #摆动时距离swing_init_point越近，奖励越高，靠近到一定阈值后，距离越远，奖励越高[并不好用]

            # 设计RF与RB；LF和LB的对角奖励
            # mirror=1.0

            #只保留最重要的rew，其余的设为0
            # action_rate=0.0
            # tracking_ang_vel = 1.5
            # tracking_lin_vel = 2.0
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # feet_air_time = 0.0
            # collision = 0.0
            # dof_acc = 0.0
            # stand_still = 0.0
            # feet_contact_forces = 0.0
            # footend_pos_xy = 0.0
            # tracking_dof = -0.0
            #CoT = -0.001
            # torques = 0.0


        only_positive_rewards = False
        tracking_sigma = 0.2
        # tracking_sigma = 0.04
        base_height_target = 0.12
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




class HexGroundCfgPPO(LeggedRobotCfgPPO):

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
        expert_alpha_min=0.0
        expert_alpha_schedule="cosine"

        pass
    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = 'ActorCriticEncoder'
        policy_class_name = 'ActorCritic'
        # algorithm_class_name = 'EGPOEncoder'
        algorithm_class_name = 'EGPO'
        save_interval = 200
        # algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000
        run_name=''
        experiment_name="hex_ground"
        load_run=-1
        expert_path = f"{LEGGED_GYM_ROOT_DIR}/resources/expert_data/bc_actor2.pth"
