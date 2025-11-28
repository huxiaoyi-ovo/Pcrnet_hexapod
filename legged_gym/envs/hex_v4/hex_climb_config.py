from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


class HexClimbCfg(LeggedRobotCfg):
    class env:
        num_envs = 1
        num_observations = 76
        num_privileged_obs = 12
        num_actions = 30
        send_timeouts = True
        episode_length_s = 20
        env_spacing = 1.0
    
    class terrain:
        mesh_type = "slope"
        static_friction = 0.7
        dynamic_friction = 0.5
        restitution = 0.01
        class scan_dot:
            num=324 #54*6
            resolution=0.05
        # mesh_type = "STL"
        # terrain_stl = f"{LEGGED_GYM_ROOT_DIR}/resources/terrain/connect_walls.STL"
    class commands:
        num_commands = 6 # lin_xyz ang_xyz 
        resampling_time = 10.
        class ranges:
            lin_vel_x = [-0.5,0.5]
            lin_vel_y = [-0.8,0.8]
            lin_vel_z = [-0.3,0.3]
            ang_vel_roll = [-1,1]
            ang_vel_pitch = [-1,1]
            ang_vel_yaw = [-1,1]

    class init_state:
        pos = [0.0, 0.0, 0.42]
        rot=[0.0,0.0,0.0,1.0]
        lin_vel=[0.0,0.0,0.0]
        ang_vel=[0.0,0.0,0.0]
        support_height = 0.08
        _tao=['lb','lf','lm','rb','rf','rm']
        _q_name=['thigh','knee','ankle','foot','ball1','ball2','suck']
        _joint=0.0
        default_joint_angles ={}
        for t in _tao:
            for qn in _q_name:
                if qn == 'thigh':
                    if t == 'rf' or t == 'lb':
                        _joint=0.35
                    elif t == 'lf' or t == 'rb':
                        _joint=-0.35
                    else:
                        _joint=0.0
                elif qn == 'knee':
                    _joint=0.7
                elif qn == 'ankle':
                    _joint=-2.14
                elif qn == 'foot':
                    _joint=-0.13
                else:
                    _joint=0.0
                default_joint_angles['j_'+t+'_' + qn]=_joint
    
    class control:
        motor_kp = 150.0
        motor_kd = 2.0
        motor_torque_limits = [-27.0,27.0]
        servo_kp = 10.0
        servo_kd = 0.1
        servo_torque_limits = [-0.45,0.45]
        action_scale = 0.25
        decimation = 4

        use_actuator_net=True
        actuator_net_file=f"{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/DM4340_24v_1.pth"

        class adhesion:
            max_force = 100.0 # [N]
            delta_force = 10.0 # [N/s] 每秒钟吸附力的变化量
            
    
    class asset:
        file=f"{LEGGED_GYM_ROOT_DIR}/resources/robots/hex_v4/urdf/hex_climb.urdf"
        name="hex_v4"
        foot_name="toe"
        penalize_contacts_on=["thigh","knee","ankle"]
        terminate_after_contacts_on=["body"]
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3

        self_collisions=0
        replace_cylinder_with_capsule=False
        flip_visual_attachments = True

        density= 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.001
        class links:
            l1 = 0.072
            l2 = 0.13
            l3 = 0.17
        class body_shape:
            x = 0.1
            y = 0.22
    
        left_leg_indices=[0,1,2] #左侧腿的索引
        
    
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass=False
        added_mass_range = [-1.,1.]

    class rewards:
        class scales:
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = 0.1
            ang_vel_xy = 0.2
            torques = -0.00001
            dof_vel = -0.00001
            dof_pos_limit = -1.0
            dof_vel_limit = -1.0

            # dof_acc = -0.00001
            collision = -1.0
            feet_air_time = 1.0
            base_align = 1.0 #身体z轴与平面法向量点乘
            feet_align = 1.0 #脚底部与地面法向量点乘
            adsorb_force = 0.01 #吸附力
            support_height = -1.0 #基座高度


        soft_dof_pos_limit=1.0
        soft_dof_vel_limit=1.0
        soft_dof_torque_limit=1.0
        tracking_sigma = 0.25

    class normalization:
        clip_observations=100.
        class obs_scales:
            quat = 1.0
            lin_vel = 2.0
            ang_vel = 0.25
            lin_acc = 0.01
            gravity = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            dof_torque = 0.1
            suction_force = 0.01
            command = 2.0
            contact = 0.01

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            quat = 0.01
            dof_pos = 0.01
            dof_vel = 1.5
            dof_torque = 1.0
            suction_force = 20.0
            lin_vel = 0.1
            ang_vel = 0.2
            lin_acc = 0.5
            gravity = 0.05
            contact = 10
            plane_norm = 0.1

            # height_measurements = 0.1
            
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HexClimbCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name=" "
        experiment_name="climb_hex_v4"
        load_run=-1
    


if __name__ == '__main__':
    print(HexClimbCfg.init_state.default_joint_angles.keys())
    