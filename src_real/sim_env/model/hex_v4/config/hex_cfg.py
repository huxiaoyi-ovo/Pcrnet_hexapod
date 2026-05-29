import math
class HexCfg:
    def __init__(self,yaml_file:str):
        self.render_mode="human"
        # self.render_mode="headless"
        #set kp and kd to simulate springs in passive joints
        self.foot_end_name="toe"
        # self.max_suck_force=100.0
        self.max_suck_force=100.0
        self.high_body = False #控制身体的高度,设置为True可以方便跨过障碍
    class init_state:
        pos=[0,0,0.3]
        rot=[0.0,0.0,0.0,1.0]
        lin_vel=[0.0,0.0,0.0]
        ang_vel=[0.0,0.0,0.0]
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
        #set foot parallel to robot base
        # default_joint_angles['foot']=-(default_joint_angles['knee']+default_joint_angles['ankle'])-math.pi/2.0
        foot_end_pos=[0.19,0,-0.08]
    class control():
        motor_kp = 100.0
        motor_kd = 1.0
        motor_torque_limits = [-27.0,27.0]
        servo_kp = 20.0
        servo_kd = 0.1
        servo_torque_limits = [-5.0,5.0]

    class link:
        l1=0.072
        l2=0.13
        l3=0.17
    class max_vec:
        x=0.5
        y=0.6
        z=0.8
        omega_adj=0.15
        omega_move=1.8
    class body_shape:
        x=0.1
        y=0.22