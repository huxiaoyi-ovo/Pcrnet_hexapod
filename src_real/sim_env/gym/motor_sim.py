#电机模拟器 输入位置速度误差，输出电机扭矩
#分为理想模型和实际模型
import torch
from torch import nn
import numpy as np
from hex_cfg import HexCfg

class MotorSim:
    def __init__(self,sim_type,hex_cfg:HexCfg,device):
        self.sim_type=sim_type
        self.device=device
        self.cfg=hex_cfg

        self.dof_names=self.cfg.init_state.default_joint_angles.keys()
        self.kp_kd_tensor=torch.zeros(2,len(self.dof_names),device=self.device)
        motor_name=['thigh','knee','ankle']
        _motor_index=[]
        _servo_index=[]
        _passive_index=[]
        _torques_limits=[]
        for i, dof_name in enumerate(self.dof_names):
            split_name=dof_name.split("_")
            if split_name[2] in motor_name:
                _motor_index.append(i)
                self.kp_kd_tensor[0,i]=self.cfg.control.motor_kp
                self.kp_kd_tensor[1,i]=self.cfg.control.motor_kd
                _torques_limits.append(self.cfg.control.motor_torque_limits)
            elif split_name[2] == 'foot':
                _servo_index.append(i)
                self.kp_kd_tensor[0,i]=self.cfg.control.servo_kp
                self.kp_kd_tensor[1,i]=self.cfg.control.servo_kd
                _torques_limits.append(self.cfg.control.servo_torque_limits)
            else:#for passive joints
                _passive_index.append(i)
                self.kp_kd_tensor[0,i]=0.5
                self.kp_kd_tensor[1,i]=0.01
                _torques_limits.append([-0.2,0.2])
        self.motor_index=torch.tensor(_motor_index,dtype=torch.int32,device=self.device)
        self.servo_index=torch.tensor(_servo_index,dtype=torch.int32,device=self.device)
        self.passive_index=torch.tensor(_passive_index,dtype=torch.int32,device=self.device)
        self.torques_limits=torch.tensor(_torques_limits,dtype=torch.float32,device=self.device) #dof_nums*2

        motor_hidden_dims=[32,32]
        motor_layers = []
        motor_layers.append(nn.Linear(6,motor_hidden_dims[0]))
        motor_layers.append(nn.ReLU())
        motor_layers.append(nn.Linear(motor_hidden_dims[0],motor_hidden_dims[1]))
        motor_layers.append(nn.ReLU())
        motor_layers.append(nn.Linear(motor_hidden_dims[1],3))
        self.motor_net = nn.Sequential(*motor_layers).to(self.device)
        if self.sim_type=='actual':
            self.motor_net.load_state_dict(torch.load('/home/val/BIH_ws/hex_sim2/src/sim_env/gym/DM4340_24v.pth',weights_only=True))
            self.motor_net.eval()
        
    def get_torque(self,pos_err,vel_err,torques):
        if self.sim_type=='ideal':
            self.ideal_torque(pos_err,vel_err,torques)
        elif self.sim_type=='actual':
            self.actual_torque(pos_err,vel_err,torques)
        else:
            print('--------->MotorSim: Invalid sim_type<-------------')

    def ideal_torque(self,pos_err:torch.Tensor,vel_err:torch.Tensor,torques:torch.Tensor):
        torques[:]=self.kp_kd_tensor[0,:]*pos_err+self.kp_kd_tensor[1,:]*vel_err #+ (torch.rand_like(torques)-0.5)*2.0
        torch.clamp_(torques,self.torques_limits[:,0],self.torques_limits[:,1])
        # print("MotorSim: ideal torque\n",torques)
    def actual_torque(self,pos_err,vel_err,torques):
        with torch.no_grad():
            motor_pos_err = pos_err[:,self.motor_index].reshape(-1,3)
            motor_vel_err = vel_err[:,self.motor_index].reshape(-1,3)
            servo_pos_err = pos_err[:,self.servo_index]
            servo_vel_err = vel_err[:,self.servo_index]
            passive_pos_err = pos_err[:,self.passive_index]
            passive_vel_err = vel_err[:,self.passive_index]
            err=torch.torch.cat([motor_pos_err,motor_vel_err],dim=1)
            # print("err,shape=",err.shape)
            motor_torques=self.motor_net(err).reshape(-1,len(self.motor_index)) #type: torch.Tensor
            servo_torques=self.kp_kd_tensor[0,self.servo_index]*servo_pos_err+self.kp_kd_tensor[1,self.servo_index]*servo_vel_err #type: torch.Tensor
            passive_torques=self.kp_kd_tensor[0,self.passive_index]*passive_pos_err+self.kp_kd_tensor[1,self.passive_index]*passive_vel_err #type: torch.Tensor
            torques[:,self.motor_index]=motor_torques
            torques[:,self.servo_index]=servo_torques
            # torques[:,self.passive_index]=passive_torques
            torques[:,self.passive_index]=0.0
        torch.clamp_(torques,self.torques_limits[:,0],self.torques_limits[:,1])
        # print("motor net predict torques:\n",torques)

