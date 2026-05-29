import numpy as np
import time
import torch
import math
from hex_cfg import HexCfg
from usr_command import Command
from kinematic import Kinematic

class CurvAdapt:
    def __init__(self,robot_config_yaml,device,
                 joint_pos_des:torch.Tensor,
                 joint_pos_cur:torch.Tensor,
                 joint_torques_cur:torch.Tensor,
                 adhesions:torch.Tensor,
                 suction_forces:torch.Tensor):
        self.device=device
        # init params and kinematic utils
        self.cfg = HexCfg(robot_config_yaml)
        self.kin=Kinematic(self.cfg.link.l1,self.cfg.link.l2,self.cfg.link.l3,device)

        # init curv variables
        self.leg_names=["LB","LF","LM","RB","RF","RM"] #the order is set accroding to dof_names in sim_env
        self.reset=False
        self.set_init_done=0 # within range count 1 time, stay within range for 0.5s 50counts
        self.swing_reach_high=False # swing leg retrace enough high
        self.gaits=torch.zeros(6,dtype=torch.bool,device=device) # 1 for stance, 0 for swing
        self.A_group_index=torch.tensor([0,1,5],dtype=torch.int32,device=device) # index tripod group
        self.B_group_index=torch.tensor([2,3,4],dtype=torch.int32,device=device) # index tripod group
        self.gaits_groups=[self.A_group_index,self.B_group_index]
        self.stance_group_index=0
        self.B_e_des=torch.ones(6,4,dtype=torch.float32,device=device)#[x,y,z,1]
        self.B_e_cur=torch.ones(6,4,dtype=torch.float32,device=device)#[x,y,z,1]
        self.q_init=torch.zeros(6,4,dtype=torch.float32,device=device)#[thigh knee ankle foot]
        self.R1_T_R=torch.eye(4,4,dtype=torch.float32,device=device) # R1 is the next body pose
        self.R1_T_R_swing=torch.eye(4,4,dtype=torch.float32,device=device) # R1 is the next body pose
        self.body_shape=torch.zeros(6,4,dtype=torch.float32,device=device) # body shape


        #init outer tensors
        self.q_des=joint_pos_des # 6*4 float32 tensor
        self.q_cur=joint_pos_cur # 6*3 float32 tensor
        self.torques_cur=joint_torques_cur # 6*3 float32 tensor
        self.adhesions=adhesions # 6 bool tensor
        self.suction_forces=suction_forces # 6 float32 tensor

        #set body shape initial
        self.body_shape[0:3,0]=-self.cfg.body_shape.x
        self.body_shape[3:6,0]=self.cfg.body_shape.x
        self.body_shape[[0,3],1]=-self.cfg.body_shape.y
        self.body_shape[[1,4],1]=self.cfg.body_shape.y

        #set initial foot end pos joint pos
        q_init=[0,0.7,-2.14]
        for i in range(3): 
            self.B_e_des[:,i]=self.cfg.init_state.foot_end_pos[i]
            self.q_init[:,i]=q_init[i]
        #set corner positions
        self.q_init[[1,3],0]=-math.pi/9.0
        self.q_init[[0,4],0]=math.pi/9.0
        # self.kin.InverseKin1(self.B_e_des[:,0:3],self.q_init[:,0:3])
        self.kin.ForwardKin(self.q_init,self.B_e_des)
        self._GetFootAngle(self.q_init)
        self.q_des.copy_(self.q_init)
        
        # print("-------------initial B_e_des-------------\n",self.B_e_des)



    def ProcessCommand(self,command:Command):
        #update current position of foot end in leg base frame
        self.kin.ForwardKin(self.q_cur,self.B_e_cur)

        if command.set_init and self.set_init_done<=100:
            #set initial joints pos and foot end pos
            self.q_des.copy_(self.q_init)
            self.kin.ForwardKin(self.q_des,self.B_e_des)
            if torch.all(torch.norm((self.B_e_des-self.B_e_cur),p=2,dim=1)<0.05):
                self.set_init_done+=1
                #set initial gaits LB LF RM set to swing
                self.stance_group_index=0 #set A group stance, B group is swing 
                self.gaits[self.gaits_groups[self.stance_group_index]]=1
            #at begnning, robot position is not stable
            if self.set_init_done>=80:
                self.adhesions[self.gaits_groups[self.stance_group_index]]=1

        if self.set_init_done>100:
            if any(command.vec) or command.omega:
                self.GaitPlanning(command)
                self.CalJointPoses()

        return self.reset



    def GaitPlanning(self,command:Command):
        # print(">>>>>>>>>>>>>GaitPlanning<<<<<<<<<<<<")
        #calculate Transform of robot body
        self._TargetTransInterp(command)
        #calculate next B_e_des
        next_B_e_des=self._CalB_e_des(command)
        #judeg if exceed working range
        range_feasi=self._FeasiCheck(next_B_e_des) #
        collide_feasi=self._CollideCheck(next_B_e_des) #
        # print("collide_feasi:\n",collide_feasi)

        stance_done=False
        swing_done=False


        #for swing leg, set 
        #select contact done
        stance_index=torch.where(self.gaits)[0]
        swing_index=torch.where(~self.gaits)[0]

        # swing_done_mask=~self.gaits&self._ContactDetection()
        swing_done_mask=self._ContactDetection()[~self.gaits] # 3 bool, corresponding to swing_index
    
        # for contact not done
        if (~swing_done_mask).any():
            # for those out of range or collide with other in next frame, only set z 
            set_z_mask=(~collide_feasi|~range_feasi)&(~self.gaits)
            set_all_mask=(collide_feasi&range_feasi)&(~self.gaits)
            # self.B_e_des[set_z_mask,2]=next_B_e_des[set_z_mask,2]
            self.B_e_des[...,2][set_z_mask]=next_B_e_des[...,2][set_z_mask]
            # for rest, set all
            self.B_e_des[set_all_mask,:]=next_B_e_des[set_all_mask,:]

        if swing_done_mask.any():
            #set adhesions to 1
            self.adhesions[swing_index[swing_done_mask]]=1
            if swing_done_mask.sum() == 3 and self.swing_reach_high:
                swing_done=True

        if swing_done_mask.any() and self.swing_reach_high:
            stance_done=True

        #for stance leg every range_feasi and collide_feasi, set B_e_des
        stance_mask=self.gaits
        if not stance_done:
            if all(range_feasi[stance_mask]&collide_feasi[stance_mask]):
                self.B_e_des[stance_mask,:]=next_B_e_des[stance_mask,:]
            else:
                stance_done=True

        # if swing done, begin adsorb
        if swing_done:
            if self._AdsorbReleaseDetection()[swing_index[swing_done_mask]].all():
                self.adhesions[stance_mask]=0
        
            
        # if release done, switch gait, clear start flag
        if stance_done:
            if self._AdsorbReleaseDetection()[stance_mask].all():
                self.stance_group_index=1-self.stance_group_index
                self.gaits.zero_()
                self.gaits[self.gaits_groups[self.stance_group_index]]=1
                self.swing_reach_high=False
                
        # gaits_names=["swing","stance"]
        # print("gaits:\n",[gaits_names[i] for i in self.gaits.tolist()])
        # print("swing_done_mask:",swing_done_mask.tolist())
        # print("swing_set_index",swing_set_index,"set_z_index=",set_z_index)
        # print("stance_done=",stance_done,"swing_done=",swing_done)

        #set adhesions

    def CalJointPoses(self):
        damp_inv_jac=self.kin.DampInvJac(self.q_cur) #6*3*3
        # print("q_cur:\n",self.q_cur)
        # print("B_e_des:\n",self.B_e_des)

        # self.kin.InverseKin1(self.B_e_des[:,0:3],self.q_cur[:,0:3])
        # self.q_des[:,0:3]=self.q_cur.clone()
        # print("self.B_e_des[:,0:3]-self.B_e_cur[:,0:3].T",(self.B_e_des[:,0:3]-self.B_e_cur[:,0:3]).T)
        # print("damp_inv_jac:\n",damp_inv_jac)
        pos_err=(self.B_e_des[:,0:3]-self.B_e_cur[:,0:3]).unsqueeze(-1) # 6*3*1
        self.q_des[:,0:3]=self.q_cur+60*((damp_inv_jac@pos_err).squeeze(-1))*0.01
        self._GetFootAngle(self.q_des)

    def _TargetTransInterp(self,command:Command):
        x=command.vec[0]*0.01
        y=command.vec[1]*0.01
        omega=command.omega*0.01
        R_T_R1_mat=torch.eye(4,4,dtype=torch.float32,device=self.device)
        R_T_R1_mat[0:3,0:3]=torch.tensor([[math.cos(omega),-math.sin(omega),0],
                                       [math.sin(omega),math.cos(omega),0],
                                       [0,0,1]],dtype=torch.float32,device=self.device)
        R_T_R1_mat[0,3]=x
        R_T_R1_mat[1,3]=y
        self.R1_T_R=torch.linalg.inv(R_T_R1_mat)

        x=-x
        y=-y
        omega=-omega
        R_T_R1_mat[0:3,0:3]=torch.tensor([[math.cos(omega),-math.sin(omega),0],
                                       [math.sin(omega),math.cos(omega),0],
                                       [0,0,1]],dtype=torch.float32,device=self.device)
        R_T_R1_mat[0,3]=x
        R_T_R1_mat[1,3]=y
        self.R1_T_R_swing=torch.linalg.inv(R_T_R1_mat)

    def _CalB_e_des(self,command:Command):
        #select swing and stance leg
        swing_index=self.gaits_groups[1-self.stance_group_index]
        stance_index=self.gaits_groups[self.stance_group_index]
        # print(">>>>_CalB_e_des, swing_index={}, stance_index={}".format(swing_index,stance_index))
        #calculate next B_e_des xy 
        R_e_des=self._B2R(self.B_e_des)
        next_R_e_des=torch.zeros_like(R_e_des,dtype=torch.float32,device=self.device)
        next_R_e_des[swing_index,:]=(self.R1_T_R_swing@R_e_des[swing_index,:].T).T
        next_R_e_des[stance_index,:]=(self.R1_T_R@R_e_des[stance_index,:].T).T
        next_B_e_des=self._R2B(next_R_e_des)
        #for swing legs, calculate z vz=0.3m/s
        if not self.swing_reach_high:
            if (next_B_e_des[swing_index,2]>=0.03).any():
                self.swing_reach_high=True
            else:
                next_B_e_des[swing_index,2]+=0.01*self.cfg.max_vec.z
        else:
            next_B_e_des[swing_index,2]-=0.01*self.cfg.max_vec.z
        return next_B_e_des
    
    def _GetFootAngle(self,joint_pos:torch.Tensor):
        joint_pos[:,3]=-(joint_pos[:,1]+joint_pos[:,2])-math.pi/2.0

    def _FeasiCheck(self,B_e_des:torch.Tensor)->torch.Tensor:
        x_feasi=(B_e_des[:,0]>=0.16) & (B_e_des[:,0]<=0.28)
        z_feasi=(B_e_des[:,2]>=-0.14) & (B_e_des[:,2]<=0.1)
        thigh_angle=torch.atan(B_e_des[:,1]/B_e_des[:,0])
        angle_feasi=(thigh_angle>=-math.pi/6.0) & (thigh_angle<=math.pi/6.0)
        return x_feasi&z_feasi&angle_feasi

    def _CollideCheck(self,B_e_des:torch.Tensor)->torch.Tensor:
        # LB LF LM RB RF RM
        # LM xy check if collide with lF xy LB xy
        collide_feasi_flag=torch.zeros(6,dtype=torch.bool,device=self.device)
        #transfer to R frame
        R_e_des=self._B2R(B_e_des)

        collide_feasi_flag[0]=torch.norm(R_e_des[0,0:2]-R_e_des[2,0:2])>0.12
        collide_feasi_flag[1]=torch.norm(R_e_des[1,0:2]-R_e_des[2,0:2])>0.12
        collide_feasi_flag[2]=collide_feasi_flag[0]&collide_feasi_flag[1] #feasible when both feet not collide

        collide_feasi_flag[3]=torch.norm(R_e_des[3,0:2]-R_e_des[5,0:2])>0.12
        collide_feasi_flag[4]=torch.norm(R_e_des[4,0:2]-R_e_des[5,0:2])>0.12
        collide_feasi_flag[5]=collide_feasi_flag[3]&collide_feasi_flag[4]

        # print(">>>>>>>_CollideCheck<<<<<<<<<<,\n, R_e_des:\n",R_e_des)
        # print("collide_feasi_flag:=",collide_feasi_flag)

        return collide_feasi_flag

    def _ContactDetection(self)->torch.Tensor:
        # swing reach the high point and z<-0.08
        return (self.B_e_des[:,2]<-0.07)&(self.swing_reach_high)
    
    def _AdsorbReleaseDetection(self)->torch.Tensor:
        # for stance state judge if release done
        stance_release_done=self.gaits&(self.suction_forces<10.0)
        swing_adsorb_done=~self.gaits&(self.suction_forces>=80.0)
        return stance_release_done|swing_adsorb_done
        # if set adsorb and suction force big enough  or set release and suction force small enough
        # reach_bound_bool=((self.suction_forces>=80.0)&self.adhesions) | ((self.suction_forces<10.0)&~self.adhesions)
        # return reach_bound_bool

    def _R2B(self,R_e:torch.Tensor):
        B_e=R_e.clone()
        B_e[0:3,0:2]=self.body_shape[0:3,0:2]-R_e[0:3,0:2]
        B_e[3:6,0:2]=R_e[3:6,0:2]-self.body_shape[3:6,0:2]
        return B_e

    def _B2R(self,B_e:torch.Tensor):
        R_e=B_e.clone()
        R_e[0:3,0:2]=self.body_shape[0:3,0:2]-B_e[0:3,0:2]
        R_e[3:6,0:2]=self.body_shape[3:6,0:2]+B_e[3:6,0:2]
        return R_e

    

