from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from typing import Tuple
import torch
import os
import math
import random
import numpy as np

from motor_sim import MotorSim
from hex_cfg import HexCfg

#创建仿真环境，设置参数，设置机器人与环境数量

#机器人相关参数

def SetSimParams(sim_params:gymapi.SimParams,args):
    
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 100.0
    sim_params.substeps = 2

    # if args.physics_engine == gymapi.SIM_FLEX:
    #     sim_params.flex.solver_type = 5
    #     sim_params.flex.num_outer_iterations = 4
    #     sim_params.flex.num_inner_iterations = 15
    #     sim_params.flex.relaxation = 0.75
    #     sim_params.flex.warm_start = 0.8
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        # sim_params.physx.max_gpu_contact_pairs=64
        sim_params.physx.contact_offset=0.001
        # sim_params.physx.rest_offset=0.0
        # sim_params.physx.bounce_threshold_velocity=0.5
        # sim_params.physx.default_buffer_size_multiplier=5
        # sim_params.physx.contact_collection=gymapi.ContactCollection.CC_ALL_SUBSTEPS

        # print("PhysX num threads:{},use_gpu:{}".format(args.num_threads,args.use_gpu))
    else:
        raise Exception("Can only use PhysX engine.")
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if args.use_gpu_pipeline:
        print("Use GPU pipline")
    


   


class HexEnv:
    def __init__(self,cfg:HexCfg,env_nums:int,device):


        #Create sim
        self.cfg=cfg
        self.env_nums=env_nums
        self.device=device

        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)

        self.gym=gymapi.acquire_gym()
        self.sim_params=gymapi.SimParams()
        args=gymutil.parse_arguments(description="Test Gym Environment")

        SetSimParams(self.sim_params,args)  

        self.sim=self.gym.create_sim(args.compute_device_id,args.graphics_device_id,args.physics_engine,self.sim_params)
    

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        if self.cfg.render_mode=="human":
            self.viewer=self.gym.create_viewer(self.sim,gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()

        plane_params=gymapi.PlaneParams()
        plane_params.normal=gymapi.Vec3(0,0,1)
        self.gym.add_ground(self.sim,plane_params)

        #Load asset use rosrun command, and os.getcwd()=/home/val/BIH_ws/hex_sim2
        asset_root=os.getcwd()+"/src/sim_env/model/"
        asset_file="hex_v4/urdf/hex_v4_gym.urdf"
        # asset_file="anymal_b/urdf/anymal_b.urdf"

        asset_options=gymapi.AssetOptions()
        asset_options.fix_base_link=False
        # asset_options.fix_base_link=True
        asset_options.flip_visual_attachments=True
        # asset_options.disable_gravity=True
        # asset_options.collapse_fixed_joints=True

        asset_options.armature=0.01
        hex_asset=self.gym.load_asset(self.sim,asset_root,asset_file,asset_options)

        #Get dof names
        self.dof_names=self.gym.get_asset_dof_names(hex_asset)
        self.dof_nums=self.gym.get_asset_dof_count(hex_asset)
        print("dof_names:",self.dof_names)
        #Set dof properties
        hex_dof_props=self.gym.get_asset_dof_properties(hex_asset)
        self._ProcessDofsProps(hex_dof_props)

        #Get rigid body names, find foot indexs in the asset
        self.rb_names=self.gym.get_asset_rigid_body_names(hex_asset)
        self.rb_nums=self.gym.get_asset_rigid_body_count(hex_asset)
        self.rb_foot_end_indexs=torch.tensor([i for i, rb_name in enumerate(self.rb_names) if cfg.foot_end_name in rb_name],
                                             dtype=torch.int32,device=device)
        print("rb_names:",self.rb_names)
        #Get asset rigid body properties
        rb_props=self.gym.get_asset_rigid_shape_properties(hex_asset)
        #Set random friction
        for i in range(len(rb_props)):
            rb_props[i].friction=np.random.uniform(0.6,0.8)
            rb_props[i].filter=i
        self.gym.set_asset_rigid_shape_properties(hex_asset,rb_props)

        #set env grid
        num_per_row=int(math.sqrt(env_nums))
        spacing = 0.6
        env_lower=gymapi.Vec3(-spacing,-spacing,0.0)
        env_upper=gymapi.Vec3(spacing,spacing,spacing)
        # default hex pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(self.cfg.init_state.pos[0],self.cfg.init_state.pos[1],self.cfg.init_state.pos[2])
        pose.r = gymapi.Quat(0.0, 0.0, 0.0,1.0)

        self.envs=[]
        self.hex_handles=[]
        for i in range(self.env_nums):
            #create env
            env=self.gym.create_env(self.sim,env_lower,env_upper,num_per_row)

            #add hex 1: not self collision  0: self collision
            hex_handle=self.gym.create_actor(env,hex_asset,pose,None,i,0)
            self.gym.set_actor_dof_properties(env,hex_handle,hex_dof_props)
            # rs_props=self.gym.get_actor_rigid_shape_properties(env,hex_handle)
            # for j in range(len(rs_props)):
                # print(self.rb_names[j],"filter:",rs_props[j].filter)
                # rs_props[j].thickness=0.01
                # rs_props[j].contact_offset=0.01
                # rs_props[j].filter=j
                # rs_props[j].compliance=0.01
                # rs_props[j].friction=0.8

            # self.gym.set_actor_rigid_shape_properties(env,hex_handle,rs_props)

            # rs_props=self.gym.get_actor_rigid_shape_properties(env,hex_handle)
            # for j in range(len(rs_props)):
            #     print(self.rb_names[j],"filter:",rs_props[j].filter)

            self.envs.append(env)
            self.hex_handles.append(hex_handle)

        # Set camera
        cam_pos=gymapi.Vec3(-1,-1,3)
        cam_target=gymapi.Vec3(1,1,0)
        if self.cfg.render_mode=="human":
            self.gym.viewer_camera_look_at(self.viewer,None,cam_pos,cam_target)  

        # Set keyboard callback
        if cfg.render_mode=="human":
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_F, "apply_force")
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_P, "pause")

        #set motor
        # self.motor=MotorSim('actual',cfg,device=self.device)
        self.motor=MotorSim('ideal',cfg,device=self.device)

        self.gym.prepare_sim(self.sim)
        #init buffers
        self._InitBuffers()

    def _InitBuffers(self):
        #initialize tensors to store dof states, root states, etc.
        #init root state pos+orientation+linear vec+angular vec
        self.root_init_state=torch.tensor(self.cfg.init_state.pos+self.cfg.init_state.rot+self.cfg.init_state.lin_vel+self.cfg.init_state.ang_vel,
                                          dtype=torch.float32,device=self.device)
        self._root_states=self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states=gymtorch.wrap_tensor(self._root_states)
        self.last_root_lin_v=torch.zeros(self.env_nums,3,dtype=torch.float32,device=self.device)
        self.gravity_vec=torch.tensor([0.0,0.0,-1.0],device=self.device).repeat(self.env_nums,1) # env_nums*3 gravity vector
        self.gym.refresh_actor_root_state_tensor(self.sim)

        #init dof default state 
        self.dof_init_pos=torch.zeros(self.dof_nums,dtype=torch.float32,device=self.device)
        self.dof_init_vel=torch.zeros(self.dof_nums,dtype=torch.float32,device=self.device)
        self._SetDofInit(self.dof_names)

        self._dof_states=self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states=gymtorch.wrap_tensor(self._dof_states)
        self.dof_pos=self.dof_states.view(self.env_nums,self.dof_nums,2)[...,0]
        self.dof_vel=self.dof_states.view(self.env_nums,self.dof_nums,2)[...,1]
        self.gym.refresh_dof_state_tensor(self.sim)

        #init rigid body state tensor
        # self._rb_states=self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self.rb_states=gymtorch.wrap_tensor(self._rb_states)
        # self.rb_poses=self.rb_states.view(self.env_nums,-1,13)[...,0:3]
        # self.rb_quats=self.rb_states.view(self.env_nums,-1,13)[...,3:7]

        #init contact force tensor
        # 获取净接触力张量
        self._net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self._net_contact_forces).view(self.env_nums, -1, 3) # type:torch.Tensor
        
        #init suction force tensor
        self.force_ts_change=40.0 #change of suction force every time step
        # self.force_ts_change=0.5 #change of suction force every time step
        self.rb_forces=torch.zeros(self.env_nums,self.rb_nums,3,dtype=torch.float32,device=self.device)
        self.rb_forces_z=self.rb_forces.view(self.env_nums,self.rb_nums,3)[...,2]
        self.rb_force_pos=torch.zeros(self.env_nums,self.rb_nums,3,dtype=torch.float32,device=self.device)
        #acquire z component
        self.suction_force_z=torch.zeros(self.env_nums,len(self.rb_foot_end_indexs),dtype=torch.float32,device=self.device)
        #set suction force [0,0,-200] at pos [0,0,-0.2] in suction cup's local space
        # self.rb_force_pos[:,self.rb_foot_end_indexs,2]=-0.02

        #buffer dof torques
        self.torques=torch.zeros(self.env_nums,self.dof_nums,dtype=torch.float32,device=self.device)

        #buffer last command: joint pos, adhesions
        self.last_command_dof_pos=torch.zeros(self.env_nums,self.dof_nums,dtype=torch.float32,device=self.device)
        self.last_command_adhesions=torch.zeros(self.env_nums,6,dtype=torch.bool,device=self.device)
    
    def _ProcessDofsProps(self,dof_props):
            dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            dof_props['stiffness'].fill(0.0)
            dof_props['damping'].fill(0.0)
            # for i,dof_name in enumerate(self.dof_names):
            #     dof_name_split=dof_name.split("_")
            #     if dof_name_split[2]=='thigh':
            #         if dof_name_split[1]=='rf' or dof_name_split[1]=='lb':
            #             dof_props[i]['lower']=-math.pi*2.0/9.0
            #             dof_props[i]['upper']=math.pi/2.0
            #         elif dof_name_split[1]=='rb' or dof_name_split[1]=='lf':
            #             dof_props[i]['lower']=-math.pi/2.0
            #             dof_props[i]['upper']=math.pi*2.0/9.0
            #         elif dof_name_split[1]=='rm' or dof_name_split[1]=='lm':
            #             dof_props[i]['lower']=-math.pi*2.0/9.0
            #             dof_props[i]['upper']=math.pi*2.0/9.0

    def _SetDofInit(self,dof_names):
        pass
        #set custom dof init state
        #set index of joint direction
        for i,dof_name in enumerate(dof_names):
            if dof_name in self.cfg.init_state.default_joint_angles.keys():
                self.dof_init_pos[i]=self.cfg.init_state.default_joint_angles[dof_name]
            else:
                print("Can not find default angle for dof:{}, set to 0.0 ".format(dof_name))
                self.dof_init_pos[i]=0.0
            # print("dof_name:",dof_name)
            # dof_name_split=dof_name.split("_") #['j','rm','thigh'] ['j','lb','suck']
            # # print("dof_name_split[2]:",dof_name_split[2])
            # if dof_name_split[2] in self.cfg.init_state.default_joint_angles.keys():
            #     self.dof_init_pos[i]=self.cfg.init_state.default_joint_angles[dof_name_split[2]]
            # else:
            #     print("Can not find default angle for dof:{}, set to 0.0 ".format(dof_name))
            #     self.dof_init_pos[i]=0.0
        # print("dof_init_pos:\n",self.dof_init_pos)


    def ResetIdx(self,env_ids:torch.Tensor):
        #set root position
        self.root_states[env_ids]=self.root_init_state
        #set random root pose and velocity
        env_ids_int32=env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),len(env_ids_int32))
        #set joint position
        
        # self.dof_pos[env_ids]=self.dof_init_pos+torch.zeros(len(env_ids),self.dof_nums,dtype=torch.float32,device=self.device)
        self.dof_pos[env_ids]=self.dof_init_pos
        self.dof_vel[env_ids]=0.0
        # print("dof_pos:\n",self.dof_pos)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32),len(env_ids_int32))
        pass


    def GetJointStates(self):
        """
        return joint pos, vel, torques
        """
        return self.dof_pos, self.dof_vel, self.torques
    
    def GetRootStates(self)->Tuple[torch.Tensor,torch.Tensor]:
        """
        return [env_nums*(4+3+3+3), env_nums*3] [quat,angular_vel,acc] [linear_vec,garvity_vec]
        """
        # pos quat is relative to the every env coordinate system
        # vec are represented in env coordinate, we need to transform them to robot coordinate
        root_quat=self.root_states[:,3:7]
        root_lin_v=self.root_states[:,7:10]
        root_ang_v=self.root_states[:,10:13]
        root_acc=(root_lin_v-self.last_root_lin_v)/self.sim_params.dt + torch.randn_like(root_lin_v)*0.5
        root_acc=torch.clamp(root_acc-self.gravity_vec,min=-16.0*9.8,max=16.0*9.8)

        lin_v=quat_rotate_inverse(root_quat,root_lin_v)
        ang_v=quat_rotate_inverse(root_quat,root_ang_v)
        gravity_vec=quat_rotate_inverse(root_quat,self.gravity_vec)
        acc=quat_rotate_inverse(root_quat,root_acc)
        base_state=torch.cat([root_quat,ang_v,acc],dim=1)
        priv_state=torch.cat([lin_v,gravity_vec],dim=1)
        self.last_root_lin_v=root_lin_v
        # print("root_states.shape:",root_states.shape)
        # print("root_states:\n",root_states)
        return base_state, priv_state

    def GetFootContacts(self)->torch.Tensor:
        return self.contact_forces[:,self.rb_foot_end_indexs,2]

    def SetAdhesions(self,adhesions:torch.Tensor)->torch.Tensor:
        """@input:adhesions: env_nums*6 bool @output:abs(suction force z) env_nums*6 """

        #set force at target positions adhesions shapes:env_nums*6
        #set suction force [0,0,-200] at pos [0,0,-0.2] in suction cup's local space
        # contact_mask=self.contact_forces.norm(dim=2,p=1)!=0.0
        contact_mask=self.contact_forces[:,self.rb_foot_end_indexs,2]>0.5
        
        need_adsorb_tuple = torch.where((adhesions*contact_mask)>0.8)
        need_release_tuple = torch.where((adhesions*contact_mask)==0.0)
        self.rb_forces[need_adsorb_tuple[0],self.rb_foot_end_indexs[need_adsorb_tuple[1]],2] -= self.force_ts_change
        self.rb_forces[need_release_tuple[0],self.rb_foot_end_indexs[need_release_tuple[1]],2] += self.force_ts_change
        self.rb_forces = self.rb_forces.clamp(min=-self.cfg.max_suck_force,max=0)

        # need_suck_tuple=torch.where(adhesions) #[env_index,index of rb_foot_end_indexs]
        # need_release_tuple=torch.where(~adhesions) #[env_index,index of rb_foot_end_indexs]
        # apply_force_mask=torch.zeros(self.env_nums,self.rb_nums,dtype=torch.bool,device=self.device)
        # cancle_force_mask=torch.zeros(self.env_nums,self.rb_nums,dtype=torch.bool,device=self.device)

        # apply_force_mask[need_suck_tuple[0],self.rb_foot_end_indexs[need_suck_tuple[1]]]=1
        # apply_force_mask=apply_force_mask&contact_mask
        # continue_apply_force_mask=torch.abs(self.rb_forces_z)<self.cfg.max_suck_force
        # self.rb_forces_z[apply_force_mask&continue_apply_force_mask]-=self.force_ts_change+(random.random()-0.5)*10
        # # self.rb_forces_z[apply_force_mask&continue_apply_force_mask]-=self.force_ts_change+(random.random()-0.5)*0.2
        # set_max_mask=torch.abs(self.rb_forces_z)>self.cfg.max_suck_force
        # self.rb_forces_z[set_max_mask]=-self.cfg.max_suck_force
        # cancle_force_mask[need_release_tuple[0],self.rb_foot_end_indexs[need_release_tuple[1]]]=1
        # continue_cancle_force_mask=torch.abs(self.rb_forces_z)>0
        # self.rb_forces_z[cancle_force_mask&continue_cancle_force_mask]+=self.force_ts_change+(random.random()-0.5)*10
        # # self.rb_forces_z[cancle_force_mask&continue_cancle_force_mask]+=self.force_ts_change+(random.random()-0.5)*0.2
        # set_zero_mask=self.rb_forces_z>0
        # self.rb_forces_z[set_zero_mask]=0.0

        self.suction_force_z[:]=self.rb_forces[:,self.rb_foot_end_indexs,2].abs()

        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim,
                                                       gymtorch.unwrap_tensor(self.rb_forces),
                                                       gymtorch.unwrap_tensor(self.rb_force_pos),
                                                       gymapi.LOCAL_SPACE)
        #record as last command
        self.last_command_adhesions.copy_(adhesions)
        return self.suction_force_z

    def SetJointPos(self,positions:torch.Tensor):
        #positions:[env_nums,dof_nums] dof_nums contains 4 active and 3 passive joints
        pos_error=positions-self.dof_pos
        vel_error=-self.dof_vel
        # print("des joint pos:\n",positions)
        # print("pos error:\n",pos_error)

        #calculate torques
        self.motor.get_torque(pos_error,vel_error,self.torques)
        # print("torques:\n",self.torques.reshape(self.env_nums,6,7))
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self.torques))

        # record last command
        self.last_command_dof_pos.copy_(positions)

    def SetJointTorque(self,torques:torch.Tensor):
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(torques))


    def Simulate(self):
        # self.SetJointPos(dof_command_pos)
        # hex_env.SetAdhesions(rb_forces)
        
        self.gym.simulate(self.sim)
        #update 
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        #render graphics
        if self.cfg.render_mode=="human":
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer,self.sim,True)
        
        #after simulate, process results
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 刷新碰撞检测缓冲区
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def step(self,positions:torch.Tensor):
        pos_error=positions-self.dof_pos
        vel_error=-self.dof_vel
        self.motor.get_torque(pos_error,vel_error,self.torques)
        self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        #update 
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        #render graphics
        if self.cfg.render_mode=="human":
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer,self.sim,True)
        
        #after simulate, process results
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 刷新碰撞检测缓冲区
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return 

if __name__=="__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hex_cfg=HexCfg("hex_cfg.yaml")
    env_nums=2
    hex_env=HexEnv(hex_cfg,env_nums,device)

    joint_pos=torch.zeros(env_nums,hex_env.dof_nums,dtype=torch.float32,device=device)
    joint_torques=torch.zeros_like(joint_pos).view(env_nums,6,-1)
    joint_torques[...,0]=4

    # joint_torques[...,1]=0.2
    # joint_torques[:,[0,1,2],[]]=5

    joint_pos_env=joint_pos.view(env_nums,6,-1)
    # joint_pos_env[:]=1
    joint_pos_env[...,1]=0.7
    joint_pos_env[...,2]=-2.14
    # joint_pos_env[:,0,3]=math.pi/2
    # joint_pos_env[...,2]=-1.8
    # joint_pos_env[...,3]=-(joint_pos_env[...,1]+joint_pos_env[...,2])-math.pi/2
    # joint_pos_env[...,4]=1
    # joint_pos_env[...,5]=1
    # i=0

    adhesions=torch.zeros(env_nums,6,dtype=torch.bool,device=device)
    adhesions[:]=True
    for _ in range(4):
        hex_env.ResetIdx(torch.arange(env_nums,dtype=torch.int32,device=device))
    
    steps=0
    while not hex_env.gym.query_viewer_has_closed(hex_env.viewer):

        # for evt in hex_env.gym.query_viewer_action_events(hex_env.viewer):
        #     if evt.action=="reset" and evt.value>0:
        #         hex_env.ResetIdx(torch.arange(0,env_nums,dtype=torch.int32,device=device))
        #     elif evt.action=="apply_force" and evt.value>0:
        #         hex_env.SetAdhesions(adhesions)
        #         print("set adhesions:\n",adhesions)
        
        # hex_env.SetJointPos(joint_pos)
        # hex_env.SetJointTorque(joint_torques)
        # if steps%200==0:
        #     joint_torques[...,0]=pow(-1,steps//200)*4
        # steps+=1
        hex_env.Simulate()
        # print("lf thigh",hex_env.dof_pos[0,0])
        # for i,rb_name in enumerate(hex_env.rb_names):
        #     if rb_name=='l_lb_toe':
        #         print(rb_name,"=",hex_env.contact_forces[0,i,:])
        #         print("\n")

        # print("rigid body net contact force:\n",hex_env.contact_forces[0])
        # print("set join pos:\n",joint_pos_env)
        # print("joint cur\n",hex_env.dof_pos.view(env_nums,6,7))
        # print("\n")



# print("Done")
# hex_env.gym.destroy_viewer(hex_env.viewer)
# hex_env.gym.destroy_sim(hex_env.sim)






