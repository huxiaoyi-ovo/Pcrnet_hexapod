#!/usr/bin/python3
import rospy
import sys
from interface.msg import joy_command
from gazebo_msgs.srv import SetModelState, SetPhysicsProperties, GetPhysicsProperties, SetPhysicsPropertiesRequest
from gazebo_msgs.msg import ModelState
def SetPos(client:SetModelState,msg:ModelState):
    response=client(msg)

def SetGravity(set_gravity_client:SetPhysicsProperties,
               gravity_request:SetPhysicsPropertiesRequest):
    global enable_gravity
    if not enable_gravity:
        gravity_request.gravity.z=-9.81
        set_gravity_client(gravity_request)
        enable_gravity=True
    else:
        gravity_request.gravity.z=0
        set_gravity_client(gravity_request)
        enable_gravity=False  
    
def SetInit(pub:rospy.Publisher,msg:joy_command):
    msg.set_init=True
    msg.moving=False
    msg.disable_pump=False
    pub.publish(msg)
def SetMove(pub:rospy.Publisher,msg:joy_command):
    msg.set_init=True
    msg.moving=True
    pub.publish(msg)
def SetVelocity(pub:rospy.Publisher,msg:joy_command,vec_str:str):
    # vx,vy,omega=map(float,input("Enter velocity (vx vy omega): ").split(" "))
    vx=vec_str[0]
    vy=vec_str[1]
    omega=vec_str[2]
    print("vx: ",vx," vy: ",vy," omega: ",omega)
    
    msg.y_vec=0.25 if vy>0.25 else vy
    msg.x_vec=0.2 if vx>0.2 else vx
    msg.w_twist=0.5 if omega>0.5 else omega
    pub.publish(msg)
def SetStop(pub:rospy.Publisher,msg:joy_command):
    msg.stop=True
    pub.publish(msg)
def ReleaseForce(pub:rospy.Publisher,msg:joy_command):
    msg.disable_pump=True
    pub.publish(msg)
def DisableTor(pub:rospy.Publisher,msg:joy_command):
    msg.disable_torque=True
    pub.publish(msg)



if __name__=="__main__":
    rospy.init_node("joy_simu")
    if len(sys.argv)!=2:
        print(">>set running mode by --gazebo / --real, set default to --real<<")
        running_mode='--real'
    else:
        running_mode=sys.argv[1]

    print("------------------->Joy simu node<---------------------")
    if running_mode=="--gazebo":
        print("x: set robot to origin position")
        rospy.wait_for_service("gazebo/set_model_state")
        set_model_client=rospy.ServiceProxy("gazebo/set_model_state",SetModelState)
        model_state_msg=ModelState()
        model_state_msg.model_name=rospy.get_param("robot_name","hexapod_model703")
        model_state_msg.pose.position.x=1.5
        model_state_msg.pose.position.y=-9
        model_state_msg.pose.position.z=2.3
        model_state_msg.pose.orientation.w=1
        
        print("g: set gravity on/off")
        enable_gravity=False
        rospy.wait_for_service("/gazebo/set_physics_properties")
        set_gravity_client=rospy.ServiceProxy("/gazebo/set_physics_properties",SetPhysicsProperties)
        get_gravity_client=rospy.ServiceProxy("/gazebo/get_physics_properties",GetPhysicsProperties)
        physics_properties_msg=get_gravity_client()
        gravity_request=SetPhysicsPropertiesRequest()
        gravity_request.time_step=physics_properties_msg.time_step
        gravity_request.max_update_rate=physics_properties_msg.max_update_rate
        gravity_request.ode_config=physics_properties_msg.ode_config
    
    print("b: set robot initial state")
    print("y: set robot start moving")
    print("r: release the force")
    print("q: set velocity")
    print("a: set robot stop moving")
    print("d: set torque disable")
    
    
    pub=rospy.Publisher("/usr/command",joy_command,queue_size=10)
    

    while not rospy.is_shutdown():
        joy_msg=joy_command()
        key=input("Enter command: ")
        if key=="x":
            SetPos(set_model_client,model_state_msg)
            # pass
        elif key=="g":
            # pass
            SetGravity(set_gravity_client,gravity_request)            
        elif key=="b":
            SetInit(pub,joy_msg)
        elif key=="y":
            SetMove(pub,joy_msg)
        elif key=="r":
            ReleaseForce(pub,joy_msg)
        elif key[0]=='q':
            key=list(map(float,key[2:].split(" ")))
            SetVelocity(pub,joy_msg,key)
        elif key=="a":
            SetStop(pub,joy_msg)
        elif key=="d":
            DisableTor(pub,joy_msg)
        else:
            print("Invalid command")