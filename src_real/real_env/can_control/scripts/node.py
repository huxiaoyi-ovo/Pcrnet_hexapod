import rospy
from DM_CAN import *
from std_msgs.msg import Bool, Float64MultiArray, UInt32
import serial
import threading
import time
import numpy as np
angle_list=[0,0,0,0,0,0,0,0,0]
speed_list=[0,0,0,0,0,0,0,0,0]
Lock=threading.Lock()
class part(IntEnum):
    left = 0
    right = 1

class modelist(IntEnum):
    Disable = -1
    Zero_Torque = 0
    Pos_vel = 1
    Traj_follow = 2

JOINT_LIMITS = {
    "LB": (
        (-0.6981317007977318, 1.5707963267948966),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
    "LF": (
        (-1.5707963267948966, 0.6981317007977318),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
    "LM": (
        (-0.6981317007977318, 0.6981317007977318),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
    "RB": (
        (-1.5707963267948966, 0.6981317007977318),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
    "RF": (
        (-0.6981317007977318, 1.5707963267948966),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
    "RM": (
        (-0.6981317007977318, 0.6981317007977318),
        (-2.0943951023931953, 2.181661564992912),
        (-2.705260340591211, 2.443460952792061),
    ),
}
VALID_MODES = {int(mode) for mode in modelist}
MAX_JOINT_VELOCITY = 5.0
FEEDBACK_PROBE_TIMEOUT_S = 0.6
PEER_READY_TIMEOUT_S = 1.0
FEEDBACK_STALE_TIMEOUT_S = 0.25
FEEDBACK_PROBE_RESEND_S = 0.04

class leglist(IntEnum):
    Front = 1
    Middle = 2
    Back = 3
class target_joint:
    def __init__(self, body_part: int):
        if body_part == body_part.right:

           self.status_cur = "right_sita_cur"

           self.fro_ID = 0x01
           self.mid_ID = 0x04
           self.bac_ID = 0x07
           self.pn_list = [1,-1,1]

        elif body_part == body_part.left:

           self.status_cur = "left_sita_cur"
           self.fro_ID = 0x01
           self.mid_ID = 0x04
           self.bac_ID = 0x07
           self.pn_list = [1,1,-1]

class leg_list:
    def __init__(self, num: int):
        self.motor=[]
        Motor1 = Motor(DM_Motor_Type.DM4340,num,0x10+num)
        Motor2 = Motor(DM_Motor_Type.DM4340,num+0x01,0x11+num)
        Motor3 = Motor(DM_Motor_Type.DM4340,num+0x02,0x12+num)
        self.motor.append(Motor1)
        self.motor.append(Motor2)
        self.motor.append(Motor3)

class rosnode:
    def __init__(self,node_name='leg_control',body_part=part.left,port='/dev/ttyACM0',record_txt=False):
        target_joints = target_joint(body_part)
        self.pn_list = target_joints.pn_list
        rospy.init_node(node_name, anonymous=True)

        self.sita_cur_pub = rospy.Publisher(target_joints.status_cur, Float64MultiArray, queue_size=1)

        print("node init------------")
        self.rate = rospy.Rate(200) # 100hz

        self.fro_leg = leg_list(target_joints.fro_ID)
        self.mid_leg = leg_list(target_joints.mid_ID)
        self.bac_leg = leg_list(target_joints.bac_ID)
        self.leglist_= [self.bac_leg,self.fro_leg,self.mid_leg]
        self.all_motors = tuple(
            motor
            for leg in self.leglist_
            for motor in leg.motor
        )

        self.body_part = body_part
        self.side_name = "left" if body_part == part.left else "right"
        self.peer_name = "right" if body_part == part.left else "left"
        self.leg_names = ("LB", "LF", "LM") if body_part == part.left else ("RB", "RF", "RM")
        self.rec_flag=False
        self.local_serial_fault=False
        self.system_serial_fault=False
        self.side_ready=False
        self.pos_vel_attempt_id=0
        self.peer_ready_attempt_id=0
        self.serial_device = serial.Serial(port, 921600, timeout=0.5)

        #创建用于写入的文件流
        self.record_txt=record_txt
        if self.record_txt:
            self.file_name = target_joints.status_cur.split('_')[0]
            with open('/home/nvidia/BIH_ws/hex_sim2/bag/'+self.file_name+'.txt','w') as file:
                file.close()


        print("serial is opened-----------")
        self.MC=MotorControl(self.serial_device)

        for a in self.fro_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("front leg is opened--------")
        for a in self.mid_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("midle leg is opened--------")
        for a in self.bac_leg.motor:
            self.MC.addMotor(a)
            self.MC.switchControlMode(a,Control_Type.MIT)
            # self.MC.enable(a)
            print("back leg is opened--------")
        self.serial_fault_pub = rospy.Publisher(
            "/can_control/serial_fault",
            Bool,
            queue_size=1,
            latch=True,
        )
        self.serial_fault_sub = rospy.Subscriber(
            "/can_control/serial_fault",
            Bool,
            self.serial_fault_cb,
            queue_size=1,
        )
        self.side_ready_pub = rospy.Publisher(
            "/can_control/%s_ready" % self.side_name,
            UInt32,
            queue_size=1,
            latch=True,
        )
        self.peer_ready_sub = rospy.Subscriber(
            "/can_control/%s_ready" % self.peer_name,
            UInt32,
            self.peer_ready_cb,
            queue_size=1,
        )
        self.side_ready_pub.publish(UInt32(data=0))
        self.sita_des_sub = rospy.Subscriber(
            "/sita_des",
            Float64MultiArray,
            self.sita_des,
            queue_size=1,
        )

    def serial_fault_cb(self, msg):
        if msg.data and not self.system_serial_fault:
            self.system_serial_fault = True
            self.set_side_ready(False)
            rospy.logerr(
                "[%s] shared serial fault latched; only Disable commands remain allowed",
                self.body_part.name,
            )

    def peer_ready_cb(self, msg):
        self.peer_ready_attempt_id = int(msg.data)

    def set_side_ready(self, ready):
        self.side_ready = bool(ready)
        if hasattr(self, "side_ready_pub"):
            ready_attempt_id = self.pos_vel_attempt_id if self.side_ready else 0
            self.side_ready_pub.publish(UInt32(data=ready_attempt_id))

    def latch_command_fault(self, reason):
        if not self.system_serial_fault:
            self.system_serial_fault = True
            self.set_side_ready(False)
            rospy.logfatal("[%s] unsafe sita_des rejected: %s", self.body_part.name, reason)
            self.serial_fault_pub.publish(Bool(data=True))

    def feedback_is_fresh(self):
        now = time.monotonic()
        return all(
            motor.feedback_count > 0
            and now - motor.last_feedback_monotonic <= FEEDBACK_STALE_TIMEOUT_S
            for motor in self.all_motors
        )

    def probe_all_motor_feedback(self):
        self.set_side_ready(False)
        baseline_counts = {
            motor.SlaveID: motor.feedback_count
            for motor in self.all_motors
        }
        deadline = time.monotonic() + FEEDBACK_PROBE_TIMEOUT_S
        next_send = 0.0

        while time.monotonic() < deadline and not self.system_serial_fault:
            now = time.monotonic()
            missing = [
                motor
                for motor in self.all_motors
                if motor.feedback_count <= baseline_counts[motor.SlaveID]
            ]
            if not missing:
                self.set_side_ready(True)
                rospy.loginfo("[%s] all 9 motors returned fresh state feedback", self.side_name)
                return True
            if now >= next_send:
                for motor in missing:
                    self.MC.controlMIT(motor, 0, 0, 0, 0, 0)
                next_send = now + FEEDBACK_PROBE_RESEND_S
            self.MC.recv()
            time.sleep(0.002)

        missing_ids = [
            motor.SlaveID
            for motor in self.all_motors
            if motor.feedback_count <= baseline_counts[motor.SlaveID]
        ]
        self.latch_command_fault(
            "motor feedback probe timed out; missing slave IDs=%s" % missing_ids
        )
        return False

    def wait_for_peer_ready(self):
        deadline = time.monotonic() + PEER_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.system_serial_fault:
                return False
            if (
                self.peer_ready_attempt_id == self.pos_vel_attempt_id
                and self.peer_ready_sub.get_num_connections() > 0
            ):
                return True
            time.sleep(0.005)
        self.latch_command_fault(
            "%s CAN side did not confirm all 9 motor feedback packets" % self.peer_name
        )
        return False

    def prepare_pos_vel(self):
        self.pos_vel_attempt_id += 1
        if not self.probe_all_motor_feedback():
            return False
        if not self.wait_for_peer_ready():
            return False
        if not self.feedback_is_fresh():
            self.latch_command_fault("motor feedback became stale before Pos_vel start")
            return False
        return True

    def leg_is_enable(self,leg_num:leg_list):
        return all(motor.isEnable for motor in leg_num.motor)

    def leg_has_enabled_motor(self,leg_num:leg_list):
        return any(motor.isEnable for motor in leg_num.motor)
            
    def leg_set_enable(self,leg_num:leg_list,ablestatus=True):
        if ablestatus == True:
            for a in leg_num.motor:
                self.MC.enable(a)
                pass
        else:
            for i in leg_num.motor:
                self.MC.disable(i)
                pass

    def leg_set_mode(self,leg_num:leg_list,mode:Control_Type):
        for i in leg_num.motor:
            self.MC.switchControlMode(i,mode)
            time.sleep(1)

            pass
        time.sleep(1)

    def limit_check(self,data:Float64MultiArray):
        limit_data=list(data)
        if len(limit_data) < 19:
            self.latch_command_fault("side command length=%d, expected 19" % len(limit_data))
            limit_data=[modelist.Disable]+[0.0]*18
            return limit_data
        mode = float(limit_data[0])
        if not np.isfinite(mode) or int(mode) not in VALID_MODES or mode != int(mode):
            self.latch_command_fault("invalid motor mode=%r" % limit_data[0])
            return [modelist.Disable]+[0.0]*18

        for leg_idx, leg_name in enumerate(self.leg_names):
            for joint_idx, (low, high) in enumerate(JOINT_LIMITS[leg_name]):
                pos_idx = 1 + leg_idx * 6 + joint_idx
                vel_idx = 4 + leg_idx * 6 + joint_idx
                pos = float(limit_data[pos_idx])
                vel = float(limit_data[vel_idx])
                if not np.isfinite(pos) or not np.isfinite(vel):
                    self.latch_command_fault("non-finite command for %s joint %d" % (leg_name, joint_idx))
                    limit_data=[modelist.Disable]+[0.0]*18
                    return limit_data
                clipped = float(np.clip(pos, low, high))
                if clipped != pos:
                    limit_data[pos_idx] = clipped
                    limit_data[vel_idx] = 0.0
                    rospy.logwarn_throttle(
                        1.0,
                        "[%s] joint command clipped: leg=%s joint=%d %.3f -> %.3f",
                        self.body_part.name,
                        leg_name,
                        joint_idx,
                        pos,
                        clipped,
                    )
                else:
                    limit_data[vel_idx] = float(np.clip(vel, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY))
        return limit_data
    def motor_staus_pub(self,leg_num:leg_list):
        status = []
        n=0
        for i in leg_num.motor:
            status.append(i.getPosition()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            status.append(i.getTorque()*self.pn_list[n])
            n=n+1
        n=0
        for i in leg_num.motor:
            status.append(i.getVelocity()*self.pn_list[n])
            n=n+1
        return status

    def return_position(self):
        des_poses=[]
        for leg in self.leglist_:
            for t in range(3):
                des_poses.append(leg.motor[t].getPosition()*self.pn_list[t])
        return des_poses
    def data2leg(self,data:float):
        # print("IN DATA 2 LEG")
        if data[0]==modelist.Disable:
            self.set_side_ready(False)
            for leg in self.leglist_:
                if self.leg_has_enabled_motor(leg):
                    self.leg_set_enable(leg,False)
                else :
                    pass
            return False
        else:
            for leg in self.leglist_:
                if not self.leg_is_enable(leg):
                    self.leg_set_enable(leg,True)
        
            if data[0]==modelist.Zero_Torque:
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)
                        pass
                for leg in self.leglist_:
                    for count_ in range(3):
                        self.MC.controlMIT(leg.motor[count_], 0, 0, 0, 0, 0)
                return False

            elif data[0]==modelist.Pos_vel:
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)

                if not self.prepare_pos_vel():
                    return False
                des_poses = np.array(list(data[1:4]) + list(data[7:10]) + list(data[13:16]),dtype=np.float64)
                cur_poses = np.array(self.return_position(), dtype=np.float64)
                distance = np.linalg.norm(cur_poses-des_poses)
                dis_norm = (des_poses-cur_poses)/distance if distance > 0.0 else np.zeros_like(des_poses)
                reach_target=False
                while not reach_target:
                    if self.system_serial_fault:
                        return False
                    if np.linalg.norm(cur_poses-des_poses)>0.02:
                        cur_poses=dis_norm*0.02+cur_poses
                    else:
                        cur_poses=des_poses
                        reach_target=True

                    for leg_idx, leg in enumerate(self.leglist_):
                        for joint_idx in range(3):
                            if self.system_serial_fault:
                                return False
                            idx = leg_idx * 3 + joint_idx
                            target = cur_poses[idx] * self.pn_list[joint_idx]
                            self.MC.controlMIT(leg.motor[joint_idx], 120, 1.0, target, 0, 0)
                            self.MC.recv()
                            time.sleep(0.001)
                    if not self.feedback_is_fresh():
                        self.latch_command_fault("motor feedback became stale during Pos_vel")
                        return False
                return True

            elif data[0]==modelist.Traj_follow:
                if (
                    not self.side_ready
                    or self.peer_ready_attempt_id != self.pos_vel_attempt_id
                    or self.peer_ready_sub.get_num_connections() <= 0
                ):
                    self.latch_command_fault("Traj_follow requested before both CAN sides were ready")
                    return False
                if self.fro_leg.motor[0].NowControlMode!=Control_Type.MIT:
                    for leg in self.leglist_:
                        self.leg_set_mode(leg,Control_Type.MIT)
                        pass
                for leg_idx, leg in enumerate(self.leglist_):
                    for i in range(3):
                        self.MC.controlMIT(leg.motor[i],100,1.0,data[i+1+leg_idx*6]*self.pn_list[i],data[i+4+leg_idx*6],0)
                        self.MC.recv()
                        sleep(0.00001)
                self.MC.recv()
                if not self.feedback_is_fresh():
                    self.latch_command_fault("motor feedback became stale during Traj_follow")
                    return False
                return True
            else:
                self.latch_command_fault("unsupported motor mode=%r" % data[0])
                return False

                
                



    
    def sita_des(self,data):
        with Lock:
            if self.local_serial_fault:
                return
            try:
                if len(data.data) < 43:
                    self.latch_command_fault("full command length=%d, expected 43" % len(data.data))
                    self.data2leg([modelist.Disable]+[0.0]*18)
                    return
                if self.system_serial_fault and data.data[0] != modelist.Disable:
                    return
                status_data=Float64MultiArray()
                des_data=[]
                set_data=[0.0]*19
                set_data[0]=data.data[0]
                set_data[1:7]=data.data[1+self.body_part*21:7+self.body_part*21]
                set_data[7:13]=data.data[8+self.body_part*21:14+self.body_part*21]
                set_data[13:19]=data.data[15+self.body_part*21:21+self.body_part*21]

                set_data=self.limit_check(set_data)
                state_valid = self.data2leg(set_data)
                if not state_valid:
                    return
                if not self.feedback_is_fresh():
                    self.latch_command_fault("refusing to publish stale motor state")
                    return
                # 扁平化拼接状态并发布
                status_flat = []
                status_flat.extend(self.motor_staus_pub(self.bac_leg))
                status_flat.extend(self.motor_staus_pub(self.fro_leg))
                status_flat.extend(self.motor_staus_pub(self.mid_leg))
                status_data.data = status_flat
                self.sita_cur_pub.publish(status_data)
            except (serial.SerialException, OSError) as exc:
                self.local_serial_fault = True
                self.system_serial_fault = True
                self.set_side_ready(False)
                rospy.logfatal(
                    "[%s] serial I/O fault; all motion commands are latched off until control nodes restart: %s",
                    self.body_part.name,
                    exc,
                )
                self.serial_fault_pub.publish(Bool(data=True))
                try:
                    self.serial_device.close()
                except Exception:
                    pass

            #将期望值和当前值写入一个文件中
            # if self.record_txt:
            #     if data.data[0]==modelist.Traj_follow:
            #         with open('/home/nvidia/BIH_ws/hex_sim2/bag/'+self.file_name+'.txt','a') as file:
            #             file.write(f"des={des_data};cur={status_flat}\n")
            #             file.close()





    def send_data(self):
        if self.fro_leg.motor[0].isEnable and self.fro_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.fro_leg.motor[i], 100, 1, angle_list[i], speed_list[i], 0)
                self.MC.recv()
                sleep(0.0002)
        if self.mid_leg.motor[0].isEnable and self.mid_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.mid_leg.motor[i], 100, 1, angle_list[i+3], speed_list[i+3], 0)
                self.MC.recv()
                sleep(0.0002)
        if self.bac_leg.motor[0].isEnable and self.bac_leg.motor[0].NowControlMode==Control_Type.MIT and self.rec_flag:
            for i in range(3):
                self.MC.controlMIT(self.bac_leg.motor[i], 100, 1, angle_list[i+6], speed_list[i+6], 0)
                self.MC.recv()
                sleep(0.0002)
        # print("in here----------",selaaf.rec_flag)


            
        pass


    def run(self):
        rospy.spin()
