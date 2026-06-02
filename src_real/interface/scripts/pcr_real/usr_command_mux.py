#!/usr/bin/env python3
"""Select one safe source for src_real /usr/command."""

import argparse
import json
import time
from copy import deepcopy


rospy = None
JoyCommand = None
String = None


def absmax_cmd(msg):
    return max(abs(float(msg.x_vec)), abs(float(msg.y_vec)), abs(float(msg.z_vec)), abs(float(msg.w_twist)))


def make_stand_msg():
    msg = JoyCommand()
    msg.set_init = True
    msg.moving = False
    msg.disable_pump = False
    msg.disable_torque = False
    msg.action_valve = False
    msg.stop = False
    msg.change_mode = False
    msg.x_vec = 0.0
    msg.y_vec = 0.0
    msg.z_vec = 0.0
    msg.w_twist = 0.0
    return msg


def has_manual_priority(msg, xy_deadband, yaw_deadband):
    if bool(msg.disable_pump or msg.disable_torque or msg.action_valve):
        return True
    if bool(msg.set_init or msg.stop or msg.change_mode):
        return True
    if abs(float(msg.x_vec)) > xy_deadband or abs(float(msg.y_vec)) > xy_deadband:
        return True
    if abs(float(msg.z_vec)) > xy_deadband or abs(float(msg.w_twist)) > yaw_deadband:
        return True
    return False


def is_disable_msg(msg):
    return bool(msg.disable_pump or msg.disable_torque or msg.action_valve)


def is_latch_reset_msg(msg):
    return bool(msg.set_init) and not is_disable_msg(msg)


def is_motion_msg(msg, xy_deadband, yaw_deadband):
    if bool(msg.moving):
        return True
    if abs(float(msg.x_vec)) > xy_deadband or abs(float(msg.y_vec)) > xy_deadband:
        return True
    if abs(float(msg.z_vec)) > xy_deadband or abs(float(msg.w_twist)) > yaw_deadband:
        return True
    return False


def normalized_manual_msg(msg):
    # run_agent2.py ignores stop, so a stop-only pulse should become a stand pulse.
    if bool(msg.stop) and not bool(msg.disable_pump or msg.disable_torque or msg.action_valve or msg.change_mode):
        return make_stand_msg()
    return deepcopy(msg)


class CommandMux:
    def __init__(self, args):
        self.args = args
        self.manual_msg = None
        self.manual_stamp = 0.0
        self.pcr_msg = None
        self.pcr_stamp = 0.0
        self.last_manual_forward_stamp = 0.0
        self.last_pcr_forward_stamp = 0.0
        self.last_stand_publish = 0.0
        self.last_source = "none"
        self.last_debug = 0.0
        self.disable_latched_msg = None
        self.disable_latched_stamp = 0.0

        self.pub = rospy.Publisher(args.output_topic, JoyCommand, queue_size=1)
        self.debug_pub = rospy.Publisher(args.debug_topic, String, queue_size=1)
        self.manual_sub = rospy.Subscriber(args.manual_topic, JoyCommand, self.on_manual, queue_size=1)
        self.pcr_sub = rospy.Subscriber(args.pcr_topic, JoyCommand, self.on_pcr, queue_size=1)

    def on_manual(self, msg):
        self.manual_msg = deepcopy(msg)
        self.manual_stamp = time.monotonic()

    def on_pcr(self, msg):
        self.pcr_msg = deepcopy(msg)
        self.pcr_stamp = time.monotonic()

    def publish_stand_if_needed(self, source, now):
        repeat_dt = 1.0 / max(float(self.args.stand_repeat_hz), 1e-6)
        if self.last_source != source or now - self.last_stand_publish >= repeat_dt:
            self.pub.publish(make_stand_msg())
            self.last_stand_publish = now
            self.last_source = source
            return True
        return False

    def publish_latched_disable_if_needed(self, now):
        repeat_dt = 1.0 / max(float(self.args.stand_repeat_hz), 1e-6)
        source = "manual_disable_latched"
        if self.last_source != source or now - self.last_stand_publish >= repeat_dt:
            self.pub.publish(deepcopy(self.disable_latched_msg))
            self.last_stand_publish = now
            self.last_source = source
            return True
        return False

    def publish_debug(self, source, forwarded, now):
        if now - self.last_debug < 1.0 / max(float(self.args.debug_hz), 1e-6):
            return
        self.last_debug = now
        manual_age = None if self.manual_stamp <= 0.0 else now - self.manual_stamp
        pcr_age = None if self.pcr_stamp <= 0.0 else now - self.pcr_stamp
        msg = String()
        msg.data = json.dumps(
            {
                "source": source,
                "forwarded": bool(forwarded),
                "manual_age_s": manual_age,
                "pcr_age_s": pcr_age,
                "manual_topic": self.args.manual_topic,
                "pcr_topic": self.args.pcr_topic,
                "output_topic": self.args.output_topic,
                "disable_latched": self.disable_latched_msg is not None,
            },
            ensure_ascii=False,
        )
        self.debug_pub.publish(msg)

    def step(self):
        now = time.monotonic()
        forwarded = False
        source = "idle_stand"

        manual_fresh = self.manual_msg is not None and now - self.manual_stamp <= float(self.args.manual_timeout_s)
        pcr_fresh = self.pcr_msg is not None and now - self.pcr_stamp <= float(self.args.pcr_timeout_s)

        if manual_fresh and is_disable_msg(self.manual_msg):
            source = "manual_disable"
            self.disable_latched_msg = normalized_manual_msg(self.manual_msg)
            self.disable_latched_stamp = self.manual_stamp
            if self.manual_stamp > self.last_manual_forward_stamp:
                self.pub.publish(deepcopy(self.disable_latched_msg))
                self.last_manual_forward_stamp = self.manual_stamp
                self.last_source = source
                forwarded = True
        elif manual_fresh and is_latch_reset_msg(self.manual_msg):
            source = "manual"
            self.disable_latched_msg = None
            if self.manual_stamp > self.last_manual_forward_stamp:
                self.pub.publish(normalized_manual_msg(self.manual_msg))
                self.last_manual_forward_stamp = self.manual_stamp
                self.last_source = source
                forwarded = True
        elif self.disable_latched_msg is not None:
            source = "manual_disable_latched"
            forwarded = self.publish_latched_disable_if_needed(now)
        elif manual_fresh and has_manual_priority(self.manual_msg, self.args.manual_xy_deadband, self.args.manual_yaw_deadband):
            source = "manual"
            if self.manual_stamp > self.last_manual_forward_stamp:
                self.pub.publish(normalized_manual_msg(self.manual_msg))
                self.last_manual_forward_stamp = self.manual_stamp
                self.last_source = source
                forwarded = True
        elif bool(self.args.pcr_enabled) and pcr_fresh:
            if bool(self.args.pcr_zero_as_stand) and not is_motion_msg(
                self.pcr_msg,
                self.args.pcr_xy_deadband,
                self.args.pcr_yaw_deadband,
            ):
                source = "pcr_stand"
                forwarded = self.publish_stand_if_needed(source, now)
            else:
                source = "pcr"
                if self.pcr_stamp > self.last_pcr_forward_stamp:
                    self.pub.publish(deepcopy(self.pcr_msg))
                    self.last_pcr_forward_stamp = self.pcr_stamp
                    self.last_source = source
                    forwarded = True
        else:
            forwarded = self.publish_stand_if_needed(source, now)

        self.publish_debug(source, forwarded, now)

    def spin(self):
        rospy.loginfo(
            "usr_command_mux: manual=%s pcr=%s output=%s rate=%.1fHz",
            self.args.manual_topic,
            self.args.pcr_topic,
            self.args.output_topic,
            float(self.args.rate_hz),
        )
        rate = rospy.Rate(float(self.args.rate_hz))
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(description="Safe selector for interface/joy_command sources.")
    parser.add_argument("--manual_topic", default="/usr/command_manual")
    parser.add_argument("--pcr_topic", default="/usr/command_pcr")
    parser.add_argument("--output_topic", default="/usr/command")
    parser.add_argument("--debug_topic", default="/usr_command_mux/debug")
    parser.add_argument("--rate_hz", type=float, default=50.0)
    parser.add_argument("--debug_hz", type=float, default=2.0)
    parser.add_argument("--manual_timeout_s", type=float, default=0.2)
    parser.add_argument("--pcr_timeout_s", type=float, default=0.35)
    parser.add_argument("--stand_repeat_hz", type=float, default=2.0)
    parser.add_argument("--manual_xy_deadband", type=float, default=0.03)
    parser.add_argument("--manual_yaw_deadband", type=float, default=0.05)
    parser.add_argument("--pcr_xy_deadband", type=float, default=1e-4)
    parser.add_argument("--pcr_yaw_deadband", type=float, default=1e-4)
    parser.add_argument("--pcr_enabled", dest="pcr_enabled", action="store_true", default=True)
    parser.add_argument("--no_pcr_enabled", dest="pcr_enabled", action="store_false")
    parser.add_argument("--pcr_zero_as_stand", dest="pcr_zero_as_stand", action="store_true", default=True)
    parser.add_argument("--no_pcr_zero_as_stand", dest="pcr_zero_as_stand", action="store_false")
    return parser.parse_args()


def main():
    global rospy, JoyCommand, String
    args = parse_args()
    try:
        import rospy as rospy_mod
        from interface.msg import joy_command as joy_command_msg
        from std_msgs.msg import String as string_msg
    except ImportError as exc:
        raise SystemExit("ROS Python modules not found; source the catkin workspace first.") from exc
    rospy = rospy_mod
    JoyCommand = joy_command_msg
    String = string_msg

    rospy.init_node("usr_command_mux")
    CommandMux(args).spin()


if __name__ == "__main__":
    main()
