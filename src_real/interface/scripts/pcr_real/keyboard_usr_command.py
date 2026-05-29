#!/usr/bin/env python3
"""Low-speed keyboard publisher for src_real /usr/command."""

import argparse
import select
import signal
import sys
import termios
import threading
import time
import tty


rospy = None
JoyCommand = None


class TerminalMode:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key(timeout_s):
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    return sys.stdin.read(1)


def clip(value, limit):
    return max(-limit, min(limit, value))


def make_msg(
    x_vec=0.0,
    y_vec=0.0,
    z_vec=0.0,
    w_twist=0.0,
    *,
    set_init=False,
    moving=False,
    disable_pump=False,
    disable_torque=False,
    action_valve=False,
    stop=False,
    change_mode=False,
):
    if JoyCommand is None:
        raise RuntimeError("interface.msg.joy_command is not loaded; source the catkin workspace first.")
    msg = JoyCommand()
    msg.set_init = bool(set_init)
    msg.moving = bool(moving or abs(x_vec) > 1e-6 or abs(y_vec) > 1e-6 or abs(z_vec) > 1e-6 or abs(w_twist) > 1e-6)
    msg.disable_pump = bool(disable_pump)
    msg.disable_torque = bool(disable_torque)
    msg.action_valve = bool(action_valve)
    msg.stop = bool(stop)
    msg.change_mode = bool(change_mode)
    msg.x_vec = float(x_vec)
    msg.y_vec = float(y_vec)
    msg.z_vec = float(z_vec)
    msg.w_twist = float(w_twist)
    return msg


def publish_zero(pub, repeat=8, rate_hz=30.0):
    delay = 1.0 / max(rate_hz, 1e-6)
    msg = make_msg(stop=True)
    for _ in range(repeat):
        pub.publish(msg)
        time.sleep(delay)


def check_no_joy_ctrl():
    try:
        nodes = subprocess_check_output(["rosnode", "list"], timeout_s=1.0)
    except Exception:
        return
    if any(node.rstrip("/").endswith("/joy_ctrl") or node == "/joy_ctrl" for node in nodes.splitlines()):
        raise RuntimeError("joy_ctrl is running; stop it before starting keyboard_usr_command.py")


def subprocess_check_output(cmd, timeout_s):
    import subprocess

    return subprocess.check_output(cmd, timeout=timeout_s, text=True, stderr=subprocess.DEVNULL)


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard to interface/joy_command.")
    parser.add_argument("--topic", default="/usr/command")
    parser.add_argument("--rate_hz", type=float, default=20.0)
    parser.add_argument("--x_speed", type=float, default=0.06, help="right speed while key is held, m/s")
    parser.add_argument("--y_speed", type=float, default=0.10, help="forward speed while key is held, m/s")
    parser.add_argument("--z_speed", type=float, default=0.06, help="vertical command while key is held")
    parser.add_argument("--yaw_speed", type=float, default=0.20, help="yaw speed while key is held, rad/s")
    parser.add_argument("--max_x", type=float, default=0.08, help="max right speed before run_agent scaling, m/s")
    parser.add_argument("--max_y", type=float, default=0.12, help="max forward speed before run_agent scaling, m/s")
    parser.add_argument("--max_z", type=float, default=0.08, help="max vertical command")
    parser.add_argument("--max_yaw", type=float, default=0.25, help="max yaw speed before run_agent scaling, rad/s")
    parser.add_argument("--key_hold_timeout_s", type=float, default=0.18, help="stop if no repeat key event arrives")
    parser.add_argument("--usr_command_x_scale", type=float, default=1.6)
    parser.add_argument("--usr_command_y_scale", type=float, default=2.4)
    parser.add_argument("--usr_command_yaw_scale", type=float, default=0.375)
    parser.add_argument("--input_mode", choices=["auto", "pynput", "terminal"], default="auto")
    parser.add_argument("--allow_joy_ctrl", action="store_true")
    return parser.parse_args()


def command_from_keys(active_keys, args):
    x_cmd = float(args.x_speed) * (float("d" in active_keys) - float("a" in active_keys))
    y_cmd = float(args.y_speed) * (float("w" in active_keys) - float("s" in active_keys))
    z_cmd = float(args.z_speed) * (float("r" in active_keys) - float("f" in active_keys))
    yaw_cmd = float(args.yaw_speed) * (float("q" in active_keys) - float("e" in active_keys))
    return clip(x_cmd, args.max_x), clip(y_cmd, args.max_y), clip(z_cmd, args.max_z), clip(yaw_cmd, args.max_yaw)


def publish_command(pub, args, x_cmd, y_cmd, z_cmd, yaw_cmd, *, pulse=None):
    x_vec = clip(x_cmd / args.usr_command_x_scale, 1.0)
    y_vec = clip(y_cmd / args.usr_command_y_scale, 1.0)
    z_vec = clip(z_cmd, 1.0)
    w_twist = clip(yaw_cmd / args.usr_command_yaw_scale, 1.0)
    kwargs = {}
    if pulse:
        x_vec = y_vec = z_vec = w_twist = 0.0
        if pulse == "moving":
            kwargs["set_init"] = True
            kwargs["moving"] = True
        else:
            kwargs[pulse] = True
    pub.publish(make_msg(x_vec=x_vec, y_vec=y_vec, z_vec=z_vec, w_twist=w_twist, **kwargs))
    print(
        f"\rcmd=({x_cmd:+.3f},{y_cmd:+.3f},{z_cmd:+.3f},{yaw_cmd:+.3f}) "
        f"joy=({x_vec:+.3f},{y_vec:+.3f},{z_vec:+.3f},{w_twist:+.3f}) pulse={pulse or '-'}     ",
        end="",
        flush=True,
    )


PULSE_KEYS = {
    "1": "set_init",
    "2": "moving",
    "3": "stop",
    "4": "change_mode",
    "5": "disable_pump",
    "6": "disable_torque",
    "7": "action_valve",
}


def run_pynput_loop(pub, args, should_stop):
    from pynput import keyboard

    active_keys = set()
    pending_pulses = []
    lock = threading.Lock()

    def key_to_char(key):
        if key == keyboard.Key.space:
            return " "
        if key == keyboard.Key.esc:
            return "\x1b"
        try:
            return str(key.char).lower()
        except AttributeError:
            return ""

    def on_press(key):
        char = key_to_char(key)
        with lock:
            if char in ("w", "s", "a", "d", "q", "e", "r", "f"):
                active_keys.add(char)
            elif char in PULSE_KEYS:
                pending_pulses.append(PULSE_KEYS[char])
            elif char in (" ", "x"):
                active_keys.clear()
            elif char == "\x1b":
                should_stop.set()
                return False
        return True

    def on_release(key):
        char = key_to_char(key)
        with lock:
            active_keys.discard(char)
        return not should_stop.is_set()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    rate = rospy.Rate(args.rate_hz)
    try:
        while not rospy.is_shutdown() and not should_stop.is_set():
            with lock:
                keys = set(active_keys)
                pulse = pending_pulses.pop(0) if pending_pulses else None
            publish_command(pub, args, *command_from_keys(keys, args), pulse=pulse)
            rate.sleep()
    finally:
        listener.stop()


def run_terminal_loop(pub, args, should_stop):
    active_keys = {}
    rate = rospy.Rate(args.rate_hz)
    with TerminalMode():
        while not rospy.is_shutdown() and not should_stop.is_set():
            now = time.time()
            key = read_key(1.0 / max(args.rate_hz, 1e-6))
            pulse = None
            if key:
                key = key.lower()
                if key in ("w", "s", "a", "d", "q", "e", "r", "f"):
                    active_keys[key] = now
                elif key in PULSE_KEYS:
                    pulse = PULSE_KEYS[key]
                elif key in (" ", "x"):
                    active_keys.clear()
                elif key == "\x03":
                    should_stop.set()
                    continue

            timeout_s = max(float(args.key_hold_timeout_s), 0.01)
            active_keys = {k: t for k, t in active_keys.items() if now - t <= timeout_s}
            publish_command(pub, args, *command_from_keys(active_keys, args), pulse=pulse)
            rate.sleep()


def main():
    global rospy, JoyCommand
    args = parse_args()
    try:
        import rospy as rospy_mod
        from interface.msg import joy_command as joy_command_msg
    except ImportError as exc:
        raise SystemExit("ROS Python modules not found; run: source devel/setup.bash") from exc
    rospy = rospy_mod
    JoyCommand = joy_command_msg

    if not args.allow_joy_ctrl:
        check_no_joy_ctrl()

    rospy.init_node("keyboard_usr_command")
    pub = rospy.Publisher(args.topic, JoyCommand, queue_size=1)
    should_stop = threading.Event()

    def handle_signal(_signum, _frame):
        should_stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Keyboard /usr/command control")
    print("  Hold W/S/A/D/Q/E/R/F to move; release to stop. X or Space: zero")
    print("  1:set_init 2:moving_init 3:stop 4:change_mode")
    print("  5:disable_pump 6:disable_torque 7:action_valve")
    print("  Ctrl+C: zero and exit")
    print("  Keep this node exclusive with joy_ctrl and PCR publish_cmd.")

    try:
        if args.input_mode in ("auto", "pynput"):
            try:
                run_pynput_loop(pub, args, should_stop)
            except ImportError:
                if args.input_mode == "pynput":
                    raise
                print("\npynput not found; falling back to terminal key-repeat mode.")
                run_terminal_loop(pub, args, should_stop)
        else:
            run_terminal_loop(pub, args, should_stop)
    finally:
        print("\nPublishing zero command before exit.")
        publish_zero(pub)


if __name__ == "__main__":
    main()
