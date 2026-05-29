#!/usr/bin/python3
import threading
import queue
import time
import signal
import rospy
from std_msgs.msg import String
from pynput.keyboard import Listener, Key
from interface.msg import joy_command

# 创建一个线程安全的队列
key_queue = queue.Queue(maxsize=2)
exit_event=threading.Event()
last_key_time=time.time()

def keyboard_listener():
    with Listener(on_press=on_press) as listener:
        listener.join()

def on_press(key):
    global last_key_time
    current_time=time.time()
    if current_time-last_key_time<=0.05:
        return
    last_key_time=current_time
    if exit_event.is_set():
        print("on press")
        return False
    try:
        key_char=key.char
    except:
        key_char=str(key)
    print(key_char)
    if key == Key.esc:
        exit_event.set()
        return False
    try:
        key_queue.put(key_char,block=False)
    except queue.Full:
        print("key queue if full, try latter")
    except Exception as e:
        exit_event.set()
        print("Error: ",e," occured")

def ros_publisher():
    
    command_pub = rospy.Publisher('/usr/command',joy_command,queue_size=10)
    joy_msg=joy_command()

    while not rospy.is_shutdown() and not exit_event.is_set():
        joy_msg=joy_command()
        # print("publisher continue")
        try:
            keyboard_input = key_queue.get(block=True,timeout=0.1)
            if keyboard_input == 'b':
                joy_msg.set_init=True
            elif keyboard_input == 'w':
                joy_msg.y_vec=0.9
            elif keyboard_input =='s':
                joy_msg.y_vec=-0.9
            elif keyboard_input == 'a':
                joy_msg.x_vec=-0.9
            elif keyboard_input == 'd':
                joy_msg.x_vec=0.9
            elif keyboard_input == 'Key.left':
                joy_msg.w_twist=0.9
            elif keyboard_input == 'Key.right':
                joy_msg.w_twist=-0.9
            elif keyboard_input == 'e':
                joy_msg.disable_torque=True
            else:
                continue
            for _ in range(20):
                command_pub.publish(joy_msg)
                time.sleep(0.005)
        except queue.Empty:
            pass

def exit_handler(_,__):
    exit_event.set()
    print("handel ctrl+c")

def main():
    signal.signal(signal.SIGINT,exit_handler)
    rospy.init_node('keyboard_publisher')
    listener_thread = threading.Thread(target=keyboard_listener)
    # listener_thread = threading.Thread(target=task1)
    publisher_thread = threading.Thread(target=ros_publisher)

    listener_thread.start()
    publisher_thread.start()
    listener_thread.join()
    publisher_thread.join()

    print("---------------->Exit keyborad_publisher node<---------------")

if __name__ == "__main__":
    main()