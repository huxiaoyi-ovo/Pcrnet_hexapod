# import threading
# import queue
# import time
# import random
# import numpy as np
# task_queue=queue.Queue(maxsize=100)
# stop_event=threading.Event()

# def task1():
#     while not stop_event.is_set():
#         task_vale=task_queue.get(block=True)
#         print("task1 get ",task_vale," in queue")
#         if task_vale<0.2:
#             stop_event.set()
#             break


# def task2():
#     while not stop_event.is_set():
#         task_value=random.random()
#         task_queue.put(task_value,block=True)
#         print("task2 put ",task_value," in queue")
#         time.sleep(0.5)



# def main():
#     get_thread=threading.Thread(target=task1)
#     put_thread=threading.Thread(target=task2)

#     get_thread.start()
#     put_thread.start()

#     get_thread.join()
#     put_thread.join()


# if __name__=='__main__':
#     a=np.zeros(3)
#     b=np.array([1.0,8.9,0.8])
#     print(np.dtype(b[0]))
#     main()
import rospy
from std_msgs.msg import String
import time
import threading
start_time=time.time()
Lock=threading.Lock()
def callback(data,args):
    with Lock:
        print(args,"->",time.time()-start_time)
        time.sleep(0.5)
    

def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/topic1", String, callback,callback_args="topic1")
    rospy.Subscriber("/topic2", String, callback,callback_args="topic2")
    rospy.spin()

if __name__ == '__main__':
    main()
