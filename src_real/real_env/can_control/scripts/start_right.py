#!/usr/bin/env python3
from node import *

if __name__ == '__main__':
    try:
        node = rosnode(node_name='right_control',body_part=part.right,port='/dev/DM_r')
        # node = rosnode(node_name='right_control',body_part=part.right,port='/dev/ttyACM1')
        node.run()
    except rospy.ROSInterruptException:
        pass
        