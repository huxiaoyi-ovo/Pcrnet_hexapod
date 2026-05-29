#!/usr/bin/env python3
from node import *

if __name__ == '__main__':
    try:
        node = rosnode(node_name='left_control',body_part=part.left,port='/dev/DM_l')
        node.run()
    except rospy.ROSInterruptException:
        pass
        