#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64MultiArray
import threading

LEFT_LEN = 27
RIGHT_LEN = 27

class SitaCurAggregator:
    def __init__(self):
        rospy.init_node("sita_cur_aggregator")

        # 发布频率（Hz），启动后立即发布
        self.rate_hz = float(rospy.get_param("~rate", 100.0))

        # 订阅输入话题
        self.left_sub = rospy.Subscriber("left_sita_cur", Float64MultiArray, self.left_cb, queue_size=10)
        self.right_sub = rospy.Subscriber("right_sita_cur", Float64MultiArray, self.right_cb, queue_size=10)

        # 发布输出话题
        self.pub = rospy.Publisher("/sita_cur", Float64MultiArray, queue_size=10)

        # 缓存最新数据（零初始化：总长度 54）
        self._lock = threading.Lock()
        self._left = [0.0] * LEFT_LEN
        self._right = [0.0] * RIGHT_LEN

        # 定时发布
        period = 1.0 / max(self.rate_hz, 1e-6)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)

        rospy.loginfo("sita_cur_aggregator started. rate=%.2f Hz, left_len=%d, right_len=%d",
                      self.rate_hz, LEFT_LEN, RIGHT_LEN)

    def _normalize_side(self, arr, expected_len):
        # 截断或零填充，保持固定长度
        out = [0.0] * expected_len
        n = min(expected_len, len(arr))
        out[:n] = arr[:n]
        return out

    def left_cb(self, msg: Float64MultiArray):
        with self._lock:
            self._left = self._normalize_side(list(msg.data), LEFT_LEN)

    def right_cb(self, msg: Float64MultiArray):
        with self._lock:
            self._right = self._normalize_side(list(msg.data), RIGHT_LEN)

    def timer_cb(self, _event):
        with self._lock:
            left = self._left
            right = self._right

        out = Float64MultiArray()
        # 左侧数据在前，右侧数据在后；固定总长度 54
        out.data = list(left) + list(right)
        self.pub.publish(out)

if __name__ == "__main__":
    try:
        SitaCurAggregator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass