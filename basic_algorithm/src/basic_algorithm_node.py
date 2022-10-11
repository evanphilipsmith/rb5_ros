#!/usr/bin/env python
import time
import rospy
from sensor_msgs.msg import Joy

class BasicAlgorithmNode:
    def __init__(self):
        self.pub_joy = rospy.Publisher("/joy", Joy, queue_size=1)

    def run(self):
        for _ in range(5):
            joy_msg = Joy()
            self.pub_joy.publish(joy_msg)
            time.sleep(1)


if __name__ == "__main__":
    basic_algorithm_node = BasicAlgorithmNode()
    rospy.init_node("basic_algorithm")
    basic_algorithm_node.run()
