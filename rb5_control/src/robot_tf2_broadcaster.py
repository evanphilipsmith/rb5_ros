#!/usr/bin/env python  
import rospy
import tf_conversions
import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped


def handle_pose(msg):
    # type: (Pose) -> None
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "robot"
    t.transform.translation.x = msg.position.x
    t.transform.translation.y = msg.position.y
    t.transform.translation.z = 0.0
    t.transform.rotation.x = msg.orientation.x
    t.transform.rotation.y = msg.orientation.y
    t.transform.rotation.z = msg.orientation.z
    t.transform.rotation.w = msg.orientation.w

    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('robot_tf2_broadcaster')
    rospy.Subscriber('/pose', Pose, handle_pose)
    rospy.spin()
