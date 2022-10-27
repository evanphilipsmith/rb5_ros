#!/usr/bin/env python
import sys
import rospy
from geometry_msgs.msg import Twist, Pose
import numpy as np
import tf_conversions
import tf2_ros

"""
The class of the pid controller.
"""
class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.1

    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the different between two states
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result 

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        # scale down the twist if its norm is more than the maximum value. 
        resultNorm = np.linalg.norm(result)
        if(resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result

def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0] 
    twist_msg.linear.y = desired_twist[1] 
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg

def genPoseMsg(x, y, yaw):
    # type: (float, float, float) -> Pose
    pose_msg = Pose()
    pose_msg.position.x = x
    pose_msg.position.y = y
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]
    return pose_msg

def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)

def correct_pose(tfBuffer, tag_id, tag_pose):
    try:
        t = tfBuffer.lookup_transform('robot', tag_id, rospy.Time())
        x, y, theta = unpack_transform(t)
        dist_to_tag = np.sqrt(x**2 + y**2)
        angle_offset = np.arctan(y/x)
        angle = theta - angle_offset
        robot_pose_x = tag_pose[0] - (dist_to_tag * np.cos(angle))
        robot_pose_y = tag_pose[1] + (dist_to_tag * np.sin(angle))
        robot_yaw = tag_pose[2] - theta
        return np.array([robot_pose_x, robot_pose_y, robot_yaw])
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        return None

def unpack_transform(t):
    x = t.transform.translation.x
    y = t.transform.translation.y
    quaternion = (
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
    )
    euler = tf_conversions.transformations.euler_from_quaternion(quaternion)
    theta = euler[2]
    return x, y, theta

if __name__ == "__main__":
    import time
    rospy.init_node("hw1")
    pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
    pub_pose = rospy.Publisher("/pose", Pose, queue_size=1)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    waypoints = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, np.pi]),
        np.array([0.0, 0.0, 0.0]),
    )
    tag_ids = (
        "marker_0",
        "marker_0",
        "marker_1",
        "marker_0",
    )
    tag_poses = (
        np.array([1.5, 0.0, 0.0]),
        np.array([1.5, 0.0, 0.0]),
        np.array([0.5, 2.0, np.pi]),
        np.array([1.5, 0.0, 0.0]),
    )

    # init pid controller
    pid = PIDcontroller(0.02,0.005,0.005)

    # init current state
    current_state = np.array([0.0,0.0,0.0])
    pub_pose.publish(genPoseMsg(x=current_state[0], y=current_state[1], yaw=current_state[2]))

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for i in range(len(waypoints)):
        wp = waypoints[i]
        tag_id = tag_ids[i]
        tag_pose = tag_poses[i]
        print("Move to waypoint {}".format(wp))
        # set wp as the target point
        pid.setTarget(wp)

        update_value = pid.update(current_state)
        pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
        time.sleep(0.05)
        current_state += update_value
        pub_pose.publish(genPoseMsg(x=current_state[0], y=current_state[1], yaw=current_state[2]))

        # Open loop traversal
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # check the error between current state and current way point
            update_value = pid.update(current_state)
            pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
            time.sleep(0.05)
            current_state += update_value
            pub_pose.publish(genPoseMsg(x=current_state[0], y=current_state[1], yaw=current_state[2]))
        
        # Closed loop error minimization
        updated_pose = None
        while updated_pose is None:
            print("Looking for {}...".format(tag_id))
            time.sleep(0.05)
            updated_pose = correct_pose(tfBuffer, tag_id, tag_pose)
        while np.linalg.norm(pid.getError(updated_pose, wp)) > 0.05:
            pub_pose.publish(genPoseMsg(x=updated_pose[0], y=updated_pose[1], yaw=updated_pose[2]))
            current_state = updated_pose
            update_value = pid.update(current_state)
            pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
            time.sleep(0.05)
            updated_pose = correct_pose(tfBuffer, tag_id, tag_pose)
            

    # stop the car and exit
    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
