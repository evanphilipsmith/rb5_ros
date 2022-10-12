#!/usr/bin/env python
import time
import csv
import os
import math
import rospy
from sensor_msgs.msg import Joy
import numpy as np
from megapi_controller import MegapiController

WHEELBASE = 0.111125  # dist between centers of front & rear wheels in meters
TRACK = 0.136525      # dist between centers of front wheels in meters
WHEEL_RADIUS = 0.03   # meters

LIN_VEL = 0.3
ANG_VEL = 0.1
CFG_TIME_TO_ROTATE_90_DEG = 2.0
CFG_TIME_TO_TRAVERSE_1_METER = 4.0

class BasicAlgorithmNode:
    def __init__(self, dryrun=True):
        self.pub_joy = rospy.Publisher("/joy", Joy, queue_size=1)
        self.waypoints = []
        self.robot = MegapiController()
        self.dryrun = dryrun

    def read_waypoints_csv_file(self, filename):
        # type: (str) -> None
        with open(filename) as f:
            reader = csv.reader(f)
            self.add_waypoints(list(reader))

    def add_waypoints(self, waypoints):
        # type: (list) -> None
        for waypoint in waypoints:
            if len(waypoint) != 3:
                print('Invalid waypoint: {}'.format(waypoint))
            try:
                waypoint = (
                    int(waypoint[0]),
                    int(waypoint[1]),
                    float(waypoint[2]),
                )
            except ValueError:
                print('Invalid waypoint: {}'.format(waypoint))
            self.waypoints.append(waypoint)

    def run(self):
        print(self.waypoints)
        if not self.waypoints:
            return
        current_pose = self.waypoints.pop(0)
        while self.waypoints:
            goal = self.waypoints.pop(0)
            print('Current pose: {}\tGoal: {}'.format(current_pose, goal))
            distance = round(math.sqrt((goal[0]-current_pose[0])**2 + (goal[1]-current_pose[1])**2), 2)
            angle_from_point = round(math.atan2(goal[1]-current_pose[1], goal[0]-current_pose[0]), 2)
            goal_angle = current_pose[2] - angle_from_point
            if abs(goal_angle) == round(math.pi, 2):
                print('Skipping preturn, going in reverse')
                goal_angle = 0
                distance *= -1
            final_angle = -(goal[2] - (current_pose[2] - goal_angle))
            if abs(final_angle) == round(math.pi, 2):
                print('Flipping goal angle, going in reverse')
                goal_angle *= -1
                distance *= -1
                final_angle = -(goal[2] - (current_pose[2] - goal_angle))

            # Drive
            if abs(goal_angle) > 0.02:  # Skip if turn is ~1 degree or less
                self.turn(goal_angle)
            self.driveStraight(distance)
            if abs(final_angle) > 0.02:  # Skip if turn is ~1 degree or less
                self.turn(final_angle)
            current_pose = goal

            print('-'*15)
            time.sleep(1)

    def turn(self, angle):
        # type: (float) -> None
        print('Turn {} radians'.format(angle))
        if self.dryrun:
            return
        turn_clockwise = 1 if angle > 0 else -1
        motors = self.kinematic_model(0, 0, ANG_VEL * turn_clockwise)
        self.robot.setMotors(*motors)
        # Normalize and multiply by time it takes to rotate 90 degrees
        time.sleep(abs(angle) / 1.57 * CFG_TIME_TO_ROTATE_90_DEG)
        self.robot.stop()
    
    def driveStraight(self, distance):
        # type: (float) -> None
        drive_direction = 1 if distance > 0 else -1
        print('Drive {} meters'.format(distance))
        if self.dryrun:
            return
        motors = self.kinematic_model(0, LIN_VEL * drive_direction, 0)
        self.robot.setMotors(*motors)
        time.sleep(abs(distance) * CFG_TIME_TO_TRAVERSE_1_METER)
        self.robot.stop()

    def kinematic_model(self, linear_vel_x, linear_vel_y, angular_vel):
        # type: (float, float, float) -> tuple
        #      __
        #     /  \     Front
        # [1]|    |[2]
        #    |    | 
        # [3]|    |[4]
        #     \__/     Rear
        GEOM_COEFFICIENT = (WHEELBASE / 2) + (TRACK / 2)
        jacobian = np.array([
            [1, -1, -GEOM_COEFFICIENT],
            [1, 1, GEOM_COEFFICIENT],
            [1, 1, -GEOM_COEFFICIENT],
            [1, -1, GEOM_COEFFICIENT],
        ]) * 1 / WHEEL_RADIUS
        inputs = np.array([linear_vel_x, linear_vel_y, angular_vel])
        wheel_velocities = np.dot(jacobian, inputs)
        return tuple(wheel_velocities.flatten())


if __name__ == "__main__":
    basic_algorithm_node = BasicAlgorithmNode()
    waypoint_filename = os.path.join(os.path.dirname(__file__), 'waypoints.txt')
    rospy.init_node("basic_algorithm")
    basic_algorithm_node.read_waypoints_csv_file(waypoint_filename)
    basic_algorithm_node.run()
