#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv
from geometry_msgs.msg import Twist
import rospy
from april_detection.msg import AprilTagDetectionArray
from open_loop import PIDcontroller, coord, genTwistMsg

INITIAL_COVARIANCE = np.diag([0.005, 0.005, 0.001]) ** 2
NUM_LANDMARK_PARAMS = 2
VALID_LANDMARK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RANGE_UNCERTAINTY = 0.1  # meters
BEARING_UNCERTAINTY = math.pi / 180 # radians
SENSOR_NOISE = np.diag([RANGE_UNCERTAINTY, BEARING_UNCERTAINTY]) ** 2

LANDMARK_SCALE_FACTOR = 2.44  # meters (8 ft)
LANDMARKS = LANDMARK_SCALE_FACTOR * np.array([
    [0.25, 0.0],
    [0.75, 0.0],
    [1.0, 0.25],
    [1.0, 0.75],
    [0.75, 1.0],
    [0.25, 1.0],
    [0.0, 0.75],
    [0.0, 0.25],
])

SQUARE_SCALE_FACTOR = 1.22  # meters (4 ft)
SQUARE_PATH = SQUARE_SCALE_FACTOR * np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, math.pi / 2],
    [0.0, 1.0, math.pi],
    [0.0, 0.0, -(math.pi / 2)],
])
SQUARE_TRANSLATE_FACTOR = np.array([0.61, 0.61])  # meters ([2 ft, 2 ft])
SQUARE_LANDMARKS = LANDMARKS - SQUARE_TRANSLATE_FACTOR

OCTAGON_SCALE_FACTOR = 0.61  # meters (2 ft)
OCTAGON_PATH = OCTAGON_SCALE_FACTOR * np.array([
    [0.0, 0.0, 0.0],
    [math.sqrt(2)/2, 1-math.sqrt(2)/2, math.pi / 8],
    [1.0, 1.0, 3 * math.pi / 8],
    [math.sqrt(2)/2, 1+math.sqrt(2)/2, 5 * math.pi / 8],
    [0.0, 2.0, 7 * math.pi / 8],
    [-math.sqrt(2)/2, 1+math.sqrt(2)/2, 9 * math.pi / 8],
    [-1.0, 1.0, 11 * math.pi / 8],
    [-math.sqrt(2)/2, 1-math.sqrt(2)/2, 13 * math.pi / 8],
    [0.0, 0.0, 15 * math.pi / 8],
])
OCTAGON_TRANSLATE_FACTOR = np.array([1.22, 0.61])  # meters ([4 ft, 2 ft])
OCTAGON_LANDMARKS = LANDMARKS - OCTAGON_TRANSLATE_FACTOR

class SLAMNode:
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0])
        self.landmark_id_lookup = {}
        self.covariance = INITIAL_COVARIANCE
        self.twist_history = np.array([[0.0, 0.0, time.time()]])
        self.last_image_timestamp = time.time()

    @staticmethod
    def predict_vehicle_covariance(current_covariance, current_state, pose_delta):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
        LINEAR_UNCERTAINTY = 0.02  # meters
        ANGULAR_UNCERTAINTY = math.pi / 180 # radians
        NOISE = np.diag([LINEAR_UNCERTAINTY, ANGULAR_UNCERTAINTY]) ** 2
        linear_pose_delta, angular_pose_delta = pose_delta
        _, _, vehicle_orientation = current_state
        f_x = np.array([
            # FIXME: should it be sin(vehicle_orientation) or sin(angular_pose_delta) ?
            [1, 0, -linear_pose_delta * math.sin(vehicle_orientation)],
            [0, 1, linear_pose_delta * math.cos(vehicle_orientation)],
            [0, 0, 1]
        ])
        f_v = np.array([
            [np.cos(vehicle_orientation), 0],
            [np.sin(vehicle_orientation), 0],
            [0, 1]
        ])
        return np.matmul(np.matmul(f_x, current_covariance), f_x.T) + np.matmul(np.matmul(f_v, NOISE), f_v.T)

    @staticmethod
    def predict_vehicle_state(current_state, pose_delta):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        linear_pose_delta, angular_pose_delta = pose_delta
        _, _, orientation = current_state
        predicted_state = current_state + np.array([
            linear_pose_delta * math.cos(orientation),
            linear_pose_delta * math.sin(orientation),
            angular_pose_delta
        ])
        return predicted_state

    def predict_state(self, vehicle_pose_delta):
        # type: (np.ndarray) -> None
        # Landmark state doesn't change since we assume landmarks are stationary.
        predicted_vehicle_state = self.predict_vehicle_state(self.state[:3], vehicle_pose_delta)
        self.state = np.concatenate((predicted_vehicle_state, self.state[3:]))

    def predict_covariance(self, vehicle_pose_delta):
        # type: (np.ndarray) -> None
        # Landmark covariance doesn't change since we assume landmarks are stationary.
        vehicle_covariance = self.covariance[:3, :3]
        landmark_covariance = self.covariance[3:, 3:]
        # vehicle_landmark_correlation = self.covariance[:3, 3:]

        predicted_vehicle_covariance = self.predict_vehicle_covariance(vehicle_covariance, self.state[:3], vehicle_pose_delta)
        predicted_vehicle_landmark_correlation = np.matmul(self.state[:3].reshape(-1, 1), self.state[3:].reshape(1, -1))
        self.covariance = np.concatenate((
            np.concatenate(
                (predicted_vehicle_covariance, predicted_vehicle_landmark_correlation),
                axis=1),
            np.concatenate(
                (predicted_vehicle_landmark_correlation.T, landmark_covariance),
                axis=1)),
        axis=0)

    def predict(self, pose_delta):
        # type: (np.ndarray) -> None
        self.predict_state(pose_delta)
        self.predict_covariance(pose_delta)

    def update(self, sensor_observations, landmark_ids):
        # type: (np.ndarray, list) -> None
        def landmark_idx_to_state_idx(i):
            # type: (int) -> int
            return (i * NUM_LANDMARK_PARAMS) + 3
        sensor_observations = sensor_observations.reshape(-1, 2)
        for x, observation in zip(landmark_ids, sensor_observations):
            landmark_idx = self.landmark_id_lookup[x]
            landmark_pose_x, landmark_pose_y = self.state[landmark_idx:landmark_idx+2]
            vehicle_pose_x, vehicle_pose_y, vehicle_orientation = self.state[:3]
            measured_range, _ = observation
            predicted_observation = np.array([
                math.sqrt((landmark_pose_y-vehicle_pose_y)**2 + (landmark_pose_x-vehicle_pose_x)**2),
                math.atan2(landmark_pose_y-vehicle_pose_y, landmark_pose_x-vehicle_pose_x) - vehicle_orientation,
            ])
            innovation = observation - predicted_observation
            h_x_v = np.array([
                [-(landmark_pose_x-vehicle_pose_x) / measured_range, -(landmark_pose_y-vehicle_pose_y) / measured_range, 0],
                [(landmark_pose_y-vehicle_pose_y) / measured_range**2, -(landmark_pose_x-vehicle_pose_x) / measured_range**2, 1],
            ])
            h_p = np.array([
                [(landmark_pose_x-vehicle_pose_x) / measured_range, (landmark_pose_y-vehicle_pose_y) / measured_range],
                [-(landmark_pose_y-vehicle_pose_y) / measured_range**2, (landmark_pose_x-vehicle_pose_x) / measured_range**2],
            ])
            state_idx = landmark_idx_to_state_idx(landmark_idx)
            h_x = np.concatenate((
                h_x_v,
                np.zeros((2, state_idx-3)),
                h_p,
                np.zeros((2, self.state.size-(state_idx+NUM_LANDMARK_PARAMS)))), axis=1)
            s = np.matmul(np.matmul(h_x, self.covariance), h_x.T) + SENSOR_NOISE
            try:
                kalman_gain = np.matmul(np.matmul(self.covariance, h_x.T), np.linalg.inv(s))
            except np.linalg.LinAlgError as e:
                print("Encountered LinAlgError: {}.\tSkipping update...".format(e))
                return
            self.state = self.state + np.matmul(kalman_gain, innovation)
            self.covariance = self.covariance - np.matmul(np.matmul(kalman_gain, h_x), self.covariance)

    def extend_covariance(self, sensor_observation):
        # type: (np.ndarray) -> None
        n = self.covariance.shape[0]
        vehicle_orientation = self.state[2]
        r, b = sensor_observation
        insertion_jacobian = np.concatenate((
            np.concatenate((
                np.identity(n),
                np.zeros((n, 2))
            ), axis=1),
            np.concatenate((
                np.array([
                    [1, 0, -r * math.sin(vehicle_orientation + b)],
                    [0, 1, r * math.cos(vehicle_orientation + b)]]),
                np.zeros((2, n-3)),
                np.array([
                    [math.cos(vehicle_orientation + b), -r * math.sin(vehicle_orientation + b)],
                    [math.sin(vehicle_orientation + b), r * math.cos(vehicle_orientation + b)]])
            ), axis=1)
            ), axis=0)
        template = np.concatenate((
            np.concatenate((self.covariance, np.zeros((n, NUM_LANDMARK_PARAMS))), axis=1),
            np.concatenate((np.zeros((NUM_LANDMARK_PARAMS, n)), SENSOR_NOISE), axis=1)),
            axis=0)
        self.covariance = np.matmul(np.matmul(insertion_jacobian, template), insertion_jacobian.T)

    def extend_state(self, sensor_observations):
        # type: (np.ndarray) -> None
        if sensor_observations.size == 0:
            return
        assert sensor_observations.size % NUM_LANDMARK_PARAMS == 0
        vehicle_pose_x, vehicle_pose_y, vehicle_orientation = self.state[:3]
        num_new_observations = int(sensor_observations.size / NUM_LANDMARK_PARAMS)
        sensor_observations.reshape(num_new_observations, -1)

        def transform_observation(observation):
            r, b = observation
            x = vehicle_pose_x + r * math.cos(vehicle_orientation + b)
            y = vehicle_pose_y + r * math.sin(vehicle_orientation + b)
            return x, y
        observations_tf = np.apply_along_axis(
            transform_observation,
            1,
            sensor_observations.reshape(num_new_observations, NUM_LANDMARK_PARAMS))
        self.state = np.concatenate((self.state, observations_tf.flatten()))

    def extend(self, sensor_observations, new_landmark_ids):
        # type: (np.ndarray, list) -> None
        # Remove landmarks that aren't in the set of supported landmarks
        valid_landmark_indices = [i for i, x in enumerate(new_landmark_ids) if x in VALID_LANDMARK_IDS]
        new_landmark_ids = [x for x in new_landmark_ids if x in VALID_LANDMARK_IDS]
        sensor_observations = np.take(sensor_observations, valid_landmark_indices, axis=0)
        self.extend_state(sensor_observations)
        for observation, landmark_id in zip(sensor_observations, new_landmark_ids):
            self.extend_covariance(observation)
            self.landmark_id_lookup[landmark_id] = len(self.landmark_id_lookup)
        return sensor_observations, new_landmark_ids

    def twist_callback(self, msg):
        r_velocity = math.hypot(msg.linear.x, msg.linear.y)
        self.twist_history = np.append(self.twist_history, np.array([[r_velocity, msg.angular.z, time.time()]]), axis=0)

    def image_callback(self, msg):
        # Compute pose delta by integrating twist commands since the last image
        target_twist = np.where(
            np.logical_and(
                self.twist_history[:, 2] >= self.last_image_timestamp,
                self.twist_history[:, 2] <= time.time()))
        self.last_image_timestamp = time.time()
        pose_delta = np.array([
            np.trapz(self.twist_history[target_twist][:, 0], self.twist_history[target_twist][:, 2]),  # range
            np.trapz(self.twist_history[target_twist][:, 1], self.twist_history[target_twist][:, 2]),  # bearing
        ])
        known_landmark_observations = []
        known_landmark_ids = []
        unknown_landmark_observations = []
        unknown_landmark_ids = []
        for tag in msg.detections:
            if tag.id in self.landmark_id_lookup:
                known_landmark_ids.append(tag.id)
                known_landmark_observations.append([tag.pose.position.x, tag.pose.position.y])
            else:
                unknown_landmark_ids.append(tag.id)
                unknown_landmark_observations.append([tag.pose.position.x, tag.pose.position.y])
        known_landmark_observations = np.array(known_landmark_observations).reshape(-1, 2)
        unknown_landmark_observations = np.array(unknown_landmark_observations).reshape(-1, 2)
        # Convert landmark x,y coords to range and bearing
        known_delta = known_landmark_observations - self.state[:2]
        known_landmark_observations = np.concatenate((
            np.hypot(known_landmark_observations[:, 0], known_landmark_observations[:, 1]).reshape(-1, 1),
            np.arctan2(known_delta[:, 0], known_delta[:, 1]).reshape(-1, 1)
        ), axis=1)
        unknown_delta = unknown_landmark_observations - self.state[:2]
        unknown_landmark_observations = np.concatenate((
            np.hypot(unknown_landmark_observations[:, 0], unknown_landmark_observations[:, 1]).reshape(-1, 1),
            np.arctan2(unknown_delta[:, 0], unknown_delta[:, 1]).reshape(-1, 1)
        ), axis=1)

        new_landmark_observations, new_landmark_ids = self.extend(unknown_landmark_observations, unknown_landmark_ids)
        landmark_observations = np.concatenate((known_landmark_observations, new_landmark_observations), axis=0)
        landmark_ids = known_landmark_ids + new_landmark_ids
        self.predict(pose_delta)
        self.update(landmark_observations, landmark_ids)
        # TODO: save state and covariance history

def shutdown_callback():
    print('shutting down...')
    twist_pub.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))

def plot_results(path, landmarks):
    # type: (np.ndarray, np.ndarray) -> None
    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()

if __name__ == '__main__':
    DATA_FILENAME = "octagon_path_state.csv"
    waypoints = OCTAGON_PATH
    slam_node = SLAMNode()
    rospy.init_node('slam_node')
    rospy.on_shutdown(shutdown_callback)
    rospy.Subscriber('/twist', Twist, slam_node.twist_callback, queue_size=1)
    rospy.Subscriber('/apriltag_detection_array', AprilTagDetectionArray, slam_node.image_callback, queue_size=1)
    twist_pub = rospy.Publisher("/twist", Twist, queue_size=1)
    rate = rospy.Rate(10)  # 10 hz
    pid_controller = PIDcontroller(0.02, 0.005, 0.005)
    with open(DATA_FILENAME, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for waypoint in waypoints:
            print("Move to waypoint {}".format(waypoint))
            pid_controller.setTarget(waypoint)
            current_state = slam_node.state[:3]
            while not rospy.is_shutdown() and np.linalg.norm(pid_controller.getError(current_state, waypoint)) > 0.05:
                update_value = pid_controller.update(current_state)
                twist_pub.publish(genTwistMsg(coord(update_value, current_state)))
                rate.sleep()
                current_state = slam_node.state[:3]
                csv_writer.writerow(slam_node.state)
            if rospy.is_shutdown():
                break
    twist_pub.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))
