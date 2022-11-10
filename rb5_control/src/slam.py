import numpy as np
import math
import time
from geometry_msgs.msg import Twist
import rospy
from april_detection.msg import AprilTagDetectionArray

INITIAL_COVARIANCE = np.diag([0.005, 0.005, 0.001]) ** 2
NUM_LANDMARK_PARAMS = 2
VALID_LANDMARK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RANGE_UNCERTAINTY = 0.1  # meters
BEARING_UNCERTAINTY = math.pi / 180 # radians
SENSOR_NOISE = np.diag([RANGE_UNCERTAINTY, BEARING_UNCERTAINTY]) ** 2

class SLAMNode:
    def __init__(self) -> None:
        self.state = np.array([0.0, 0.0, 0.0])
        self.landmark_id_lookup = {}
        self.covariance = INITIAL_COVARIANCE
        self.twist_history = np.array([[0.0, 0.0, time.time()]])
        self.last_image_timestamp = time.time()

    @staticmethod
    def predict_vehicle_covariance(current_covariance: np.ndarray, current_state: np.ndarray, pose_delta: np.ndarray):
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
        return f_x @ current_covariance @ f_x.T + f_v @ NOISE @ f_v.T

    @staticmethod
    def predict_vehicle_state(current_state: np.ndarray, pose_delta: np.ndarray):
        linear_pose_delta, angular_pose_delta = pose_delta
        _, _, orientation = current_state
        predicted_state = current_state + np.array([
            linear_pose_delta * math.cos(orientation),
            linear_pose_delta * math.sin(orientation),
            angular_pose_delta
        ])
        return predicted_state

    def predict_state(self, vehicle_pose_delta: np.ndarray):
        # Landmark state doesn't change since we assume landmarks are stationary.
        predicted_vehicle_state = self.predict_vehicle_state(self.state[:3], vehicle_pose_delta)
        self.state = np.concatenate((predicted_vehicle_state, self.state[3:]))

    def predict_covariance(self, vehicle_pose_delta: np.ndarray):
        # Landmark covariance doesn't change since we assume landmarks are stationary.
        vehicle_covariance = self.covariance[:3, :3]
        landmark_covariance = self.covariance[3:, 3:]
        # vehicle_landmark_correlation = self.covariance[:3, 3:]

        predicted_vehicle_covariance = self.predict_vehicle_covariance(vehicle_covariance, self.state[:3], vehicle_pose_delta)
        predicted_vehicle_landmark_correlation = (
            self.state[:3].reshape(-1, 1)
            @ self.state[3:].reshape(1, -1))
        self.covariance = np.concatenate((
            np.concatenate(
                (predicted_vehicle_covariance, predicted_vehicle_landmark_correlation),
                axis=1),
            np.concatenate(
                (predicted_vehicle_landmark_correlation.T, landmark_covariance),
                axis=1)),
        axis=0)

    def predict(self, pose_delta: np.ndarray):
        self.predict_state(pose_delta)
        self.predict_covariance(pose_delta)

    def update(self, sensor_observations: np.ndarray, landmark_ids: list):
        def landmark_idx_to_state_idx(i: int) -> int:
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
            s = h_x @ self.covariance @ h_x.T + SENSOR_NOISE
            kalman_gain = self.covariance @ h_x.T @ np.linalg.inv(s)
            self.state = self.state + (kalman_gain @ innovation)
            self.covariance = self.covariance - (kalman_gain @ h_x @ self.covariance)

    def extend_covariance(self, sensor_observation: np.ndarray):
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
        self.covariance = insertion_jacobian @ template @ insertion_jacobian.T

    def extend_state(self, sensor_observations: np.ndarray):
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

    def extend(self, sensor_observations: np.ndarray, new_landmark_ids: list):
        # Remove landmarks that aren't in the set of supported landmarks
        valid_landmark_indices = [i for i, x in enumerate(new_landmark_ids) if x in VALID_LANDMARK_IDS]
        new_landmark_ids = [x for x in new_landmark_ids if x in VALID_LANDMARK_IDS]
        sensor_observations = np.take(sensor_observations, valid_landmark_indices, axis=0)
        self.extend_state(sensor_observations)
        for observation, landmark_id in zip(sensor_observations, new_landmark_ids):
            self.extend_covariance(observation)
            self.landmark_id_lookup[landmark_id] = len(self.landmark_id_lookup)
        return sensor_observations, new_landmark_ids

    def twist_callback(self, msg: Twist):
        r_velocity = math.hypot(msg.linear.x, msg.linear.y)
        self.twist_history = np.append(self.twist_history, np.array([[r_velocity, msg.angular.z, time.time()]]))

    def image_callback(self, msg: AprilTagDetectionArray):
        # Compute pose delta by integrating twist commands since the last image
        target_twist = np.where(
            np.logical_and(
                self.twist_history[:, 2] >= self.last_image_timestamp,
                self.twist_history[:, 2] <= time.time()))
        self.last_image_timestamp = time.time()
        pose_delta = np.array([
            np.trapz(target_twist[:, 0], target_twist[:, 2]),  # range
            np.trapz(target_twist[:, 1], target_twist[:, 2]),  # bearing
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
        ))

        new_landmark_observations, new_landmark_ids = self.extend(unknown_landmark_observations, unknown_landmark_ids)
        landmark_observations = np.concatenate((known_landmark_observations, new_landmark_observations), axis=0)
        landmark_ids = known_landmark_ids + new_landmark_ids
        self.predict(pose_delta)
        self.update(landmark_observations, landmark_ids)


if __name__ == '__main__':
    slam_node = SLAMNode()
    rospy.init_node('slam_node')
    rospy.subscriber('/twist', Twist, slam_node.twist_callback, queue_size=1)
    rospy.subscriber('/apriltag_detection_array', AprilTagDetectionArray, slam_node.image_callback, queue_size=1)
    rospy.spin()
