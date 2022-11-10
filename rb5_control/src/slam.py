import numpy as np
import math

state_vector = np.array([0.0, 0.0, 0.0])
INITIAL_COVARIANCE = np.diag([0.005, 0.005, 0.001]) ** 2
NUM_LANDMARK_PARAMS = 2
VALID_LANDMARK_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RANGE_UNCERTAINTY = 0.1  # meters
BEARING_UNCERTAINTY = math.pi / 180 # radians
SENSOR_NOISE = np.diag([RANGE_UNCERTAINTY, BEARING_UNCERTAINTY]) ** 2

def predict_vehicle_covariance(current_covariance, current_state, pose_delta):
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

def predict_vehicle_state(current_state, pose_delta):
    linear_pose_delta, angular_pose_delta = pose_delta
    _, _, orientation = current_state
    predicted_state = current_state + np.array([
        linear_pose_delta * math.cos(orientation),
        linear_pose_delta * math.sin(orientation),
        angular_pose_delta
    ])
    return predicted_state

def predict_state(current_state, vehicle_pose_delta):
    # Landmark state doesn't change since we assume landmarks are stationary.
    predicted_vehicle_state = predict_vehicle_state(current_state[:3], vehicle_pose_delta)
    predicted_state = np.concatenate((predicted_vehicle_state, current_state[3:]))
    return predicted_state

def predict_covariance(current_covariance, current_state, vehicle_pose_delta):
    # Landmark covariance doesn't change since we assume landmarks are stationary.
    vehicle_covariance = current_covariance[:3, :3]
    landmark_covariance = current_covariance[3:, 3:]
    # vehicle_landmark_correlation = current_covariance[:3, 3:]

    predicted_vehicle_covariance = predict_vehicle_covariance(vehicle_covariance, current_state[:3], vehicle_pose_delta)
    predicted_vehicle_landmark_correlation = (
        current_state[:3].reshape(-1, 1)
        @ current_state[3:].reshape(1, -1))
    predicted_covariance = np.concatenate((
        np.concatenate(
            (predicted_vehicle_covariance, predicted_vehicle_landmark_correlation),
            axis=1),
        np.concatenate(
            (predicted_vehicle_landmark_correlation.T, landmark_covariance),
            axis=1)),
    axis=0)
    return predicted_covariance

def predict(current_state: np.ndarray, current_covariance: np.ndarray, pose_delta: np.ndarray):
    predicted_state = predict_state(current_state, pose_delta)
    predicted_covariance = predict_covariance(current_covariance, current_state, pose_delta)
    return predicted_state, predicted_covariance

def update_vehicle_state(
        predicted_state: np.ndarray,
        landmark_id_lookup: dict,
        predicted_covariance: np.ndarray,
        sensor_observations: np.ndarray,
        landmark_ids: list):
    def landmark_idx_to_state_idx(i: int) -> int:
        return (i * NUM_LANDMARK_PARAMS) + 3
    sensor_observations = sensor_observations.reshape(-1, 2)
    updated_state = predicted_state
    updated_covariance = predicted_covariance
    for x, observation in zip(landmark_ids, sensor_observations):
        landmark_idx = landmark_id_lookup[x]
        landmark_pose_x, landmark_pose_y = updated_state[landmark_idx:landmark_idx+2]
        vehicle_pose_x, vehicle_pose_y, vehicle_orientation = updated_state[:3]
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
            np.zeros((2, updated_state.size-(state_idx+NUM_LANDMARK_PARAMS)))), axis=1)
        s = h_x @ updated_covariance @ h_x.T + SENSOR_NOISE
        kalman_gain = updated_covariance @ h_x.T @ np.linalg.inv(s)
        updated_state = updated_state + (kalman_gain @ innovation)
        updated_covariance = updated_covariance - (kalman_gain @ h_x @ updated_covariance)
    return updated_state, updated_covariance


def extend_covariance(prev_covariance: np.ndarray, vehicle_orientation: float, sensor_observation: np.ndarray):
    n = prev_covariance.shape[0]
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
        np.concatenate((prev_covariance, np.zeros((n, NUM_LANDMARK_PARAMS))), axis=1),
        np.concatenate((np.zeros((NUM_LANDMARK_PARAMS, n)), SENSOR_NOISE), axis=1)),
        axis=0)
    extended_covariance = insertion_jacobian @ template @ insertion_jacobian.T
    return extended_covariance

def extend_state(prev_state: np.ndarray, sensor_observations: np.ndarray):
    if sensor_observations.size == 0:
        return prev_state
    assert sensor_observations.size % NUM_LANDMARK_PARAMS == 0
    vehicle_pose_x, vehicle_pose_y, vehicle_orientation = prev_state
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
    new_state = np.concatenate((prev_state, observations_tf.flatten()))
    return new_state

def extend(
        prev_state: np.ndarray,
        landmark_id_lookup: dict,
        prev_covariance: np.ndarray,
        sensor_observations: np.ndarray,
        new_landmark_ids: list):
    # Remove landmarks that aren't in the set of supported landmarks
    valid_landmark_indices = [i for i, x in enumerate(new_landmark_ids) if x in VALID_LANDMARK_IDS]
    new_landmark_ids = [x for x in new_landmark_ids if x in VALID_LANDMARK_IDS]
    sensor_observations = np.take(sensor_observations, valid_landmark_indices, axis=0)
    extended_state = extend_state(prev_state, sensor_observations)
    c = prev_covariance
    for observation, landmark_id in zip(sensor_observations, new_landmark_ids):
        c = extend_covariance(c, state[2], observation)
        landmark_id_lookup[landmark_id] = len(landmark_id_lookup)
    extended_covariance = c
    return extended_state, landmark_id_lookup, extended_covariance, sensor_observations, new_landmark_ids


# TODO
# Create a twist callback that saves velocities and timestamps to a matrix to be integrated later
# Create a camera callback that:
#   * Computes a pose delta by integrating velocities from last image timestamp to current image timestamp
#   * Organizes april tag inputs into a list of range/bearing measurements
#   * Runs extend(), predict(), update()

if __name__ == '__main__':
    # test cases
    state = np.array([0.0, 0.0, 0.0])
    landmark_ids = {}
    covariance = INITIAL_COVARIANCE
    # observations = np.array([1.0, 0.0, 0.5, math.pi/4])
    # print(extend_state(state, observations))


    pose_delta = np.array([0.001, 0.0])
    print(covariance)

    # covariance = predict_vehicle_covariance(covariance, state, pose_delta)
    # print(covariance)
    # state = predict_vehicle_state(state, pose_delta)
    # print(state)

    obs = np.array([
        [2, 0],
        [5, 0],
    ])
    ids = [1, 2]
    # obs = np.array([])
    # ids = []
    state, landmark_ids, covariance, obs, ids = extend(state, landmark_ids, covariance, obs, ids)
    print(state)
    print(covariance)
    
    state, covariance = predict(state, covariance, pose_delta)
    print(state)
    print(covariance)

    state, covariance = update_vehicle_state(state, landmark_ids, covariance, obs, ids)
    print(state)
    print(covariance)
