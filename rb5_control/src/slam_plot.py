import matplotlib.pyplot as plt
import numpy as np
from slam import SQUARE_PATH, SQUARE_LANDMARKS, OCTAGON_PATH, OCTAGON_LANDMARKS


def plot_results(path: np.ndarray, landmarks: np.ndarray, title: str, data: np.ndarray = None):
    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    if data is not None:
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    plot_results(SQUARE_PATH, SQUARE_LANDMARKS, "Pose Estimation - Square Path")
    plot_results(OCTAGON_PATH, OCTAGON_LANDMARKS, "Ground Truth - Octagon Path")
