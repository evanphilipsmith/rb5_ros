import matplotlib.pyplot as plt
import numpy as np
from slam import SQUARE_PATH, SQUARE_LANDMARKS, OCTAGON_PATH, OCTAGON_LANDMARKS

def plot_results(path: np.ndarray, landmarks: np.ndarray):
    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()

if __name__ == '__main__':
    # plot_results(SQUARE_PATH, SQUARE_LANDMARKS)
    plot_results(OCTAGON_PATH, OCTAGON_LANDMARKS)
