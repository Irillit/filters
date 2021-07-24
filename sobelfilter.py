import numpy as np
from scipy.signal import convolve2d


class SobelFilter:
    def __init__(self):
        self.filter1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.filter2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def edge_detection(self, array):
        output1 = convolve2d(array, self.filter1, boundary='symm', mode='same')
        output2 = convolve2d(array, self.filter2, boundary='symm', mode='same')

        output = np.ones(array.shape)
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                output[x][y] = abs(output1[x][y]) + abs(output2[x][y])
        return output

    def apply(self, array):
        output = np.ones(array.shape)
        output[:, :, 0] = self.edge_detection(array[:, :, 0])
        output[:, :, 1] = self.edge_detection(array[:, :, 1])
        output[:, :, 2] = self.edge_detection(array[:, :, 2])

        return np.array(output, np.uint8)

    def apply_grayscale(self, gray_array):
        output = np.ones(gray_array.shape)
        output[:, :, 0] = self.edge_detection(gray_array[:, :, 0])
        output[:, :, 1] = output[:, :, 0]
        output[:, :, 2] = output[:, :, 2]
        return np.array(output, np.uint8)
