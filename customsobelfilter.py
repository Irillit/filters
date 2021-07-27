import numpy as np
from convolutions import Convolutions
import time


class CustomSobelFilter:

    def __init__(self):
        self.filter1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.filter2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def apply(self, array):
        start = time.time()
        output1 = np.ones(array.shape)
        output1[:, :, 0] = Convolutions.conv2d(array[:, :, 0], self.filter1, mode="same")
        output1[:, :, 1] = output1[:, :, 0]
        output1[:, :, 2] = output1[:, :, 0]

        output2 = np.ones(array.shape)
        output2[:, :, 0] = Convolutions.conv2d(array[:, :, 0], self.filter2, mode="same")
        output2[:, :, 1] = output2[:, :, 0]
        output2[:, :, 2] = output2[:, :, 0]

        output = np.abs(output1) + np.abs(output2)

        duration = time.time() - start
        print(f"Duration: {duration}")
        return np.array(output, np.uint8)
