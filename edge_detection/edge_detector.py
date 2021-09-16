from strategy import Strategy

import numpy as np
from convolutions import Convolutions
from edge_detection.kernels import Kernels
import time


class EdgeDetector(Strategy):

    def __init__(self, filter_type):
        if filter_type == "prewitt":
            self.kernel1, self.kernel2 = Kernels.get_prewitt_kernels()
        else:
            self.kernel1, self.kernel2 = Kernels.get_sobel_kernels()

    def apply(self, array):
        start = time.time()
        output1 = np.ones(array.shape)
        output1[:, :, 0] = Convolutions.conv2d(array[:, :, 0], self.kernel1, mode="same")
        output1[:, :, 1] = output1[:, :, 0]
        output1[:, :, 2] = output1[:, :, 0]

        output2 = np.ones(array.shape)
        output2[:, :, 0] = Convolutions.conv2d(array[:, :, 0], self.kernel2, mode="same")
        output2[:, :, 1] = output2[:, :, 0]
        output2[:, :, 2] = output2[:, :, 0]

        output = np.abs(output1) + np.abs(output2)

        duration = time.time() - start
        print(f"Duration: {duration}")
        return np.array(output, np.uint8)