from strategy import Strategy

import math
import numpy as np


class MedianFilter(Strategy):

    def __init__(self, size):
        self.size = size

    def median(self, array):
        output = np.ones(array.shape)
        size = 4
        window_arr_size = size ** 2
        window = np.ones(window_arr_size)
        edge = math.floor(size / 2.0)

        for x in range(edge, array.shape[0] - edge):
            for y in range(edge, array.shape[1] - edge):
                for m in range(0, size):
                    mi = m * size
                    window[mi:mi + size] = array[x + m - edge][y + m - edge]

                window.sort()
                output[x][y] = window[math.floor(window_arr_size / 2)]
        return output

    def apply(self, array):
        output = np.ones(array.shape)
        output[:, :, 0] = self.median(array[:, :, 0])
        output[:, :, 1] = self.median(array[:, :, 1])
        output[:, :, 2] = self.median(array[:, :, 2])
        return np.array(output, np.uint8)
