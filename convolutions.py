import numpy as np


class Convolutions:

    @staticmethod
    def conv2d(array, kernel, mode="valid"):
        new_array = np.zeros((array.shape[0] + 2, array.shape[1] + 2))
        new_array[1:array.shape[0] + 1, 1:array.shape[1] + 1] = array
        output = np.zeros(new_array.shape)
        for i in range(1, array.shape[0]):
            for j in range(1, array.shape[1]):
                fragment = new_array[i - 1: i + 2, j - 1: j + 2]
                output[i][j] = np.sum(np.multiply(fragment, kernel))
        if mode == "same":
            return output[1:array.shape[0] + 1, 1:array.shape[1] + 1]
        return output