import numpy as np


class TrivialFilters:

    @staticmethod
    def grayscale(array):
        gray = np.round(0.2989 * array[:, :, 0] + 0.5870 * array[:, :, 1] + 0.1140 * array[:, :, 2]).astype(np.uint8)
        new_array = np.stack((gray, gray, gray), axis=2)
        return new_array

    @staticmethod
    def inversion(array):
        output = 255 - array
        return output