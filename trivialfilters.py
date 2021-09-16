from strategy import Strategy
import numpy as np


class Grayscale(Strategy):

    def apply(self, array):
        gray = np.round(0.2989 * array[:, :, 0] + 0.5870 * array[:, :, 1] + 0.1140 * array[:, :, 2]).astype(np.uint8)
        new_array = np.stack((gray, gray, gray), axis=2)
        return new_array


class Sepia(Strategy):

    def apply(self, array):
        grayscale = Grayscale()
        gray_array = grayscale.apply(array)
        return self.sepia(gray_array)

    def sepia(self, gray_array):
        normalized_gray = np.array(gray_array, np.float32) / 255.0

        sepia = np.ones(gray_array.shape)
        sepia[:, :, 0] *= 255 * normalized_gray[:, :, 0]
        sepia[:, :, 1] *= 204 * normalized_gray[:, :, 1]
        sepia[:, :, 2] *= 153 * normalized_gray[:, :, 2]

        return np.array(sepia, np.uint8)


class Inversion(Strategy):

    def apply(self, array):
        return 255 - array
