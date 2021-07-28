import numpy as np


class Kernels:

    @staticmethod
    def get_prewitt_kernels():
        kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        return kernel1, kernel2

    @staticmethod
    def get_sobel_kernels():
        kernel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return kernel1, kernel2