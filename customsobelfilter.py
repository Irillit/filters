import numpy as np


class CustomSobelFilter:

    def __init__(self):
        self.filter1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.filter2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def conv2d(self, array, kernel):
        output = np.ones(array.shape)

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                acc = 0
                for ik in range(len(kernel)):
                    for jk in range(len(kernel[0])):
                        i_corr = i - ik
                        j_corr = j - jk
                        if i_corr >= 0 and j_corr >= 0:
                            if i_corr < array.shape[0] and j_corr < array.shape[1]:
                                acc += array[i_corr][j_corr] * kernel[ik][jk]
                output[i][j] = acc

        return output

    def apply_grayscale(self, array):
        output1 = np.ones(array.shape)
        output1[:, :, 0] = self.conv2d(array[:, :, 0], self.filter1)
        output1[:, :, 1] = output1[:, :, 0]
        output1[:, :, 2] = output1[:, :, 0]

        output2 = np.ones(array.shape)
        output2[:, :, 0] = self.conv2d(array[:, :, 0], self.filter2)
        output2[:, :, 1] = output2[:, :, 0]
        output2[:, :, 2] = output2[:, :, 0]

        output = np.ones(array.shape)
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                output[x][y] = abs(output1[x][y]) + abs(output2[x][y])

        return np.array(output, np.uint8)
