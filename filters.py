from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from trivialfilters import TrivialFilters
from customsobelfilter import CustomSobelFilter
from convolutions import Convolutions


def sepia(gray_array):
    normalized_gray = np.array(gray_array, np.float32) / 255.0

    sepia = np.ones(gray_array.shape)
    sepia[:, :, 0] *= 255 * normalized_gray[:, :, 0]
    sepia[:, :, 1] *= 204 * normalized_gray[:, :, 1]
    sepia[:, :, 2] *= 153 * normalized_gray[:, :, 2]

    return np.array(sepia, np.uint8)


def print_image(array):
    image = Image.fromarray(array)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    array = np.array(Image.open('skins-dva.jpg'))
    print(array.shape)
    gray_array = TrivialFilters.grayscale(array)
    sobel = CustomSobelFilter()
    sobel_array = sobel.apply(gray_array)
    """conv = Convolutions()
    filter1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    arr = conv.conv2d_same(array[:, :, 0], filter1)"""
    print_image(sobel_array)
