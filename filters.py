from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from trivialfilters import TrivialFilters
from edge_detection.edge_detector import EdgeDetector
from medianfilter import MedianFilter


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
    median = MedianFilter(5)
    sobel_array = median.apply(array)
    print_image(sobel_array)
