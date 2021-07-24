from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from customsobelfilter import CustomSobelFilter


def grayscale(array):
    gray = np.round(0.2989 * array[:, :, 0] + 0.5870 * array[:, :, 1] + 0.1140 * array[:, :, 2]).astype(np.uint8)
    new_array = np.stack((gray, gray, gray), axis=2)
    return new_array


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
    array = np.array(Image.open('image.jpg'))
    print(array.shape)
    gray_array = grayscale(array)
    custom = CustomSobelFilter()
    custom_sobel = custom.apply_grayscale(gray_array)
    print_image(custom_sobel)
