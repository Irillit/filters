import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class Filter:

    def __init__(self, image_name, concrete_filter):
        self.image_name = image_name
        self.original = None
        self.transformed = None
        self.filter = concrete_filter

    def get_content(self):
        self.original = np.array(Image.open(self.image_name))

    def apply(self):
        self.get_content()
        self.transformed = self.filter.apply(self.original)

    def print_transformed_image(self):
        output_image = Image.fromarray(self.transformed)
        plt.imshow(output_image)
        plt.show()




