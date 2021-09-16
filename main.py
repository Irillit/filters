from filters import Filter
from edge_detection.edge_detector import EdgeDetector


if __name__ == "__main__":
    img_filter = Filter('abscls.jpg', EdgeDetector("prewitt"))
    img_filter.apply()
    img_filter.print_transformed_image()
