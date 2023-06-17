import numpy as np

def rgb2gray(image):
    if image is None:
        raise ValueError("Invalid image")

    # Get the dimensions of the input image
    height, width, _ = image.shape

    # Perform the RGB to grayscale conversion
    gray_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray = 0.2989 * r + 0.587 * g + 0.114 * b
            gray_image[i, j] = gray

    return gray_image


def bgr2rgb(image):
    # Create an empty array with the same shape as the input image
    height, width, channels = image.shape
    converted_image = np.zeros_like(image)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Swap the values of the blue and red channels
            converted_image[y, x, 0] = image[y, x, 2]  # Blue channel
            converted_image[y, x, 1] = image[y, x, 1]  # Green channel
            converted_image[y, x, 2] = image[y, x, 0]  # Red channel

    return converted_image