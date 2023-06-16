import cv2
import numpy as np

def gkernel(length, sigma):
    """
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, sigma):
    # Create the Gaussian kernel
    kernel = gkernel(kernel_size, sigma)

    # Apply the Gaussian kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred
