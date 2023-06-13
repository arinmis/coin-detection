import numpy as np

def dilate_image(image, kernel_size, iterations):
    height, width = image.shape[:2]
    dilated_image = np.zeros_like(image)

    # Create Kernel
    kernel = np.ones(kernel_size, dtype=np.uint8)

    # Iteration
    for _ in range(iterations):
        for i in range(height):
            for j in range(width):
                if image[i, j] != 0:
                    dilated_image[i:i + kernel_size[0], j:j + kernel_size[1]] = 255

    return dilated_image