import cv2
import numpy as np

def canny_edge_detection(image):

    # Calculate gradient in x and y directions
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    high_threshold = 30
    low_threshold = 150

    strong_edges = np.where(gradient_magnitude > high_threshold, 255, 0).astype(np.uint8)
    weak_edges = np.where((gradient_magnitude <= high_threshold) & (gradient_magnitude > low_threshold), 255, 0).astype(np.uint8)

    edges = np.zeros_like(strong_edges)
    edges[strong_edges > 0] = 255

    # Define the neighborhood kernel
    kernel = np.ones((3, 3), dtype=np.uint8)

    # Perform dilation to connect weak edges to strong edges
    dilated_weak_edges = cv2.dilate(weak_edges, kernel, iterations=1)

    # Determine final edges by suppressing weak edges not connected to strong edges
    final_edges = np.where((edges > 0) | (dilated_weak_edges > 0), 255, 0).astype(np.uint8)

    return final_edges