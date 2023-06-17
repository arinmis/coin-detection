import numpy as np
import cv2


def find_contours(binary):
    contours = []
    visited = np.zeros(binary.shape, dtype=bool)

    for i in range(1, binary.shape[0] - 1):
        for j in range(1, binary.shape[1] - 1):
            if binary[i, j] == 255 and not visited[i, j]:
                contour = []
                stack = [(i, j)]

                while stack:
                    x, y = stack.pop()
                    visited[x, y] = True

                    neighbors = get_neighbors(binary, visited, x, y)
                    for neighbor in neighbors:
                        nx, ny = neighbor
                        stack.append((nx, ny))
                        visited[nx, ny] = True

                    if not any(get_neighbors(binary, visited, x, y)):
                        contour.append((x, y))

                contours.append(contour)

    return contours

def get_neighbors(binary, visited, x, y):
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if nx >= 0 and nx < binary.shape[0] and ny >= 0 and ny < binary.shape[1] and binary[nx, ny] == 255 and not visited[nx, ny]:
            neighbors.append((nx, ny))

    return neighbors

def draw_contours(image, contours, color=(0, 255, 0)):
    for contour in contours:
        contour = np.array(contour)
        contour = contour.squeeze().reshape(-1, 2)
        for i in range(len(contour)):
            pt1 = tuple(contour[i])
            pt2 = tuple(contour[(i + 1) % len(contour)])
            draw_line(image, pt1, pt2, color)



def draw_line(image, pt1, pt2, color):
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate the distance between the two points
    distance = max(abs(x2 - x1), abs(y2 - y1))

    if distance == 0:
        # Points are the same, no need to draw a line
        return

    # Determine the x and y steps based on the distance
    x_step = (x2 - x1) / distance if distance != 0 else 0
    y_step = (y2 - y1) / distance if distance != 0 else 0

    # Drawing process
    for i in range(int(distance)):
        x = int(x1 + i * x_step)
        y = int(y1 + i * y_step)

        # Check if the pixel coordinates are within the image dimensions
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = color