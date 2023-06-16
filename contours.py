import numpy as np

def draw_contours(image, contours, color=(0, 255, 0)):
    for contour in contours:
        for i in range(len(contour)):
            pt1 = tuple(contour[i][0])
            pt2 = tuple(contour[(i + 1) % len(contour)][0])
            draw_line(image, pt1, pt2, color)


def draw_line(image, pt1, pt2, color):
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate the distance between the two points
    distance = max(abs(x2 - x1), abs(y2 - y1))

    # Determine the x and y steps based on the distance
    x_step = (x2 - x1) / distance
    y_step = (y2 - y1) / distance

    # Drawing process
    for i in range(int(distance)):
        x = int(x1 + i * x_step)
        y = int(y1 + i * y_step)

        # Draw three pixels
        image[y, x] = color
        image[y + 1, x] = color
        image[y + 2, x] = color