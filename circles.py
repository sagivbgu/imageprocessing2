import math
import cv2 as cv

from utils import get_pixels_with_value

THRESHOLD = 0.8  # Used to choose votes from the hough matrix.
                 # If the amount of votes > the circle's perimeter * THRESHOLD, then draw this circle. Otherwise, assume
                 # it's not a circle but noise and ignore this cell in the hough matrix.
                 # In tests we've done, a value between 0.3~0.6 is optimal most of the times.
MIN_RADIUS = 8  # Circles with radius less than MIN_RADIUS will be considered as noise and ignored.
DETECTED_CIRCLE_COLOR = (0, 255, 255)


class HoughMatrix3D:
    def __init__(self, img_height, img_width):
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)
        self._a_max = img_width
        self._b_max = img_height
        self._r_max = math.floor(img_diagonal)
        self._mat = [[[0 for _ in range(self._r_max)] for _ in range(self._b_max)] for _ in range(self._a_max)]

    def increment_by_x_y(self, x, y):
        for a in range(self._a_max):
            for b in range(self._b_max):
                r = math.sqrt((x - a) ** 2 + (y - b) ** 2)
                r = math.floor(r)
                self._mat[a][b][r] += 1

    def get_all_above_threshold(self):
        return [  # List comprehension
            ((a, b, r), self._mat[a][b][r])
            for r in range(self._r_max)
            for b in range(self._b_max)
            for a in range(self._a_max)
            if self._is_above_threshold(r, self._mat[a][b][r])
        ]

    def _is_above_threshold(self, radius, cell_value):
        if radius <= MIN_RADIUS:  # Don't catch too small circles to reduce noise
            return False
        perimeter = 2 * math.pi * radius
        return cell_value >= perimeter * THRESHOLD


def detect_circles(image):
    # Image: The binary image to find circles in it
    h, w = image.shape
    hough_matrix = HoughMatrix3D(h, w)

    for (x, y) in get_pixels_with_value(image):
        hough_matrix.increment_by_x_y(x, y)

    circles = hough_matrix.get_all_above_threshold()
    circles = [c for c in circles if is_circle_inside_image(c[0], h, w)]  # c[0] is the circle (a, b, r), c[1] is the value of the corresponding cell in the hough matrix
    return get_unique_circles(circles)


def is_circle_inside_image(circle, img_height, img_width):
    a, b, r = circle
    return a + r < img_width and a - r >= 0 and b + r < img_height and b - r >= 0


def get_unique_circles(circles):
    final_circles = []
    circles = sorted(circles, key=lambda i: i[1])
    while circles:
        top = circles.pop()
        final_circles.append(top[0])
        circles = [c for c in circles if not close_points(top[0], c[0])]
    return final_circles


def close_points(p1, p2):
    # Points are considered close if the distance between their a, b and r values is <= 2 pixels
    return all(abs(p2[i] - p1[i]) <= 2 for i in range(3))


def draw_circles_on_img(circles, img):
    if not len(circles):
        return
    for (a, b, r) in circles:
        cv.circle(img, (a, b), r, DETECTED_CIRCLE_COLOR, 1)
