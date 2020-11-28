import json
from math import sqrt, floor, sin, cos, pi
import cv2 as cv

from utils import get_pixels_with_value

DETECTED_CIRCLE_COLOR = (255, 255, 0)

"""
Parametric equation of a circle:
x = a + rcos(theta)
y = b + rsin(theta)
0 <= theta < 2 * pi
"""


class HoughMatrix3D:
    def __init__(self, img_height, img_width):
        img_diagonal = sqrt(img_height ** 2 + img_width ** 2)
        self._r_max = self._transform_r(img_diagonal)
        self._a_max = self._transform_a(img_width)
        self._b_max = self._transform_b(img_height)
        self._theta_slices = 100
        self._theta_quanta = 2 * pi / self._theta_slices
        self._theta_max = self._transform_theta(2 * pi)
        self._mat = [[[0 for _ in range(self._b_max)] for _ in range(self._a_max)] for _ in range(self._r_max)]

    def increment_by_x_y(self, x, y):
        for r_index in range(self._r_max):
            r = self._inverse_r(r_index)
            for theta_index in range(self._theta_max):
                theta = self._inverse_theta(theta_index)
                a = x - int(r * cos(theta))
                b = y - int(r * sin(theta))
                a_index = self._transform_a(a)
                b_index = self._transform_b(b)
                if 0 <= a_index < self._a_max and 0 <= b_index < self._b_max:
                    self._mat[r_index][a_index][b_index] += 1

    def get_all_above_threshold(self):
        circles = []
        for r_index in range(self._r_max):
            r = self._inverse_r(r_index)
            for a_index in range(self._a_max):
                a = self._inverse_a(a_index)
                for b_index in range(self._b_max):
                    b = self._inverse_b(b_index)
                    value = self._mat[r_index][a_index][b_index]
                    if self._is_above_threshold(r, value):
                        new_circle = (a, b, r)
                        circles.append((new_circle, value))
        return circles

    def _is_above_threshold(self, radius, cell_value):
        if radius <= 8:
            return False
        threshold = 0.3  # Recommended so far: 0.3 ~ 0.33 / 0.45 ~ 0.6
        return cell_value / self._theta_slices >= threshold

    def dump(self, filename):
        with open(filename, "w") as f:
            json.dump(self._mat, f)

    def load(self, filename):
        with open(filename, "r") as f:
            self._mat = json.load(f)

    def _transform_r(self, r):
        return int(r)

    def _transform_a(self, a):
        return int(a)

    def _transform_b(self, b):
        return int(b)

    def _transform_theta(self, theta):
        new_theta = theta / self._theta_quanta
        return floor(new_theta)

    def _inverse_r(self, index):
        return index

    def _inverse_a(self, index):
        return index

    def _inverse_b(self, index):
        return index

    def _inverse_theta(self, index):
        return index * self._theta_quanta


def detect_circles(image, debug=False):  # TODO: Remove debug feature
    # Image: The binary image to find circles in it
    h, w = image.shape
    hough_matrix = HoughMatrix3D(h, w)

    if debug:
        hough_matrix.load("mat.json")
    else:
        for (x, y) in get_pixels_with_value(image):
            hough_matrix.increment_by_x_y(x, y)
        hough_matrix.dump("mat.json")

    circles = hough_matrix.get_all_above_threshold()
    return get_unique_circles(circles)


def get_unique_circles(circles):
    final_circles = []
    circles = sorted(circles, key=lambda i: i[1])
    while circles:
        top = circles.pop()
        final_circles.append(top[0])
        circles = [c for c in circles if not close_points(top[0], c[0])]
    return final_circles


def close_points(p1, p2):
    return all(abs(p2[i] - p1[i]) <= 2 for i in range(3))


def draw_circles_on_img(circles, img):
    for (a, b, r) in circles:
        cv.circle(img, (a, b), r, DETECTED_CIRCLE_COLOR, 1)
