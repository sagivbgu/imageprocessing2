import numpy as np
import cv2 as cv
import math


class HoughMatrix2D:
    def __init__(self, img_height, img_width, rho_quanta=2, theta_quanta=(math.pi / 90)):
        self._threshold = 0

        self._img_height = img_height
        self._img_width = img_width
        self.img_diagonal = math.ceil(math.sqrt(img_height ** 2 + img_width ** 2))

        self.rho_quanta = rho_quanta
        self.rho_min = self._transform_rho(- self.img_diagonal)
        self.rho_max = self._transform_rho(self.img_diagonal)

        self.theta_quanta = theta_quanta
        self.theta_min = self._transform_theta(- math.pi)
        self.theta_max = self._transform_theta(math.pi)

        self._mat = np.zeros((self.rho_max, self.theta_max))

    def increment_by_x_y(self, x, y):
        for theta in range(self.theta_max):
            inv_theta = self._inverse_theta(theta)
            rho = self._calc_rho(x, y, inv_theta)
            rho = self._transform_rho(rho)
            self._mat[rho][theta] += 1

    def get_all_above_threshold(self):
        self._calc_threshold()
        return [(self._inverse_rho(rho), self._inverse_theta(theta)) for theta in range(self.theta_max) for rho in range(self.rho_max) if
                self._mat[rho][theta] >= self._threshold]

    def _transform_rho(self, rho):
        return int((rho + self.img_diagonal) // self.rho_quanta)

    def _inverse_rho(self, index):
        return ((index * 2) - self.img_diagonal)

    def _transform_theta(self, theta):
        new_theta = (theta + math.pi) / self.theta_quanta
        return math.floor(new_theta)

    def _inverse_theta(self, index):
        return (index * self.theta_quanta) - math.pi

    def _calc_rho(self, x, y, theta):
        return x * math.cos(theta) + y * math.sin(theta)

    def _calc_threshold(self):
        # TODO: maybe better than this
        self._threshold = np.amax(self._mat) * 1 / 2


def detect_lines(img):
    hough_mat = build_hough_mat(img)

    lines_coordinates = get_lines_coordinates(hough_mat, img)

    lines_coordinates = calc_edges_points_of_lines(lines_coordinates, img)

    img_with_lines = draw_lines_on_img(lines_coordinates, img)

    return lines_coordinates, img_with_lines


def draw_lines_on_img(coordinates, img):
    new_img = img.copy()
    new_img = cv.cvtColor(new_img, cv.COLOR_GRAY2RGB)

    for (p1, p2) in coordinates:
        cv.line(new_img, p1, p2, color=(255, 255, 0))

    return new_img


def get_lines_coordinates(hough_mat, img):
    height, width = img.shape

    lines = hough_mat.get_all_above_threshold()
    lines_coordinates = []

    for line in lines:
        p1, p2 = calc_line_coordinates(line[0], line[1], width, height)
        lines_coordinates.append((p1, p2))

    return lines_coordinates


def calc_slope_and_intercept(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # slope is infinity
    if x2 == x1:
        return None, None

    m = (y2 - y1) / (x2 - x1)
    c = y2 - m * x2

    return m, c


def build_hough_mat(img):
    height, width = img.shape
    hough_mat = HoughMatrix2D(height, width)

    for y in range(height):
        for x in range(width):
            # consider only white pixels
            if img[y][x] == 0:
                continue

            hough_mat.increment_by_x_y(x, y)

    return hough_mat


def calc_line_coordinates(rho, theta, width, height):
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = rho * a
    y0 = rho * b

    x1 = int(x0 + (2 * width * -b))
    y1 = int(y0 + 2 * height * a)

    x2 = int(x0 - (2 * width * -b))
    y2 = int(y0 - 2 * height * a)

    return (x1, y1), (x2, y2)


def calc_edges_points_of_lines(lines, img):
    height, width = img.shape

    lines_edges_points = []

    for line in lines:
        (p1, p2) = line
        x1, y1 = p1
        m, c = calc_slope_and_intercept(p1, p2)

        # if vertical
        if m is None:
            if is_intercept_valid(x1, width):
                a = (x1, 0)
                b = (x1, height - 1)
            else:
                continue

        # if horizontal
        elif m == 0:
            if is_intercept_valid(c, height):
                a = (0, c)
                b = (width - 1, c)
            else:
                continue

        else:
            intercept_with_x = -c / m  # y = 0
            intercept_with_bottom_side = ((height - 1) - c) / m  # y = (height - 1)
            intercept_with_right_side = m * (width - 1) + c  # x = (width - 1)

            if is_intercept_valid(intercept_with_x, width):
                # TOP and LEFT
                if is_intercept_valid(c, height):
                    a = (intercept_with_x, 0)
                    b = (0, c)
                # TOP and RIGHT
                elif is_intercept_valid(intercept_with_right_side, height):
                    a = (intercept_with_x, 0)
                    b = (width - 1, intercept_with_right_side)
                # TOP and BOTTOM
                elif is_intercept_valid(intercept_with_bottom_side, width):
                    a = (intercept_with_x, 0)
                    b = (intercept_with_bottom_side, 0)
                else:
                    continue
            elif is_intercept_valid(intercept_with_bottom_side, width):
                # BOTTOM and LEFT
                if is_intercept_valid(c, height):
                    a = (intercept_with_bottom_side, height - 1)
                    b = (0, c)
                # BOTTOM and RIGHT
                if is_intercept_valid(intercept_with_right_side, height):
                    a = (intercept_with_bottom_side, height - 1)
                    b = (width - 1, intercept_with_right_side)
                else:
                    continue
            elif is_intercept_valid(c, height):
                # LEFT and RIGHT
                if is_intercept_valid(intercept_with_right_side, height):
                    a = (0, c)
                    b = (width - 1, intercept_with_right_side)
                else:
                    continue
            else:
                continue

        x1, y1 = a
        x2, y2 = b

        a = (int(x1), int(y1))
        b = (int(x2), int(y2))

        lines_edges_points.append((a, b))

    return lines_edges_points


def is_intercept_valid(x, max_val):
    return (x >= 0) and (x < max_val)
