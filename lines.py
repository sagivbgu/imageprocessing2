import numpy as np
import cv2 as cv
import math


class HoughMatrix2D:
    def __init__(self, img_height, img_width, img):
        self._img = img

        self._threshold = 0

        self._img_height = img_height
        self._img_width = img_width
        self.img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        self.rho_quanta = 1
        self.rho_min = self._transform_rho(0)
        self.rho_max = self._transform_rho(self.img_diagonal)

        # This was too verbose
        # self.theta_quanta = (math.pi / (self._img_height + self._img_width))
        self.theta_quanta = math.pi / 90

        self.theta_min = self._transform_theta(- math.pi)
        self.theta_max = self._transform_theta(math.pi)

        self._mat = np.zeros((self.rho_max, self.theta_max))
        #print("rho_max = {0} | rho_min = {1} | theta_max = {2} | theta_min = {3}".format(self.rho_max, self.rho_min,
         #                                                                                self.theta_max, self.theta_min))

    def increment_by_x_y(self, x, y):
        for theta in range(self.theta_max):
            inv_theta = self._inverse_theta(theta)
            rho = self._calc_rho(x, y, inv_theta)
            rho = self._transform_rho(rho)

            self._mat[rho][theta] += 1

    def get(self, rho, theta):
        rho = self._transform_rho(rho)
        theta = self._transform_theta(theta)
        return self._mat[rho][theta]

    def get_all_above_threshold(self):
        self._calc_threshold()
        return [(self._inverse_rho(rho), self._inverse_theta(theta)) for theta in range(self.theta_max) for rho in range(self.rho_max) if
                self._mat[rho][theta] >= self._threshold]

    def _transform_rho(self, rho):
        return int(rho // 2)

    def _inverse_rho(self, index):
        return index * 2

    def _transform_theta(self, theta):
        new_theta = (theta + math.pi) / self.theta_quanta
        return math.floor(new_theta)

    def _inverse_theta(self, index):
        return (index * self.theta_quanta) - math.pi

    def _calc_rho(self, x, y, theta):
        return x * math.cos(theta) + y * math.sin(theta)

    def _calc_threshold(self):
        # TODO: maybe better than this
        number_of_white_pixels = np.count_nonzero(self._img)
        percentage_of_most_dominant_line = self._mat.max() / number_of_white_pixels
        # self._threshold = self._mat.max() * 1 / 4
        self._threshold = percentage_of_most_dominant_line * self._mat.max()


def detect_lines(img):
    height, width = img.shape

    hough_mat = build_hough_mat(img)

    lines_coordinates = get_lines_coordinates(hough_mat, img)

    lines_coordinates = eliminate_lines_outside_the_image(lines_coordinates, img)

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


def eliminate_lines_outside_the_image(lines, img):
    height, width = img.shape

    new_lines = []

    for line in lines:
        (p1, p2) = line
        x1, y1 = p1
        m, c = calc_slope_and_intercept(p1, p2)

        # slope is infinity
        if m is None:
            intercept_with_x = x1
            intercept_with_bottom_side = x1
        else:
            intercept_with_x = -c / m
            intercept_with_bottom_side = m * (height - 1) + c

        print("line between p1={0} -> p2={1}".format(p1,p2))
        print("\tm={0} | c={1}".format(m,c))
        print("\twith_x={0}".format(intercept_with_x))
        print("\tbottom_side={0}".format(intercept_with_bottom_side))

        if (intercept_with_x >= 0) and (intercept_with_x < width):
            new_lines.append(line)
        elif (c >= 0) and (c < height):
            new_lines.append(line)
        elif (intercept_with_bottom_side >= 0) and (intercept_with_bottom_side < width):
            new_lines.append(line)
        else:
            print("!!!ELIMINATED!!!")
        print()
    return new_lines


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
    hough_mat = HoughMatrix2D(height, width, img)

    for y in range(height):
        for x in range(width):
            # consider only white pixels
            if img[y][x] == 0:
                continue

            hough_mat.increment_by_x_y(x, y)

    return hough_mat


def calc_line_coordinates(rho, theta, width, height):
    #print("rho = {0} | theta = {1}".format(rho, theta))
    a = np.cos(theta)
    b = np.sin(theta)
    #print("a = {0} | b = {1}".format(a,b ))

    x0 = rho * a
    y0 = rho * b
    #print("x0= {0} | y0= {1}".format(x0, y0))

    x1 = int(x0 + (2 * width * -b))
    y1 = int(y0 + 2 * height * a)
    #print("x1= {0} | y1= {1}".format(x1, y1))

    x2 = int(x0 - (2 * width * -b))
    y2 = int(y0 - 2 * height * a)
    #print("x2= {0} | y2= {1}".format(x2, y2))

    #print()
    return (x1, y1), (x2, y2)


def adjust_coordinates(x, y, width, height):
    x = abs(x - (width - 1))
    y = abs(y - (height - 1))
    return x, y


def print_lines(lines):
    for line in lines:
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
        print("x1= {0} | y1= {1}".format(x1, y1))
        print("x2= {0} | y2= {1}".format(x2, y2))
        print()