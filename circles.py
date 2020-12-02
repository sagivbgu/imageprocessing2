import math
import cv2 as cv

THRESHOLD = 0.5  # Used to choose votes from the hough matrix.
                 # If the amount of votes > the circle's perimeter * THRESHOLD, then draw this circle. Otherwise, assume
                 # it's not a circle but noise and ignore this cell in the hough matrix.
                 # In tests we've done, a value between 0.3~0.6 is optimal in most cases.
MIN_RADIUS = 8  # Circles with radius less than MIN_RADIUS will be considered as noise and ignored.
MIN_DIST = 5  # According to the clarifications any two different circles should be at least five pixels apart
DETECTED_CIRCLE_COLOR = (0, 255, 255)


class HoughMatrix3D:
    """
    Class representing the Hough matrix.
        - by default, we want each a, b and r to have a cell in the matrix.
          We didn't perform a "discretization" to gain much more precise circle detection.
          Instead, we later remove circles which are too close to each other.

        - the range of a is [0 , w), when w = width of the image
        - the range of b is [0 , h), when h = height of the image
        - the range of r is [0 , d), when d = length of the diagonal
        - the (0,0) point is on the Top Left
    """

    def __init__(self, img_height, img_width):
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)
        self._a_max = img_width
        self._b_max = img_height
        self._r_max = math.floor(img_diagonal)
        self._mat = [[[0 for _ in range(self._r_max)] for _ in range(self._b_max)] for _ in range(self._a_max)]

    def increment_by_x_y(self, x, y):
        # Iterate over each possible a and b, calculate the matching radius and increment the number of votes
        for a in range(self._a_max):
            for b in range(self._b_max):
                r = math.sqrt((x - a) ** 2 + (y - b) ** 2)
                r = math.floor(r)
                self._mat[a][b][r] += 1

    def get_all_above_threshold(self):
        # Returns a list of circles, each circle is a tuple of 2 items:
        # circle[0] is the tuple of the circle parameters (a, b, r)
        # circle[1] is the number of votes for this circle in the hough matrix
        return [
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
    """
    Perform circles detection and get all the circles in the image.
    :param image: The binary image to find circles in it
    :return: A list of all the circles detected, each of the format (a, b, r)
    """
    h, w = image.shape
    hough_matrix = HoughMatrix3D(h, w)

    for (x, y) in get_pixels_with_value(image):
        hough_matrix.increment_by_x_y(x, y)

    circles = hough_matrix.get_all_above_threshold()

    # A valid circle should fall entirely within the image
    circles = [c for c in circles if is_circle_inside_image(c[0], h, w)]
    # c[0] is the circle (a, b, r), c[1] is the number of votes for this circle in the hough matrix

    # Any two different circles should be at least five pixels apart
    return get_unique_circles(circles)


def is_circle_inside_image(circle, img_height, img_width):
    a, b, r = circle
    return a + r < img_width and a - r >= 0 and b + r < img_height and b - r >= 0


def get_unique_circles(circles):
    """
    Get only the circles with the highest number of votes among close circles (to reduce noise).
    :param circles: The circles, in the format: ( (a, b, r), votes_in_hough_matrix)
    """
    final_circles = []
    circles = sorted(circles, key=lambda i: i[1])  # Sort the circles by their number of votes
    while circles:
        top = circles.pop()  # Get the circle with the highest number of votes
        final_circles.append(top[0])

        # Remove circles that are too close to the one chosen from the list to process
        circles = [c for c in circles if not close_circles(top[0], c[0])]
    return final_circles


def close_circles(c1, c2):
    # Circles are considered close if the distance between their a, b and r values is <= MIN_DIST pixels
    return all(abs(c2[i] - c1[i]) <= MIN_DIST for i in range(3))


def draw_circles_on_img(circles, img):
    if not len(circles):
        return
    for (a, b, r) in circles:
        cv.circle(img, (a, b), r, DETECTED_CIRCLE_COLOR, 1)


def get_pixels_with_value(image, val=255):
    """
    Get all the pixels in the image with intensity value as val.
    :param image: The image to scan
    :param val: The intensity value to look for
    :return: A generator of pairs (x, y) representing the pixel's location in the image
    """
    h, w = image.shape
    return [(x, y) for y in range(h) for x in range(w) if image[y][x] == val]
