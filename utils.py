import math


def get_pixels_with_value(image, val):
    """
    Get all the pixels in the image with intensity value as val.
    :param image: The image to scan
    :param val: The intensity value to look for
    :return: A generator of pairs (x, y) representing the pixel's location in the image
    """
    h, w = image.shape
    return [(x, y) for y in range(h) for x in range(w) if image[y][x] == val]


def generate_pair(image, val=255):
    """
    Generate pairs of pixels in an image with intensity value as val.
    :param image: The image to scan
    :param val: The intensity value to look for
    :return: A generator of pairs of pairs (x, y) representing the pixel's location in the image
    """
    target_pixels = get_pixels_with_value(image, val)

    for i in range(len(target_pixels) - 1):
        for j in range(i + 1, len(target_pixels)):
            yield target_pixels[i], target_pixels[j]


def generate_triple(image, val=255):
    """
    Generate triples of pixels in an image with intensity value as val.
    :param image: The image to scan
    :param val: The intensity value to look for
    :return: A generator of triples of pairs (x, y) representing the pixel's location in the image
    """
    target_pixels = get_pixels_with_value(image, val)

    for i in range(len(target_pixels) - 2):
        for j in range(i + 1, len(target_pixels) - 1):
            for k in range(j + 1, len(target_pixels)):
                yield target_pixels[i], target_pixels[j], target_pixels[k]


class HoughMatrix2D:
    def __init__(self, img_height, img_width):
        self._threshold = img_height * img_width // 10  # TODO: Play with this formula! May be just a constant

        self._img_height = img_height
        self._img_width = img_width
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        self._r_max = self._transform_r(img_diagonal)
        self._theta_max = self._transform_theta(math.pi * 2)
        self._mat = [[0 for _ in range(self._r_max)] for _ in range(self._theta_max)]

    def increment(self, r, theta):
        self._mat[r][theta] += 1

    def get(self, r, theta):
        return self._mat[r][theta]

    def get_all_above_threshold(self):
        return [(r, theta) for theta in range(self._theta_max) for r in range(self._r_max) if
                self._mat[r][theta] > self._threshold]

    # Division by 2: Discretisize the "hough matrix".
    # Any two different lines or circles may intersect, but not coincide (they should be at least two pixels apart).
    def _transform_r(self, r):
        return r // 2  # Operator // is floor division, returns int

    def _transform_theta(self, theta):
        return theta // (self._img_height + self._img_width)  # TODO: Check this formula


class HoughMatrix3D:
    def __init__(self, img_height, img_width):
        self._threshold = img_height * img_width // 10  # TODO: Play with this formula! May be just a constant

        self._img_height = img_height
        self._img_width = img_width
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        self._a_max = self._transform_a(img_width)
        self._b_max = self._transform_b(img_height)
        self._r_max = self._transform_r(img_diagonal)
        self._mat = [[[0 for _ in range(self._r_max)] for _ in range(self._b_max)] for _ in range(self._a_max)]

    def increment(self, a, b, r):
        a = self._transform_a(a)
        b = self._transform_b(b)
        r = self._transform_r(r)
        self._mat[a][b][r] += 1

    def get(self, a, b, r):
        a = self._transform_a(a)
        b = self._transform_b(b)
        r = self._transform_r(r)
        return self._mat[a][b][r]

    def get_all_above_threshold(self):
        return [(a, b, r) for r in range(self._r_max) for b in range(self._b_max) for a in range(self._a_max) if
                self._mat[a][b][r] > self._threshold]

    # Division by 2: Discretisize the "hough matrix".
    # Any two different lines or circles may intersect, but not coincide (they should be at least two pixels apart).
    def _transform_a(self, a):
        return a // 2

    def _transform_b(self, b):
        return b // 2

    def _transform_r(self, r):
        return r // 2
