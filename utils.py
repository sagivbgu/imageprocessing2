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
        self.img_height = img_height
        self.img_width = img_width
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        r_max = self._transform_r(img_diagonal)
        theta_max = self._transform_theta(math.pi * 2)
        self.mat = [0 for _ in range(r_max) for _ in range(theta_max)]

    # Division by 2: Discretisize the "hough matrix".
    # Any two different lines or circles may intersect, but not coincide (they should be at least two pixels apart).
    def _transform_r(self, r):
        return r // 2  # Operator // is floor division, returns int

    def _transform_theta(self, theta):
        return theta // (self.img_height + self.img_width)  # TODO: Check this formula

    def increment(self, r, theta):
        self.mat[r][theta] += 1

    def get(self, r, theta):
        return self.mat[r][theta]


class HoughMatrix3D:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        a_max = self._transform_a(img_width)
        b_max = self._transform_b(img_height)
        r_max = self._transform_r(img_diagonal)
        self.mat = [0 for _ in range(r_max) for _ in range(b_max) for _ in range(a_max)]

    # Division by 2: Discretisize the "hough matrix".
    # Any two different lines or circles may intersect, but not coincide (they should be at least two pixels apart).
    def _transform_a(self, a):
        return a // 2

    def _transform_b(self, b):
        return b // 2

    def _transform_r(self, r):
        return r // 2

    def increment(self, a, b, r):
        a = self._transform_a(a)
        b = self._transform_b(b)
        r = self._transform_r(r)
        self.mat[a][b][r] += 1

    def get(self, a, b, r):
        a = self._transform_a(a)
        b = self._transform_b(b)
        r = self._transform_r(r)
        return self.mat[a][b][r]
