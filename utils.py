import math


def get_pixels_with_value(image, val=255):
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
        self.img_diagonal = math.sqrt(img_height ** 2 + img_width ** 2)

        self.rho_quanta = 1
        self.rho_min = self._transform_rho(- self.img_diagonal)
        self.rho_max = self._transform_rho(self.img_diagonal)

        self.theta_quanta = (math.pi / (self._img_height + self._img_width))
        self.theta_min = self._transform_theta(- math.pi)
        self.theta_max = self._transform_theta(math.pi)
        self._mat = [[0 for _ in range(self.rho_max)] for _ in range(self.theta_max)]

    def increment(self, rho, theta):
        rho = self._transform_rho(rho)
        theta = self._transform_theta(theta)
        self._mat[rho][theta] += 1

    def get(self, rho, theta):
        rho = self._transform_rho(rho)
        theta = self._transform_theta(theta)
        return self._mat[rho][theta]

    def get_all_above_threshold(self):
        return [(rho, theta) for theta in range(self.theta_max) for rho in range(self.rho_max) if
                self._mat[rho][theta] > self._threshold]

    # Division by 2: Discretisize the "hough matrix".
    # Any two different lines or circles may intersect, but not coincide (they should be at least two pixels apart).
    def _transform_rho(self, rho):
        return int((rho + self.img_diagonal) // 2)

    def _inverse_rho(self, index):
        return (index * 2) - self.img_diagonal

    def _transform_theta(self, theta):
        new_theta = (theta + math.pi) / self.theta_quanta
        return math.floor(new_theta)

    def _inverse_theta(self, index):
        return (index * self.theta_quanta) - math.pi

    def calc_rho(self, x, y, theta):
        return -x * math.cos(theta) + y * math.sin(theta)
