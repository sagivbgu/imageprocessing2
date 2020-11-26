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
