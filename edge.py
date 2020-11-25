import cv2 as cv
import numpy as np

PADDING = 1

sobel_edge_detection_on_x = np.float32([ [-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]])

sobel_edge_detection_on_y = np.float32([ [1, 2, 1],
                                         [0, 0, 0],
                                         [-1, -2, -1]])


def apply_edge_detection(img):
    """
    :param img: a padded image
    :return:
    """
    height, width = img.shape
    new_image = create_empty_img(height, width)

    for y in range(PADDING, height - PADDING):
        for x in range(PADDING, width - PADDING):
            # Get all the 9 pixels area around the current pixel
            roi = img[y - 1: y + 2, x - 1: x + 2]
            gx = apply_filter(sobel_edge_detection_on_x, roi)
            gy = apply_filter(sobel_edge_detection_on_y, roi)
            g = ((gx ** 2) + (gy ** 2)) ** 0.5
            new_image[y][x] = g
    return new_image


def apply_filter(filter, roi):
    """
    :param filter:
    :param roi:
    :return:
    """
    return np.dot(np.float32(filter).flatten(), np.float32(roi).flatten())


def add_padding_to_image(img, add_h=2, add_w=2):
    """
    Pad an image with a given padding sizes
    :param img: The image to pad
    :param add_h: The number of pixels to add as a padding to each side (height)
    :param add_w: The number of pixels to add as a padding to each side (width)
    :return: The padded image
    """
    h, w = img.shape
    new_h = h + add_h * 2
    new_w = w + add_w * 2

    new_image = create_empty_img(new_h, new_w)
    for y in range(h):
        for x in range(w):
            new_x = x + add_w
            new_y = y + add_h
            new_image[new_y][new_x] = img[y][x]

    return new_image


def remove_padding_from_image(img, add_h=PADDING, add_w=PADDING):
    """
    Remove a padding from an image with the given padding sizes
    :param img: The image to remove pad from
    :param add_h: The number of pixels to add as a padding to each side (height)
    :param add_w: The number of pixels to add as a padding to each side (width)
    :return: The original image
    """
    h, w = img.shape
    new_h = h + add_h * 2
    new_w = w + add_w * 2

    new_image = create_empty_img(new_h, new_w)
    for y in range(h):
        for x in range(w):
            new_x = x + add_w
            new_y = y + add_h
            new_image[new_y][new_x] = img[y][x]

    return new_image


def create_empty_img(h, w, color=0):
    """
    Create a matrix representing an empty image
    :param h: The desired height
    :param w: The desired width
    :param color: The color to fill the matrix with
    """
    return color + np.zeros(shape=[h, w], dtype=np.uint8)

