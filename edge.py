import cv2 as cv
import numpy as np

PADDING = 1


def detect_edges(img, threshold_val):
    img = add_padding_to_image(img)
    img = apply_edge_detection(img)
    apply_threshold(img, threshold_val)
    img = remove_padding_from_image(img)

    return img


def apply_edge_detection(img):
    sobel_x_convoluted = np.float32([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]])

    sobel_y_convoluted = np.float32([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])

    height, width = img.shape
    new_image = create_empty_img(height, width)

    for y in range(PADDING, height - PADDING):
        for x in range(PADDING, width - PADDING):
            roi = img[y - 1: y + 2, x - 1: x + 2]
            gx = np.dot(roi.flatten(), sobel_x_convoluted.flatten())
            gy = np.dot(roi.flatten(), sobel_y_convoluted.flatten())
            g = (gx ** 2 + gy ** 2) ** 0.5

            new_image[y][x] = g

    return new_image


def add_padding_to_image(img, add_h=PADDING, add_w=PADDING):
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
    new_h = h - add_h * 2
    new_w = w - add_w * 2

    new_image = create_empty_img(new_h, new_w)
    for y in range(add_h, h - add_h):
        for x in range(add_w, w - add_w):
            new_x = x - add_w
            new_y = y - add_h
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


def apply_threshold(img, val=127):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            if img[y][x] <= val:
                img[y][x] = 0
            else:
                img[y][x] = 255
    return img
