import cv2 as cv
import numpy as np

PADDING = 1


def detect_edges(img, threshold_val, blur_val):
    img = blur_image(img, blur_val)
    img = add_padding_to_image(img)
    img = apply_edge_detection(img)
    apply_threshold(img, threshold_val)
    img = apply_dilation(img)
    img = apply_thinning(img)
    img = remove_padding_from_image(img)

    return img


def blur_image(img, val=5):
    return cv.GaussianBlur(img, (val, val), 0)


def apply_edge_detection(img):
    grad_x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    return cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


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


def apply_dilation(img):
    height, width = img.shape
    new_image = create_empty_img(height, width)

    for y in range(PADDING, height - PADDING):
        for x in range(PADDING, width - PADDING):
            # Get all the 9 pixels area around the current pixel
            roi = img[y - 1: y + 2, x - 1: x + 2]

            # if at least one pixel in the region is white, then the
            # center pixel is white
            if 255 in roi:
                new_image[y][x] = 255
            # if all other pixels are black - then the center will be black
            else:
                new_image[y][x] = 0

    return new_image


def apply_thinning(img):
    # Structuring Element
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape, dtype='uint8')

    temp_image = img.copy()

    # Loop until erosion leads to an empty set
    while cv.countNonZero(temp_image) != 0:
        # Erosion
        erode = cv.erode(temp_image, kernel)
        # Opening on eroded image
        opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        temp_image = erode.copy()

    return thin


