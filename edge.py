import cv2 as cv
import numpy as np
import math

PADDING = 1


def detect_edges(img, threshold_val):
    # According to what we've learned - you better add Gaussian blurring before detecting edges
    img = blur_image(img)

    # Add padding to the image so we can filter with a 3X3 kernel
    img = add_padding_to_image(img)

    # The Sobel edge detector
    img = apply_edge_detection(img)

    # Threshold, default value is 100
    apply_threshold(img, threshold_val)

    # According to what we've learned - you better dilate the image after edge detection
    # and then apply thinning to retrieve the best edges result
    img = apply_dilation(img)
    img = apply_thinning(img)

    # remove the unnecessary padding we added before
    img = remove_padding_from_image(img)

    return img


def blur_image(img, val=5):
    return cv.GaussianBlur(img, (val, val), 0)


def apply_edge_detection(img):
    """
    Apply Sobel filter to find the edges.
    We pre-calculated the convoluted matrices of the filters, so we can apply them directly

    :param img:
    :return:
    """
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
            # Our region of interest is the 8 pixels around our target pixel
            roi = img[y - 1: y + 2, x - 1: x + 2]

            # calculate the gradiant on each axis
            gx = np.dot(roi.flatten(), sobel_x_convoluted.flatten())
            gy = np.dot(roi.flatten(), sobel_y_convoluted.flatten())

            # we found out this is necessary to normalize the gradiant
            abs_grad_x = cv.convertScaleAbs(np.float32([gx]))
            abs_grad_y = cv.convertScaleAbs(np.float32([gy]))

            # calculate the gradiant of the pixel
            g = math.ceil(0.5 * abs_grad_x + 0.5 * abs_grad_y)

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

    # Fill the first and last rows with the same values as the first and last rows of the original image
    for y in range(add_h):
        for x in range(w):
            new_image[y][x + add_w] = img[0][x]
            new_image[-1 - y][x + add_w] = img[-1][x]

    # Fill the first and columns of each row with the same values as the first and last columns of the original image
    for x in range(add_w):
        for y in range(h):
            new_image[y + add_h][x] = img[y][0]
            new_image[y + add_h][-1 - x] = img[y][-1]

    # Fill the corners, get from the left\right pixel
    new_image[0][0] = new_image[0][1]                                   # Top Left
    new_image[new_h - 1][0] = new_image[new_h - 1][1]                   # Bottom Left
    new_image[0][new_w - 1] = new_image[0][new_w - 2]                   # Top Right
    new_image[new_h - 1][new_w - 1] = new_image[new_h - 1][new_w - 2]   # Bottom Right

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


def apply_threshold(img, val=100):
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
    height, width = img.shape
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    thin = create_empty_img(height, width)

    temp_image = img.copy()

    # Loop until erosion leads to an empty set
    while cv.countNonZero(temp_image) != 0:
        # Apply erosion
        erode = cv.erode(temp_image, kernel)
        # Apply opening morphology
        opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        temp_image = erode.copy()

    return thin
