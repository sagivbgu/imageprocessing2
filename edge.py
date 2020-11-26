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
    """
    :param img: a padded image
    :return:
    """
    height, width = img.shape
    new_image = create_empty_img(height, width)

    sobel_edge_detection_on_x = np.float32([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]])

    sobel_edge_detection_on_y = np.float32([[1, 2, 1],
                                            [0, 0, 0],
                                            [-1, -2, -1]])

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
    """

    :param img: a padded image after Sober and threshold applied
    :return:
    """
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

def apply_thinning_old(img):
    # Algorithm by
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    #
    # Try each of the 8 masks on each pixel
    # if one of them returns white - set the pixel to white
    # otherwise - set the pixel to black

    height, width = img.shape
    new_image = create_empty_img(height, width)

    mask_1 = {"whites": [(2, 0), (2, 1), (2, 2)],
              "blacks": [(0, 0), (0, 1), (0, 2)]}

    mask_2 = {"whites": [(0, 0), (1, 0), (2, 0)],
              "blacks": [(0, 2), (1, 2), (2, 2)]}

    mask_3 = {"whites": [(0, 0), (0, 1), (0, 2)],
              "blacks": [(2, 0), (2, 1), (2, 2)]}

    mask_4 = {"whites": [(0, 2), (1, 2), (2, 2)],
              "blacks": [(0, 0), (1, 0), (2, 0)]}

    mask_5 = {"whites": [(1, 0), (2, 1)],
              "blacks": [(0, 1), (0, 2), (1, 2)]}

    mask_6 = {"whites": [(0, 1), (1, 0)],
              "blacks": [(1, 2), (2, 2), (2, 1)]}

    mask_7 = {"whites": [(0, 1), (1, 2)],
              "blacks": [(1, 0), (2, 0), (2, 1)]}

    mask_8 = {"whites": [(1, 2), (2, 1)],
              "blacks": [(0, 0), (0, 1), (1, 0)]}

    masks = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8]

    for y in range(PADDING, height - PADDING):
        for x in range(PADDING, width - PADDING):
            # consider only white pixels:
            if img[y][x] == 0:
                continue

            # Get all the 9 pixels area around the current pixel
            roi = img[y - 1: y + 2, x - 1: x + 2]
            val = 0

            for mask in masks:
                val = try_mask(roi, mask["whites"], mask["blacks"])
                # if one mask returned white, then the pixel should be white
                # and there is no point to check the rest
                if val == 255:
                    break

            new_image[y][x] = val

    return new_image


def try_mask(roi, whites, blacks):
    # whites is an array of tuples (y,x) of the locations of the white pixels to check
    for (y, x) in whites:
        # if one white does not match, set the center pixel to black
        if roi[y][x] != 255:
            return 0

    for (y, x) in blacks:
        if roi[y][x] != 0:
            return 0

    return 255




