from utils import *
import numpy as np
import cv2 as cv
import math

MIN_THETA = - math.pi
MAX_THETA = math.pi
OFFSET = math.pi

def detect_lines(img):
    height, width = img.shape
    hough_mat = HoughMatrix2D(height, width)

    for y in range(height):
        for x in range(width):
            # consider only white pixels
            if img[y][x] == 0:
                continue

            for theta in range(hough_mat.theta_max):
                rho = hough_mat.calc_rho(x, y, theta)
                hough_mat.increment(rho, theta)

    return hough_mat


def calc_theta(x1, y1, x2, y2):
    pass