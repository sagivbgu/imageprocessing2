from sys import argv
import cv2 as cv
from edge import *


def apply_threshold(img, val=127):
    height, width = img.shape

    for y in range(height):
        for x in range(width):
            if img[y][x] <= val:
                img[y][x] = 0
            else:
                img[y][x] = 255
    return img


def read_image(file):
    return cv.imread(file, cv.IMREAD_GRAYSCALE)


def write_log_file(output):
    pass


def save_image_and_display(img, file):
    # TODO: display
    cv.imwrite(file, img)


def main():
    image_file = argv[1]
    output_file = argv[2]
    val = int(argv[3])

    original_image = read_image(image_file)
    new_image = apply_edge_detection(original_image)

    apply_threshold(new_image, val)
    save_image_and_display(new_image, "edge_{0}.png".format(val))

    write_log_file(output_file)


if __name__ == "__main__":
    argv.append("tucan.jpg")
    argv.append("output.txt")
    argv.append("200")

    # if len(argv) != 3: # TODO: commented out for debugging
    if len(argv) != 4:
        print("Usage: {0} [image] [output]".format(argv[0]))
        exit()

    main()

