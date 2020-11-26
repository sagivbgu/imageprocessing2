from sys import argv
import cv2 as cv
from edge import *


def read_image(file):
    return cv.imread(file, cv.IMREAD_GRAYSCALE)


def write_log_file(output):
    pass


def save_image_and_display(img):
    # TODO: display
    name = argv[1].split(".")[0]
    file_name = "{0}_edge_thresh{1}_blur{2}.png".format(name, argv[3], argv[4])
    cv.imwrite(file_name, img)


def main():
    image_file = argv[1]
    output_file = argv[2]
    threshold_val = int(argv[3])
    blur_val = int(argv[4])

    original_image = read_image(image_file)

    new_image = detect_edges(original_image, threshold_val, blur_val)

    save_image_and_display(new_image)

    write_log_file(output_file)


if __name__ == "__main__":
    argv.append("potatos.jpg")
    argv.append("output.txt")
    argv.append("100") # threshold
    argv.append("5") # blurring

    # if len(argv) != 3: # TODO: commented out for debugging
    if len(argv) != 5:
        print("Usage: {0} [image] [output]".format(argv[0]))
        exit()

    main()

