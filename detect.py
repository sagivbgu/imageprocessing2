from sys import argv
import cv2 as cv

from circles import detect_circles
from edge import *

DETECTED_CIRCLE_COLOR = (150, 0, 0)


def read_image(file):
    return cv.imread(file, cv.IMREAD_GRAYSCALE)


def get_loaded_image_name():
    return argv[1].split(".")[0]


def save_image(img):
    name = get_loaded_image_name()
    file_name = "{0}_edge_thresh{1}_blur{2}.png".format(name, argv[3], argv[4])
    cv.imwrite(file_name, img)


def display_image(img):
    name = get_loaded_image_name()
    cv.imshow(name, img)
    cv.waitKey(0)


def write_log_file(lines, circles, output_file_path):
    with open(output_file_path, "w") as f:
        # TODO: Print lines
        for c in circles:
            f.write("({a},{b},{r})".format(a=c[0], b=c[1], r=c[2]))


def draw_on_top(base_image, image_to_draw, ignore=0):
    h, w = base_image.shape
    hn, wn = image_to_draw.shape

    if hn != h or wn != w:
        raise ValueError("Can't draw image on top of an image with different dimensions")

    for y in range(h):
        for x in range(w):
            if image_to_draw[y][x] != ignore:
                base_image[y][x] = image_to_draw[y][x]


def main():
    image_file = argv[1]
    output_file = argv[2]
    threshold_val = int(argv[3])
    blur_val = int(argv[4])

    image = read_image(image_file)

    edges_image = detect_edges(image, threshold_val, blur_val)
    circles = detect_circles(edges_image)

    # TODO: Remove code comments
    # empty_image = create_empty_img(image.shape[0], image.shape[1])

    for (a, b, r) in circles:
        # print("({},{},{})".format(a, b, r))
        # cv.circle(empty_image, (a, b), r, DETECTED_CIRCLE_COLOR, 1)
        cv.circle(image, (a, b), r, DETECTED_CIRCLE_COLOR, 1)

    # display_image(empty_image)

    # draw_on_top(image, edges_image)

    display_image(image)
    save_image(image)

    lines = []  # TODO
    write_log_file(lines, circles, output_file)


if __name__ == "__main__":
    argv.append("potatos.jpg")
    argv.append("output.txt")
    argv.append("100")  # threshold
    argv.append("5")  # blurring

    # if len(argv) != 3: # TODO: commented out for debugging
    if len(argv) != 5:
        print("Usage: {0} [image] [output]".format(argv[0]))
        exit()

    main()
