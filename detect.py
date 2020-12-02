from sys import argv
import cv2 as cv

from circles import detect_circles, draw_circles_on_img
from lines import detect_lines, draw_lines_on_img
from edge import detect_edges

"""
Notes:

In the end of the process
- 2 images are displayed:
1) The image after applying edge detection (the binary image B)
2) The original image with the detected lines (light blue) and circles (yellow)

- In addition, 2 files are saved:
1) The output file                               (           {argv[2]}           )
2) The image with the detected lines and circles ( {argv[1]}_after_detection.png )
"""


def read_image(file):
    return cv.imread(file, cv.IMREAD_GRAYSCALE)


def get_loaded_image_name():
    return argv[1].split(".")[0]


def save_image(img):
    name = get_loaded_image_name()
    file_name = "{0}_after_detection.png".format(name)
    cv.imwrite(file_name, img)


def display_images(img, edges_image):
    name = get_loaded_image_name()
    cv.imshow("Lines and circles: {}".format(name), img)
    cv.imshow("After edge detection: {}".format(name), edges_image)
    cv.waitKey(0)


def write_log_file(lines, circles, output_file_path):
    with open(output_file_path, "w") as f:
        f.write("{lines} {circles}\n".format(lines=len(lines), circles=len(circles)))
        for (p1, p2) in lines:
            f.write("{x1} {y1} {x2} {y2}\n".format(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1]))
        for c in circles:
            f.write("{a} {b} {r}\n".format(a=c[0], b=c[1], r=c[2]))


def main():
    image_file = argv[1]
    output_file = argv[2]

    image = read_image(image_file)

    # According to clarifications - threshold for edge detection is 100
    edges_image = detect_edges(image, threshold_val=100)

    lines = detect_lines(edges_image)

    circles = detect_circles(edges_image)

    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    draw_lines_on_img(lines, image)
    draw_circles_on_img(circles, image)

    write_log_file(lines, circles, output_file)
    save_image(image)
    display_images(image, edges_image)


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: {0} [image] [output]".format(argv[0]))
        exit()

    main()
