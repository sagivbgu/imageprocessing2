from sys import argv
import cv2 as cv

from circles import detect_circles, draw_circles_on_img
from lines import detect_lines, draw_lines_on_img
from edge import detect_edges


def read_image(file):
    return cv.imread(file, cv.IMREAD_GRAYSCALE)


def get_loaded_image_name():
    return argv[1].split(".")[0]


def save_image(img):
    name = get_loaded_image_name()
    file_name = "{0}_after_detection.png".format(name)
    cv.imwrite(file_name, img)


def display_image(img):
    name = get_loaded_image_name()
    cv.imshow(name, img)
    cv.waitKey(0)


def write_log_file(lines, circles, output_file_path):
    with open(output_file_path, "w") as f:
        f.write("{lines} {circles}\n".format(lines=len(lines), circles=len(circles)))
        for (p1, p2) in lines:
            f.write("{x1} {y1} {x2} {y2}\n".format(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1]))
        for c in circles:
            f.write("{a} {b} {r}\n".format(a=c[0], b=c[1], r=c[2]))


# TODO: Remove if not used
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

    image = read_image(image_file)

    print("Detecting edges", end="")
    # According to clarifications - threshold for edge detection is 100
    edges_image = detect_edges(image, threshold_val=100)
    print(" - Done!")

    print("Detecting lines", end="")
    lines = detect_lines(edges_image)
    print(" - Done!")

    print("Detecting circles", end="")
    #circles = detect_circles(edges_image)
    circles = []
    print(" - Done!")

    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    draw_lines_on_img(lines, image)
    draw_circles_on_img(circles, image)

    write_log_file(lines, circles, output_file)

    save_image(image)
    display_image(image)


if __name__ == "__main__":
    argv.append("one_diagonal_line.png")
    argv.append("output.txt")
    if len(argv) != 3:
        print("Usage: {0} [image] [output]".format(argv[0]))
        exit()

    main()
