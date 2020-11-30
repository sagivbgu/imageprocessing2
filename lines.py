import numpy as np
import cv2 as cv
import math
from numpy.linalg import norm
"""
The threshold for lines in the Hough matrix is calculated as:
all the lines that got at least half of the maximum number of votes

"""
np.seterr('raise')


class Line:
    def __init__(self, rho, theta, points):
        self.rho = rho
        self.theta = theta
        self.m = None  # slope
        self.c = None  # intercept with y axis (left side of image)
        self.tag = "line"  # can be "line", "segment", "gap"
        self.points = points  # will hold the array of points

        self.points.sort()
        self.start = self.points[0] if len(points) > 0 else None # always with the smaller x
        self.end = self.points[-1] if len(points) > 0 else None  # always with the larger x
        self.calc_slope_and_intercept()

        self.edge_point_1 = None
        self.edge_point_2 = None

    def width(self):
        return abs(self.start[0] - self.end[0])

    def height(self):
        return abs(self.start[1] - self.end[1])

    def length(self):
        if self.tag == "line" or self.tag == "segment":
            return len(self.points)
        if self.tag == "gap":
            return norm(np.asarray(self.start) - np.asarray(self.end)) + 1

    def calc_start_and_end_points(self):
        if self.start is None:
            self.start = self.points[0]
        if self.end is None:
            self.end = self.points[-1]

    def calc_slope_and_intercept(self):
        if self.start is None or self.end is None:
            return
        if self.m is None or self.c is None:
            x1, y1 = self.start
            x2, y2 = self.end

            # slope is infinity
            if x2 == x1:
                return None, None

            m = (y2 - y1) / (x2 - x1)
            c = y2 - m * x2

            self.m = m
            self.c = c

    def set_segment(self):
        self.tag = "segment"

    def set_gap(self):
        self.tag = "gap"

    def is_segment(self):
        return self.tag == "segment"


class HoughMatrix2D:
    def __init__(self, img, rho_quanta=3, theta_quanta=(math.pi / 90)):
        self._img = img
        self._threshold = 0

        self._img_height = img.shape[0]
        self._img_width = img.shape[1]
        self.img_diagonal = math.ceil(math.sqrt(self._img_height ** 2 + self._img_width ** 2))

        self.rho_quanta = rho_quanta
        self.rho_min = self._transform_rho(- self.img_diagonal)
        self.rho_max = self._transform_rho(self.img_diagonal)

        self.theta_quanta = theta_quanta
        self.theta_min = self._transform_theta(- math.pi)
        self.theta_max = self._transform_theta(math.pi)

        self._mat_of_votes = np.zeros((self.rho_max, self.theta_max))
        self._mat_of_points = [[[] for _ in range(self.theta_max)] for _ in range(self.rho_max)]

    def increment_by_x_y(self, x, y):
        for theta in range(self.theta_max):
            inv_theta = self._inverse_theta(theta)
            rho = self._calc_rho(x, y, inv_theta)
            rho = self._transform_rho(rho)
            self._mat_of_votes[rho][theta] += 1
            self._mat_of_points[rho][theta].append((x, y))

    def get_all_above_threshold(self):
        self._calc_threshold()
        return [(self._inverse_rho(rho), self._inverse_theta(theta), self._mat_of_points[rho][theta])
                for theta in range(self.theta_max) for rho in range(self.rho_max)
                if self._mat_of_votes[rho][theta] >= self._threshold]

    def _transform_rho(self, rho):
        return int((rho + self.img_diagonal) // self.rho_quanta)

    def _inverse_rho(self, index):
        return (index * self.rho_quanta) - self.img_diagonal

    def _transform_theta(self, theta):
        new_theta = (theta + math.pi) / self.theta_quanta
        return math.floor(new_theta)

    def _inverse_theta(self, index):
        return (index * self.theta_quanta) - math.pi

    def _calc_rho(self, x, y, theta):
        return x * math.cos(theta) + y * math.sin(theta)

    def _calc_threshold(self):
        #num_of_pixels = np.count_nonzero(self._img)
        #self._threshold = num_of_pixels * 0.2
        self._threshold = np.amax(self._mat_of_votes) / 2


def detect_lines(img):
    hough_mat = build_hough_mat(img)

    lines = get_lines_from_hough_matrix(hough_mat)

    if len(lines) == 0:
        print(" - No lines detected", end="")
        return []

    lines = calc_edges_points_of_lines_and_eliminate_outofimage_lines(lines, img)

    lines_divided_into_segments = calc_lines_segments(lines)

    lines_divided_into_segments_and_gaps = calc_lines_gaps(lines_divided_into_segments)

    lines_united_segments = unite_lines_segments(lines_divided_into_segments_and_gaps)

    lines_without_close_segments = eliminate_too_close_segments(lines_divided_into_segments_and_gaps)

    # Flatten all segments
    segments = [segment for line in lines_without_close_segments for segment in line]

    segments_start_and_end_points = extract_start_and_end_points(segments)

    # remove_dups
    segments_start_and_end_points = list(set(segments_start_and_end_points))

    return segments_start_and_end_points


def draw_lines_on_img(coordinates, img):
    if len(coordinates) == 0:
        return
    for (p1, p2) in coordinates:
        cv.line(img, p1, p2, color=(255, 255, 0))


def get_lines_from_hough_matrix(hough_mat):
    rhos_thetas_points = hough_mat.get_all_above_threshold()
    lines = []

    for (rho, theta, points) in rhos_thetas_points:
        line = Line(rho, theta, points)
        lines.append(line)

    return lines


def build_hough_mat(img):
    height, width = img.shape
    hough_mat = HoughMatrix2D(img)

    for y in range(height):
        for x in range(width):
            # consider only white pixels
            if img[y][x] == 0:
                continue

            hough_mat.increment_by_x_y(x, y)

    return hough_mat


def calc_line_coordinates(rho, theta, width, height):
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = rho * a
    y0 = rho * b

    x1 = int(x0 + (2 * width * -b))
    y1 = int(y0 + 2 * height * a)

    x2 = int(x0 - (2 * width * -b))
    y2 = int(y0 - 2 * height * a)

    return (x1, y1), (x2, y2)


def calc_edges_points_of_lines_and_eliminate_outofimage_lines(lines, img):
    height, width = img.shape

    new_lines = []

    for line in lines:
        x1, y1 = line.start

        # if vertical
        if line.m is None:
            if is_intercept_valid(x1, width):
                a = (x1, 0)
                b = (x1, height - 1)
            else:
                continue

        # if horizontal
        elif line.m == 0:
            if is_intercept_valid(line.c, height):
                a = (0, line.c)
                b = (width - 1, line.c)
            else:
                continue

        else:
            intercept_with_x = -line.c / line.m  # y = 0
            intercept_with_bottom_side = ((height - 1) - line.c) / line.m  # y = (height - 1)
            intercept_with_right_side = line.m * (width - 1) + line.c  # x = (width - 1)

            if is_intercept_valid(intercept_with_x, width):
                # TOP and LEFT
                if is_intercept_valid(line.c, height):
                    a = (intercept_with_x, 0)
                    b = (0, line.c)
                # TOP and RIGHT
                elif is_intercept_valid(intercept_with_right_side, height):
                    a = (intercept_with_x, 0)
                    b = (width - 1, intercept_with_right_side)
                # TOP and BOTTOM
                elif is_intercept_valid(intercept_with_bottom_side, width):
                    a = (intercept_with_x, 0)
                    b = (intercept_with_bottom_side, 0)
                else:
                    continue
            elif is_intercept_valid(intercept_with_bottom_side, width):
                # BOTTOM and LEFT
                if is_intercept_valid(line.c, height):
                    a = (intercept_with_bottom_side, height - 1)
                    b = (0, line.c)
                # BOTTOM and RIGHT
                if is_intercept_valid(intercept_with_right_side, height):
                    a = (intercept_with_bottom_side, height - 1)
                    b = (width - 1, intercept_with_right_side)
                else:
                    continue
            elif is_intercept_valid(line.c, height):
                # LEFT and RIGHT
                if is_intercept_valid(intercept_with_right_side, height):
                    a = (0, line.c)
                    b = (width - 1, intercept_with_right_side)
                else:
                    continue
            else:
                continue

        x1, y1 = a
        x2, y2 = b

        line.edge_point_1 = (int(x1), int(y1))
        line.edge_point_2 = (int(x2), int(y2))

        new_lines.append(line)

    return new_lines


def is_intercept_valid(x, max_val):
    return (x >= 0) and (x < max_val)


def distance_between_point_and_line(point, line):
    x, y = point
    line.calc_slope_and_intercept()
    # if the line is vertical
    if line.m is None:
        x_line = line.start[0]
        return abs(x - x_line)

    # if the line is horizontal
    if line.m == 0:
        y_line = line.start[1]
        return abs(y - y_line)

    orth_line_m = -1 / line.m
    orth_line_c = y - (orth_line_m * x)

    x_intersect = (orth_line_c - line.c) / (line.m - orth_line_m)
    y_intersect = orth_line_m * x_intersect + orth_line_c

    #return norm(np.asarray(point) - np.asarray((x_intersect, y_intersect)))
    return math.sqrt((x - x_intersect) ** 2 + (y - y_intersect) ** 2)


def distance_between_point_and_line_old(point, line):
    p1 = np.asarray(point)
    p2 = line.start
    p3 = line.end
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    # and we don't want to continue to division by zero
    if (norm(p2-p1) == 0) or (np.cross(p2-p1, p1-p3) == 0):
        x1, y1 = p1
        x2, y2 = p2

        # if they are on the same X axis
        if x1 == x2:
            return abs(y1 - y2)

        # if they are on the same Y axis
        if y1 == y2:
            return abs(x1 - x2)

        # Just in case
        return 0

    return norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)


def are_lines_the_same(line1, line2):

    def are_segments_the_same(seg1, seg2):
        return (seg1.start == seg2.start) and (seg1.end == seg2.end)

    if isinstance(line1, Line) and isinstance(line2, Line):
        return are_segments_the_same(line1, line2)

    if len(line1) != len(line2):
        return False

    for item1 in line1:
        for item2 in line2:
            if item1.is_segment() and item2.is_segment():
                if not are_segments_the_same(item1, item2):
                    return False

    return True


def are_lines_at_least_two_pixels_apart(line1, line2):
    line1_far = False
    line2_far = False

    # For each point in line1
    for point in line1.points:
        # if you find a point far enough, the line is far
        if distance_between_point_and_line(point, line2) > 2:
            line1_far = True
            break

    for point in line2.points:
        if distance_between_point_and_line(point, line1) > 2:
            line2_far = True
            break

    # only if both lines are far enough, they are far
    return line1_far and line2_far


def do_points_touch(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and abs(y1 - y2) == 1:
        return True

    if y1 == y2 and abs(x1 - x2) == 1:
        return True

    if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
        return True

    return False


def find_line_segments(line):
    new_line = []

    # Set the first segment of the line, has the same rho and theta
    segment = Line(line.rho, line.theta, [])
    segment.set_segment()
    new_line.append(segment)

    # add the first point of the line
    cur_point = line.start
    segment.points.append(cur_point)

    # Go for each point in the line
    for next_point in line.points[1:]:
        # if the next touches the current one, add it to the segment
        if do_points_touch(cur_point, next_point):
            cur_point = next_point
            segment.points.append(cur_point)
        # else, there is a gap between them, and for now:
        # 1) wrap up the current segment
        # 2) open a new segment
        else:
            # wrap up the current segment
            segment.calc_start_and_end_points()

            # open a new segment
            segment = Line(line.rho, line.theta, [])
            segment.set_segment()
            new_line.append(segment)

            cur_point = next_point
            segment.points.append(cur_point)

    # wrap up the last segment
    segment.calc_start_and_end_points()

    return new_line


def calc_lines_segments(lines):
    new_lines = []
    for line in lines:
        line_divided_into_segments = find_line_segments(line)
        new_lines.append(line_divided_into_segments)

    return new_lines


def calc_gap_between_segments(seg1, seg2):
    gap = Line(0, 0, [])
    gap.set_gap()

    start = [seg1.end[0], seg1.end[1]]
    end = [seg2.start[0], seg2.start[1]]

    # try to improve start and end points of the gap
    # the minimal case here - there is a gap of ONE pixel on some axis
    # in that case, both start and end points of the gap will have the same coordinate in that axis
    if start[0] < end[0]:
        start[0] += 1
        end[0] -= 1

    if start[1] < end[1]:
        start[0] += 1
        end[0] -= 1

    elif start[1] > end[1]:
        start[0] -= 1
        end[0] += 1

    gap.start = start
    gap.end = end

    return gap


def find_line_gaps(line):
    # line is now an array of segments

    # if there is a single segment, that's the line - there are no gaps
    if len(line) == 1:
        return line

    # init a new line
    new_line = []
    cur_segment = line[0]
    new_line.append(cur_segment)

    for next_segment in line[1:]:
        gap = calc_gap_between_segments(cur_segment, next_segment)
        new_line.append(gap)

        cur_segment = next_segment
        new_line.append(cur_segment)

    return new_line


def calc_lines_gaps(lines):
    new_lines = []
    for line in lines:
        line_divided_into_segments_and_gaps = find_line_gaps(line)
        new_lines.append(line_divided_into_segments_and_gaps)

    return new_lines


def extract_start_and_end_points(lines):
    segments_start_and_end_points = []

    for line in lines:
        if line.is_segment():
            segments_start_and_end_points.append((line.start, line.end))
        elif line.tag == "gap":
            continue
        else:
            raise Exception("Should be segment")

    return segments_start_and_end_points


def eliminate_too_close_segments_between_two_lines(line1, line2):
    if are_lines_the_same(line1, line2):
        return line1, None

    # a very simple method for elimination
    eliminated = []
    new_segments = []

    def add_segment(seg):
        if (seg not in new_segments) and (seg not in eliminated):
            new_segments.append(seg)

    def remove_segment(seg):
        if seg in new_segments:
            new_segments.remove(seg)
        if seg not in eliminated:
            eliminated.append(seg)

    for seg1 in line1:
        # consider only segments and not gaps
        if not seg1.is_segment():
            continue
        for seg2 in line2:
            # consider only segments and not gaps
            if not seg2.is_segment():
                continue
            # if the segments are distant - leave them
            if are_lines_at_least_two_pixels_apart(seg1, seg2):
                continue
            # the segments are too close
            else:
                if seg1.length() >= seg2.length():
                    if seg2 in line2:
                        line2.remove(seg2)
                else:
                    if seg1 in line1:
                        line1.remove(seg1)

    return line1, line2


def eliminate_too_close_segments(lines):
    new_lines = []
    for line1 in lines:
        for line2 in lines[1:]:
            line1, line2 = eliminate_too_close_segments_between_two_lines(line1, line2)
            new_lines.append(line1)
            if line2 is not None:
                new_lines.append(line2)

    return new_lines


def unite_line_segments(line):
    return line


def unite_lines_segments(lines):
    new_lines = []
    for line in lines:
        new_line = unite_line_segments(line)
        new_lines.append(new_line)
