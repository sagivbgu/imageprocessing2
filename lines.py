import numpy as np
import cv2 as cv
import math
from numpy.linalg import norm
from sys import maxsize

"""
The threshold for lines in the Hough matrix is calculated as:
all the lines that got at least 25% of the maximum number of votes
By trial and error - this got the best results for several diverse images

Explanation about the code:
1) while building the Hough matrix, we save for each line (represented by rho, theta)
    all the points that voted for this line.
    
2) After that we filter all the significant lines out of the matrix (using the threshold explained above)

3) For each line we got from the Hough matrix, we create a Line object (with the tag 'line')
    which stores all the data important for further calculations and logic (rho, theta and so on)

4) Then we filter out all the duplicated lines (due to symmetry of thetas and rhos)

5) We "follow" each line and dividing it into "segments" (new Line objects with the tag 'segment')
    segment is a sequence of connected edge points, each line may contain several segments
    and now "line" is an array of "segments" objects

6) For each line, we follow the segments and calculating the "gaps" (the sequence of blank points
    between each segment). we create for each gap a Line object (using tag 'gap')
    
    now a line is an array of segments and gaps of the form [seg1, gap1, seg2, gap2,...,gap(n-1), segn]

7) Now, we go over each line and deciding whether two segments should be united into one large segment.
    This happens when the gap between two segments is relatively short, or when the segments
    and the gap in between are short altogether (this usually happens in diagonal lines, where 
    the length of segments and gaps tend to be really small ~2-3 pixels each)

8) We now remove all the segments that are too close to each other (less than 2 pixels apart)

9) We filter out all the short segments (with less than 1% of edge points)

10) Returning the start and end points of all the segments
    
"""


class Line:
    """
    The object representing:
        - 'line' - as received from the Hough matrix
        - 'segments' - a sequence of touching edge points
        - 'gap' - a sequence of blank points between two segments
    """
    def __init__(self, rho, theta, points):
        self.rho = rho
        self.theta = theta
        self.m = None  # slope
        self.c = None  # intercept with y axis (left side of image)
        self.tag = "line"  # can be "line", "segment", "gap"
        self.points = points  # will hold the array of points

        self.points.sort()  # First by X and then by Y

        # The point on the Top Left (smallest X and smallest Y)
        self.start = self.points[0] if len(points) > 0 else None

        # The point on the Bottom Right (largest X and largest Y)
        # Because the line can be going diagonally from bottom left to top right
        # we can only be sure that start.x <= end.x
        self.end = self.points[-1] if len(points) > 0 else None

        self.calc_slope_and_intercept()

    def length(self):
        if self.tag == "line" or self.tag == "segment":
            return len(self.points)
        # because 'gap' will not hold points in its array, we calc the length as follows
        if self.tag == "gap":
            return norm(np.asarray(self.start) - np.asarray(self.end)) + 1

    def reset_start_and_end_points(self):
        self.points.sort()
        self.start = self.points[0]
        self.end = self.points[-1]

    def set_start_and_end_points(self):
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
                self.m = None
                self.c = None
                return

            # slope is finite number
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

    def is_gap(self):
        return self.tag == "gap"


class HoughMatrix2D:
    """
    Class representing the Hough matrix.
        - rho_quanta and theta_quanta are the parameters of the resolution of the matrix
          by default, we want the resolution to be a single pixel for rho, and a one
          degree angle for each cell in the matrix

        - the range of rho is [-d , d], when d = length of the diagonal
        - the range of theta is [-pi , pi]
        - the (0,0) point is on the Top Left
    """
    def __init__(self, img, rho_quanta=1, theta_quanta=(math.pi / 180)):
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

        # As explained above, we have two matrices - one for votes, and one that hold the voters
        self._mat_of_votes = np.zeros((self.rho_max, self.theta_max))
        self._mat_of_points = [[[] for _ in range(self.theta_max)] for _ in range(self.rho_max)]

    def increment_by_x_y(self, x, y):
        # Iterate over each possible theta, and calculate the matching rho
        # then increment the number of votes, and add the point to the array of voters
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
        # the range of rho is [-d, d]
        # we want to map [-d , d] -> [0, 2d] divided into '2d / rho_quanta' sections
        return int((rho + self.img_diagonal) // self.rho_quanta)

    def _inverse_rho(self, index):
        return (index * self.rho_quanta) - self.img_diagonal

    def _transform_theta(self, theta):
        # the range of theta is [-pi, pi]
        # we want to map [-pi , pi] -> [0, 2pi] divided into '2ip / theta_quanta' sections
        new_theta = (theta + math.pi) / self.theta_quanta
        return math.floor(new_theta)

    def _inverse_theta(self, index):
        return (index * self.theta_quanta) - math.pi

    def _calc_rho(self, x, y, theta):
        return x * math.cos(theta) + y * math.sin(theta)

    def _calc_threshold(self):
        # As explained above - 25% of the maximum number of votes
        self._threshold = np.amax(self._mat_of_votes) / 4


def detect_lines(img):
    """
    The main function.
    The steps explained above
    """
    hough_mat = build_hough_mat(img)

    # lines here is an array of Line(using the tag 'line')
    # all the lines here passed the threshold
    lines = get_lines_from_hough_matrix(hough_mat)

    if len(lines) == 0:
        return []

    lines = remove_duplicate_lines(lines)

    lines_segments = find_lines_segments(lines)

    lines_segments_and_gaps = find_lines_gaps(lines_segments)

    lines_united_segments = unite_lines_segments(lines_segments_and_gaps)

    lines_without_close_segments = eliminate_too_close_segments(lines_united_segments)

    # Flatten all segments
    segments = [segment for line in lines_without_close_segments for segment in line]

    # according to clarifications - the length of a line must be at least 1%
    # of all edges points
    segments = remove_too_short_segments(segments, np.count_nonzero(img) / 100)

    segments_start_and_end_points = extract_start_and_end_points(segments)

    # sort for the output file
    segments_start_and_end_points.sort()

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


def distance_between_two_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distance_between_point_and_line(point, line):
    # We want to calculate the distance between a point and a line
    # as the length of the perpendicular line connecting the point and the given line
    x, y = point
    line.calc_slope_and_intercept()

    # if the line is vertical
    if line.m is None:
        x_line = line.start[0]
        x_intersect = x_line
        y_intersect = y

    # if the line is horizontal
    elif line.m == 0:
        y_line = line.start[1]
        x_intersect = x
        y_intersect = y_line

    # the line is ordinary
    else:
        perp_line_m = -1 / line.m
        perp_line_c = y - (perp_line_m * x)

        x_intersect = (perp_line_c - line.c) / (line.m - perp_line_m)
        y_intersect = perp_line_m * x_intersect + perp_line_c

    # first check how far is the intersection from the actual line start and end points:
    if (distance_between_two_points((x_intersect, y_intersect), line.start) <= 2) or \
            (distance_between_two_points((x_intersect, y_intersect), line.end) <= 2):
        # if it is less than 2 pixels - return 0, so it will be close
        return 0

    # if it's far from the start and end points, check if the intersection is on the line
    #
    # if the intersection is NOT ON the line - meaning the perpendicular line doesn't intersect
    # the actual given line (only its continuation) - then the distance doesn't matter to us
    # and we can say that the lines are far enough from each other
    if not (line.start[0] <= x_intersect <= line.end[0] and
            (line.start[1] <= y_intersect <= line.end[1] or
                line.start[1] >= y_intersect >= line.end[1])):
        return maxsize

    # the intersection is ON the line, so return the length of the perpendicular line
    return distance_between_two_points((x, y), (x_intersect, y_intersect))


def are_lines_the_same(line1, line2):
    return (line1.start == line2.start) and (line1.end == line2.end)


def are_segments_at_least_two_pixels_apart(line1, line2):
    # First, let's check if they are parallel and deal with this situation:
    # if two lines are parallel, we want to eliminate only if (1) they "overlap" -
    # meaning, they share a common range of Xs or Ys, like that:
    #
    #   line1=          ******************
    #   line2=                  ******************
    #
    # and if (2) the distance between them is 2 pixels or less
    #
    if line1.theta == line2.theta:
        # compare start and end points
        # if one of the edge points is too close - they are close
        if (distance_between_point_and_line(line1.start, line2) <= 2) or \
            (distance_between_point_and_line(line1.end, line2) <= 2) or \
            (distance_between_point_and_line(line2.start, line1) <= 2) or \
                (distance_between_point_and_line(line2.end, line1) <= 2):
            return False
        else:
            return True

    # Now, if they are not parallel, we want to make sure that even if they intersect
    # there is a point (on one of the lines) that is at least 2 pixels far from the other line

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
    # Used when determining if two points belong to the same segment
    # touching means they either next to each other on the X axis, or
    # the Y axis, or they share a corner
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
    line_segments = []

    # Set the first segment of the line, is has the same rho and theta, and no points at the beginning
    segment = Line(line.rho, line.theta, [])
    segment.set_segment()
    line_segments.append(segment)

    # Add the first point of the line
    cur_point = line.start
    segment.points.append(cur_point)

    # Go for each point in the line
    for next_point in line.points[1:]:
        # if the next touches the current one, add it to the segment
        if do_points_touch(cur_point, next_point):
            cur_point = next_point
            segment.points.append(cur_point)
        # else, there is a gap between them:
        # 1) wrap up the current segment
        # 2) open a new segment
        else:
            # wrap up the current segment
            segment.set_start_and_end_points()

            # open a new segment
            segment = Line(line.rho, line.theta, [])
            segment.set_segment()
            line_segments.append(segment)

            cur_point = next_point
            segment.points.append(cur_point)

    # wrap up the last segment
    segment.set_start_and_end_points()

    return line_segments


def find_lines_segments(lines):
    return [find_line_segments(line) for line in lines]


def find_gap_between_segments(seg1, seg2):
    gap = Line(0, 0, [])
    gap.set_gap()

    # At first, set the start of the gap as the end point of segment 1
    # and the end point of the gap as the start point of segment 2
    # We use a list and not tuple because it's mutable
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
    # this method transforms it to an array of [seg1, gap1, seg2, gap2,...,gap(n-1), segn]

    # if there is a single segment, that's the line - there are no gaps
    if len(line) == 1:
        return line

    # init a new line
    segments_and_gaps_of_line = []
    cur_segment = line[0]
    segments_and_gaps_of_line.append(cur_segment)

    for next_segment in line[1:]:
        gap = find_gap_between_segments(cur_segment, next_segment)
        segments_and_gaps_of_line.append(gap)

        cur_segment = next_segment
        segments_and_gaps_of_line.append(cur_segment)

    return segments_and_gaps_of_line


def find_lines_gaps(lines):
    return [find_line_gaps(line) for line in lines]


def extract_start_and_end_points(segments):
    segments_start_and_end_points = {(seg.start, seg.end) for seg in segments}
    return list(segments_start_and_end_points)


def eliminate_too_close_segments_between_two_lines(line1, line2):
    for seg1 in line1:
        for seg2 in line2:
            # if the segments are distant - leave them
            if are_segments_at_least_two_pixels_apart(seg1, seg2):
                continue
            # the segments are too close
            else:
                # if seg1 is larger - keep it and remove seg2
                if seg1.length() >= seg2.length():
                    # this is to protect the method remove
                    if seg2 in line2:
                        line2.remove(seg2)
                else:
                    # seg2 is larger - keep it and remove seg1
                    #
                    # this is to protect the method remove
                    if seg1 in line1:
                        line1.remove(seg1)

    return line1, line2


def eliminate_too_close_segments(lines):
    # This kind of loop allows us to compare each couple of lines just ONCE
    # so we will not repeat the comparison more than then one time
    for i in range(len(lines)):
        line1 = lines[i]
        rest_of_lines = lines[i + 1:]

        for j in range(len(rest_of_lines)):
            line2 = rest_of_lines[j]

            line1, line2 = eliminate_too_close_segments_between_two_lines(line1, line2)

    return lines


def unite_line_segments(line):
    # line has no gaps
    if len(line) == 1:
        return line

    new_segments_of_line = []

    current_segment = line[0]

    united_segment = Line(current_segment.rho, current_segment.theta, current_segment.points)
    united_segment.set_segment()
    new_segments_of_line.append(united_segment)

    for i in range(len(line)):
        if line[i].is_gap():
            gap = line[i]
            prev_seg = line[i - 1]
            next_seg = line[i + 1]

            # if it is a very small "mistake" - unite
            if gap.length() + prev_seg.length() + next_seg.length() <= 10:
                united_segment.points += next_seg.points

            # if gap is really small compared to its neighbour segments
            elif gap.length() <= (prev_seg.length() + next_seg.length()) / 10:
                united_segment.points += next_seg.points

            # the gap is significant
            else:
                # wrap_up previous united segment
                united_segment.reset_start_and_end_points()

                # create a new segment
                united_segment = Line(next_seg.rho, next_seg.theta, next_seg.points)
                united_segment.set_segment()
                new_segments_of_line.append(united_segment)

        # wrap_up lasts united segment
        united_segment.reset_start_and_end_points()

    return new_segments_of_line


def unite_lines_segments(lines):
    return [unite_line_segments(line) for line in lines]


def remove_duplicate_lines(lines):
    for i in range(len(lines)):
        line1 = lines[i]
        # means it was deleted previously
        if line1 is None:
            continue

        rest_of_lines = lines[i + 1:]

        for j in range(len(rest_of_lines)):
            line2 = rest_of_lines[j]
            # means it was deleted previously
            if line2 is None:
                continue

            if are_lines_the_same(line1, line2):
                k = lines.index(line2)
                lines[k] = None

    return [line for line in lines if line is not None]


def remove_too_short_segments(segments, threshold):
    return [seg for seg in segments if seg.length() > math.ceil(threshold)]
