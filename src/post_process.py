import math
from shapely.geometry import LineString
import numpy as np
import cv2 as cv
from CMU_UNet_Node.msg import line_2pts


class Post_Process:
    def __init__(self, width, length, layer_2_weight=0.5,
                 layer_3_weight=0.2, length_weight=1.5):
        '''Set the min and max Lidar values for width and height,
        and the weights used for the Bresenham algorithm for
        scoring the lines'''
        self.width = width
        self.length = length

        self.layer_2_weight = layer_2_weight
        self.layer_3_weight = layer_3_weight
        self.length_weight = length_weight

    def line_to_point_distance(self, x, y, a, b, c):
        '''Determine the distance from a cartesian point to a
        standard form line'''
        return (abs(a * x + b * y + c)) / (math.sqrt(a * a + b * b))

    def absolute_angle_to_line(self, l1):
        '''Determine the angle from the origin to the given line
        (in two-points format)'''
        x1, y1 = l1[0][0], l1[0][1]
        x2, y2 = l1[0][2], l1[0][3]

        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy

        a = (dy * (200 - y1) + dx * (200 - x1)) / det
        closest_x, closest_y = x1 + a*dx, y1 + a*dy

        return math.atan2(closest_y - 200, closest_x - 200)

    def segment_to_standard(self, l1):
        '''Convert a line from two-points format to standard form'''
        x1, y1 = l1[0][0], l1[0][1]
        x2, y2 = l1[0][2], l1[0][3]

        a = y1 - y2
        b = x2 - x1
        c = (x1-x2) * y1 + (y2-y1) * x1

        return a, b, c

    def segments_intersect(self, l1, l2):
        '''Determine whether to line segments intersect
        (in two-points format)'''
        line_1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
        line_2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])

        return line_1.intersects(line_2)

    def lines_are_close(self, l1, l2, dist, theta_deg):
        '''
        Determine whether two lines are close enough to be clustered together

        Parameters:
            l1 (array of array of 4 ints): the first line
            l2 (array of array of 4 ints): the second line
            dist (double): the maximum distance between the lines (in pixels)
            theta_deg (double): the maximum angle apart the lines must be

        Returns:
            bool: Whether the two provided lines are close,
            given the parameters
        '''
        a1, b1, c1 = self.segment_to_standard(l1)
        line_1_dist = self.line_to_point_distance(200, 200, a1, b1, c1)
        line_1_angle = self.absolute_angle_to_line(l1)

        a2, b2, c2 = self.segment_to_standard(l2)
        line_2_dist = self.line_to_point_distance(200, 200, a2, b2, c2)
        line_2_angle = self.absolute_angle_to_line(l2)

        x1, y1 = l1[0][0], l1[0][1]
        x2, y2 = l1[0][2], l1[0][3]

        x3, y3 = l2[0][0], l2[0][1]
        x4, y4 = l2[0][2], l2[0][3]

        pt1_line2 = self.line_to_point_distance(x1, y1, a2, b2, c2)
        pt2_line2 = self.line_to_point_distance(x2, y2, a2, b2, c2)
        pt3_line1 = self.line_to_point_distance(x3, y3, a1, b1, c1)
        pt4_line1 = self.line_to_point_distance(x4, y4, a1, b1, c1)

        min_dist = pt1_line2
        min_dist = pt2_line2 if pt2_line2 <= min_dist else min_dist
        min_dist = pt3_line1 if pt3_line1 <= min_dist else min_dist
        min_dist = pt4_line1 if pt4_line1 <= min_dist else min_dist

        return self.segments_intersect(l1, l2) or min_dist < dist or \
            (abs(line_1_dist - line_2_dist) < dist and
                abs(line_1_angle - line_2_angle) < math.radians(theta_deg))

    def line_score(self, l1, pred_map):
        '''
        Determines a score for the line that represents
        how well the line fits the UNet output

        Parameters:
            l1 (array of array of 4 ints): the line to be scored
            pred_map (numpy.ndarray): the sigmoid of the UNet output

        Returns:
            double: the score for the line
        '''
        x1, y1 = l1[0][0], l1[0][1]
        x2, y2 = l1[0][2], l1[0][3]

        layer_1_score = 0
        layer_2_score = 0
        layer_3_score = 0

        # This is the number of points for each layer
        layer_1_points = 0
        layer_2_points = 0
        layer_3_points = 0

        # Bresenham Algorithm
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        count_in_x = True if dx > dy else False

        while not (x1 == x2 and y1 == y2):
            # Score the current pixel
            layer_1_score += pred_map[y1, x1] - 5
            layer_1_points += 1

            if count_in_x:
                if x1 <= 398 and x1 >= 2:
                    layer_2_score += pred_map[y1, x1+1]
                    layer_2_score += pred_map[y1, x1-1]
                    layer_2_points += 2
                if x1 <= 397 and x1 >= 3:
                    layer_3_score += pred_map[y1, x1+2]
                    layer_3_score += pred_map[y1, x1-2]
                    layer_3_points += 2
            else:
                if y1 <= 398 and y1 >= 2:
                    layer_2_score += pred_map[y1+1, x1]
                    layer_2_score += pred_map[y1-1, x1]
                    layer_2_points += 2
                if y1 <= 397 and y1 >= 3:
                    layer_3_score += pred_map[y1+2, x1]
                    layer_3_score += pred_map[y1-2, x1]
                    layer_3_points += 2

            # Move to the next pixel
            e2 = err << 1
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        final_layer_1 = layer_1_score / float(layer_1_points)
        final_layer_2 = layer_2_score / float(layer_2_points)
        final_layer_3 = layer_3_score / float(layer_3_points)

        final_score = final_layer_1 \
            + (final_layer_2 * self.layer_2_weight) \
            + (final_layer_3 * self.layer_3_weight)

        line_length = math.sqrt(dx ** 2 + dy ** 2)
        final_score += (line_length * self.length_weight)

        return final_score

    def segment_image_to_robot(self, lines):
        '''
        Convert an array of lines in image pixel coordinates to
        robot coordinates

        Parameters:
            lines (array): an array of lines

        Returns:
            array of line.msg: an array of line message objects
            containing the converted points
        '''
        if lines is None:
            return []

        mins = np.array([self.width[0], self.length[0]])

        scaling = np.array([
            abs(self.width[0] - self.width[1]) / 400,
            abs(self.length[0] - self.length[1]) / 400,
        ])

        new_lines = []

        for line in lines:
            new_line = line_2pts()
            new_line.x1 = line[0][0] * scaling[1] + mins[1]
            new_line.y1 = line[0][1] * scaling[0] + mins[0]
            new_line.x2 = line[0][2] * scaling[1] + mins[1]
            new_line.y2 = line[0][3] * scaling[0] + mins[0]
            new_lines.append(new_line)

        return new_lines

    def extract_lines(self, prediction_map):
        '''
        Determine, in two-points format, where there are
        clear lines in the image that represent rows

        Parameters:
            prediction_map (torch.Tensor): the sigmoid of the UNet output

        Returns:
            array: an array of lines in pixel coordinates
        '''
        # CxWxH to WxHxC and convert to
        # grayscale image format (0-255 and 8-bit int)
        prediction_gray = np.array(
            prediction_map * 255,
            dtype=np.uint8).transpose(1, 2, 0)

        # Blurring
        kernel = np.ones((4, 4), np.float32)
        blurred = cv.filter2D(prediction_gray, -1, kernel)

        # Hough Transform
        lines = cv.HoughLinesP(blurred, 2, (np.pi / 180), 150, None, 25, 50)

        if lines is None:
            return None

        # Clustering
        # Determine whether each line is not yet clustered (1 or 0)
        tracker = np.ones(len(lines))

        # For each line, see which other lines it matches
        # and put those in a group (O(n) = n^2)
        clustered_lines = []
        for i in range(0, len(lines)):
            if tracker[i] == 1:
                group = []
                group.append(lines[i])
                for j in range(1, len(lines)):
                    if tracker[j] == 1:
                        for line in group:
                            if self.lines_are_close(line, lines[j], 20, 10):
                                group.append(lines[j])
                                tracker[j] = 0
                                break
                tracker[i] = 0
                clustered_lines.append(group)

        # Selecting representative from each cluster
        final_lines = []
        if len(clustered_lines) > 0:
            for group in clustered_lines:
                best_line = group[0]
                best_score = self.line_score(group[0], blurred)
                for i in range(1, len(group)):
                    line_score = self.line_score(group[i], blurred)
                    if line_score >= best_score:
                        best_line = group[i]
                        best_score = line_score
                final_lines.append(best_line)

        return final_lines
