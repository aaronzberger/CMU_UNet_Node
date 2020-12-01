import math
from shapely.geometry import LineString
import numpy as np
import cv2 as cv


# HELPER FUNCTIONS
    
def distance(x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    return math.sqrt(dx**2 + dy**2)

def angle_to_line(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    
    a = (dy*(200-y1)+dx*(200-x1))/det
    closestX, closestY = x1+a*dx, y1+a*dy
    
    return math.atan2(closestY - 200, closestX - 200)

def line_to_point(x, y, a, b, c):
    return (abs(a * x + b * y + c)) / (math.sqrt(a * a + b * b))

def line_to_line(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.distance(line2)
    
def segment_to_standard(l1):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    a = y1 - y2
    b = x2 - x1
    c = (x1-x2)*y1 + (y2-y1)*x1
    
    return a, b, c

def intersect(l1, l2):
    line1 = LineString([(l1[0][0], l1[0][1]), (l1[0][2], l1[0][3])])
    line2 = LineString([(l2[0][0], l2[0][1]), (l2[0][2], l2[0][3])])
    
    return line1.intersects(line2)

def lines_are_close(l1, l2, dist, theta_deg):
    a1, b1, c1 = segment_to_standard(l1)
    line1Dist = line_to_point(200, 200, a1, b1, c1)
    line1Angle = angle_to_line(l1)
    
    a2, b2, c2 = segment_to_standard(l2)
    line2Dist = line_to_point(200, 200, a2, b2, c2)
    line2Angle = angle_to_line(l2)
    
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    x3, y3 = l2[0][0], l2[0][1]
    x4, y4 = l2[0][2], l2[0][3]
    
    shortestDistance = line_to_point(x1, y1, a2, b2, c2)
    shortestDistance = line_to_point(x2, y2, a2, b2, c2) if line_to_point(x2, y2, a2, b2, c2) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x3, y3, a1, b1, c1) if line_to_point(x3, y3, a1, b1, c1) <= shortestDistance else shortestDistance
    shortestDistance = line_to_point(x4, y4, a1, b1, c1) if line_to_point(x4, y4, a1, b1, c1) <= shortestDistance else shortestDistance

    return intersect(l1, l2) or shortestDistance < dist or (abs(line1Dist - line2Dist) < dist and abs(line1Angle - line2Angle) < math.radians(theta_deg))


# Arugument pred_map should be the simgoid of the output of the UNet model
def line_score(l1, pred_map, layer2Weight=0.5, layer3Weight=0.0, lengthWeight=1.0):
    x1, y1 = l1[0][0], l1[0][1]
    x2, y2 = l1[0][2], l1[0][3]
    
    layer1Score = 0
    layer2Score = 0
    layer3Score = 0
    
    # This is the number of points for each layer
    layer1Points = 0
    layer2Points = 0
    layer3Points = 0

    # Bresenham Algorithm
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    countInX = True if dx > dy else False

    while not (x1 == x2 and y1 == y2):        
        # Score the current pixel
        layer1Score += pred_map[y1, x1] - 5
        layer1Points += 1

        if countInX:
            if x1 <= 398 and x1 >= 2:
                layer2Score += pred_map[y1, x1+1]
                layer2Score += pred_map[y1, x1-1]
                layer2Points += 2
            if x1 <= 397 and x1 >= 3:
                layer3Score += pred_map[y1, x1+2]
                layer3Score += pred_map[y1, x1-2]
                layer3Points += 2
        else:
            if y1 <= 398 and y1 >= 2:
                layer2Score += pred_map[y1+1, x1]
                layer2Score += pred_map[y1-1, x1]
                layer2Points += 2
            if y1 <= 397 and y1 >= 3:
                layer3Score += pred_map[y1+2, x1]
                layer3Score += pred_map[y1-2, x1]
                layer3Points += 2

        # Move to the next pixel
        e2 = err << 1
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    finalLayer1 = layer1Score / float(layer1Points)
    finalLayer2 = layer2Score / float(layer2Points)
    finalLayer3 = layer3Score / float(layer3Points)
    
    final_score = finalLayer1 + (finalLayer2 * layer2Weight) + (finalLayer3 * layer3Weight)
    
    line_length = math.sqrt(dx ** 2 + dy ** 2)
    final_score += (line_length * lengthWeight)

    return final_score

    
def distance_to_origin(l1):
    a, b, c = segment_to_standard(l1)
    return line_to_point(200, 200, a, b, c)

def image_to_polar(lines):
    #TODO:
    # Convert the lines from x1, y1, x2, y2 to distance and theta
    left_lines = []
    right_lines = []
    return left_lines, right_lines

def get_lines(prediction_map):
    # CxWxH to WxHxC and convert to grayscale image format (0-255 and 8-bit int)
    prediction_gray = np.array(prediction_map * 255, dtype=np.uint8).transpose(1, 2, 0)

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

    # For each line, see which other lines it matches and put those in a group (O(n) = n^2)
    newLines = []
    for i in range(0, len(lines)):
        if tracker[i] == 1:
            group = []
            group.append(lines[i])
            for j in range(1, len(lines)):
                if tracker[j] == 1:
                    for line in group:
                        if lines_are_close(line, lines[j], 20, 10):
                            group.append(lines[j])
                            tracker[j] = 0
                            break
            tracker[i] = 0
            newLines.append(group)
    
    # Selecting representative from each cluster
    finalLines = []
    if len(newLines) > 0:
        for group in newLines:
            bestLine = group[0]
            bestScore = line_score(group[0], blurred, 0.5, 0.2, 1.5)
            for i in range(1, len(group)):
                lineScore = line_score(group[i], blurred, 0.5, 0.2, 1.5)
                if lineScore >= bestScore:
                    bestLine = group[i]
                    bestScore = lineScore
            finalLines.append(bestLine)
            # finalLines.append(bestLine[0])

    return image_to_polar(lines)