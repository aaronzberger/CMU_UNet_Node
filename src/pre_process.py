#!/home/aaron/py27/bin/python

import numpy as np
from transformations import euler_matrix
from math import radians
from numpy_pc2 import pointcloud2_to_xyzi_array

def pcl_to_bev(point_cloud):
    pcl = pointcloud2_to_xyzi_array(point_cloud)

    bev = np.zeros((400, 400, 24), dtype='float32')

    # If there is tilt in the sensor, use the code below & input degrees tilt
    # pitch = radians(3.0)
    # R_extrinsic = euler_matrix(0, pitch, 0)[0:3, 0:3]

    # Width, length, height
    mins = [0.0, -5.0, -1.6]
    maxes = [10.0, 5.0, 0.32]

    resolutions = np.array([
        abs(maxes[0] - mins[0]) / 400,
        abs(maxes[1] - mins[1]) / 400,
        abs(maxes[2] - mins[2]) / 24,
    ])

    pcl = pcl[
        (pcl[:, 0] > mins[0]) and
        (pcl[:, 0] < maxes[0]) and
        (pcl[:, 1] > mins[1]) and
        (pcl[:, 1] < maxes[1]) and 
        (pcl[:, 2] > mins[2]) and
        (pcl[:, 2] < maxes[2])
    ]

    pcl -= mins
    pcl /= resolutions

    coordinates = pcl.astype('int')
    bev[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

    return bev
