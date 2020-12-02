#!/home/aaron/py27/bin/python

import numpy as np
from transformations import euler_matrix
from math import radians
from numpy_pc2 import pointcloud2_to_xyz_array
import torch

class Pre_Process:
    def __init__(self, width, length, height):
        '''Set the tuples of min and max for width, length, and height of Lidar points'''
        self.width = width
        self.length = length
        self.height = height

    def get_scaling_values(self):
        '''Returns the min and max values for width, length, and height (as tuples)'''
        return self.width, self.length, self.height

    def pcl_to_bev(self, point_cloud):
        '''
        Converts a PointCloud2 object to an image, so it may be passed into UNet

        Parameters:
            point_cloud (PointCloud2): a point cloud retreived from the Lidar

        Returns:
            numpy.ndarray: A black image with white pixels where Lidar points were
        '''
        pcl = pointcloud2_to_xyz_array(point_cloud)

        bev = np.zeros((400, 400, 24), dtype='float32')

        # If there is tilt in the sensor, use the code below & input degrees tilt
        # pitch = radians(3.0)
        # R_extrinsic = euler_matrix(0, pitch, 0)[0:3, 0:3]

        scaling = np.array([
            abs(self.width[0] - self.width[1]) / 400,
            abs(self.length[0] - self.length[1]) / 400,
            abs(self.height[0] - self.height[1]) / 24,
        ])

        # Eliminate Lidar points that are out of bounds
        pcl = pcl[
            (pcl[:, 0] > self.width[0]) &
            (pcl[:, 0] < self.width[1]) &
            (pcl[:, 1] > self.length[0]) &
            (pcl[:, 1] < self.length[1]) & 
            (pcl[:, 2] > self.height[0]) &
            (pcl[:, 2] < self.height[1])
        ]

        mins = np.array([self.width[0], self.length[0], self.height[0]])

        pcl -= mins
        pcl /= scaling

        coordinates = pcl.astype('int')
        bev[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

        return bev
