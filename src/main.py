#!/home/aaron/py27/bin/python

import rospy
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
from unet import UNet
from loss import ClassificationLoss
from sensor_msgs import point_cloud2
import post_process
from msg import line_list
import std_msgs.msg
from pre_process import pcl_to_bev

class Lidar_Process_Node:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net, self.loss_fn = self.build_model(device=self.device)

        self.sub_velodyne = rospy.Subscriber('/velodyne_points', point_cloud2, self.velodyne_callback)

        self.pub_lines = rospy.Publisher('/unet_lines', line_list)


    def build_model(self, device):
        out_channels = 1
        net = UNet()

        loss_fn = ClassificationLoss(device)

        net = net.to(device)
        loss_fn = loss_fn.to(device)

        return net, loss_fn

    def velodyne_callback(self, data):
        input = pcl_to_bev(data)

        # Pass through the model and apply a sigmoid
        prediction_map = torch.sigmoid(self.net(input))

        # Extract lines from UNet output
        left_lines, right_lines = post_process.get_lines(prediction_map)
        left_lines, right_lines = np.array(left_lines, np.uint16), np.array(right_lines, np.uint16)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()

        msg = line_list()
        msg.header = header
        msg.left_lines = left_lines
        msg.right_lines = right_lines

        self.pub_lines.publish(msg)


if __name__ == "__main__":
    rospy.init_node('unet')

    lidar_process_node = Lidar_Process_Node()

    rospy.spin()