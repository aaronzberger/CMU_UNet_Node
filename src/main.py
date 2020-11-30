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
        #Convert lidar frame to 400 x 400

        # Pass through the model and apply a sigmoid
        prediction_map = torch.sigmoid(net(input))

        lines = post_process.get_lines(prediction_map)
        lines = np.array(lines, np.uint16)

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()

        msg = line_list()
        msg.header = header
        msg.lines = lines

        self.pub_lines.publish(msg)


if __name__ == "__main__":
    rospy.init_node('unet')

    lidar_process_node = Lidar_Process_Node()

    rospy.spin()