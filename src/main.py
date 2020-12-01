#!/home/aaron/py27/bin/python

import rospy
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
from unet import UNet
from sensor_msgs.msg import PointCloud2
import post_process
from cmu_unet.msg import line_list
import std_msgs.msg
from pre_process import pcl_to_bev

class Lidar_Process_Node:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = self.build_model()

        rospy.loginfo("Successfullly built UNet model")

        self.sub_velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback)

        self.pub_lines = rospy.Publisher('/unet_lines', line_list, queue_size=1)


    def build_model(self):
        out_channels = 1
        net = UNet()

        net.load_state_dict(torch.load("/home/aaron/catkin_ws/src/cmu_unet/src/state_dict/120epoch", map_location=self.device))

        net = net.to(self.device)

        net.eval()

        return net

    def velodyne_callback(self, data):
        rospy.loginfo("Received a velodyne frame")

        # Pre-process
        input = pcl_to_bev(data)

        pcl = np.array(input.cpu() * 255, dtype=np.uint8)
        image_pcl = np.amax(pcl, axis=0)
        cv.imwrite("/home/aaron/velo_frame.jpg", image_pcl)

        rospy.loginfo("Wrote an image of the point cloud")

        # Pass through the model and apply a sigmoid
        with torch.no_grad():
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
    rospy.init_node('unet', log_level=rospy.INFO)

    lidar_process_node = Lidar_Process_Node()

    rospy.spin()