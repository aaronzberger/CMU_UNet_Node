#!/home/aaron/py27/bin/python

import rospy
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
from unet import UNet
from sensor_msgs.msg import PointCloud2
from cmu_unet.msg import line_list, line
import std_msgs.msg
from pre_process import Pre_Process
from post_process import Post_Process

class Lidar_Process_Node:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = self.build_model()

        self.pre_process = Pre_Process((0.0, 10.0), (-5.0, 5.0), (-1.6, 0.32))
        self.post_process = Post_Process((0.0, 10.0), (-5.0, 5.0))

        self.sub_velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback)
        self.pub_lines = rospy.Publisher('/unet_lines', line_list, queue_size=1)

    def build_model(self):
        '''Build the UNet model'''
        net = UNet()

        # Load the weights as trained with 120 epochs
        net.load_state_dict(torch.load("/home/aaron/catkin_ws/src/cmu_unet/state_dict/120epoch", map_location=self.device))

        net = net.to(self.device)
        net.eval()

        return net

    def velodyne_callback(self, data):
        '''Callback function for the Lidar, where point clouds are converted to lines and published'''
        # Pre-process the point cloud into an image
        input = torch.from_numpy(self.pre_process.pcl_to_bev(data))
        
        # Save an image of the point cloud:
        # pcl = np.array(input.cpu() * 255, dtype=np.uint8).transpose(2, 0, 1)
        # image_pcl = np.amax(pcl, axis=0)
        # cv.imwrite("/home/aaron/velo_frame.jpg", image_pcl)

        # Convert the image to tensor format
        input = input.permute(2, 0, 1).unsqueeze(0)

        # Forward Prop
        with torch.no_grad():
            prediction_map = torch.sigmoid(self.net(input))
        
        # Post-process the image into lines
        lines = self.post_process.extract_lines(prediction_map)

        # Save an image of the network output:
        # prediction_image = np.array(prediction_map[0].cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
        # prediction_image = cv.cvtColor(prediction_image, cv.COLOR_GRAY2BGR)

        # if lines is not None:
        #     for line in lines:
        #         cv.line(prediction_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2, cv.LINE_AA)
        
        # cv.imwrite("/home/aaron/unet_output.jpg", prediction_image)

        # Convert the lines from image pixel coordinates to robot coordinates
        lines_wrt_robot = self.post_process.segment_image_to_robot(lines)

        # Construct a line_list message to publish, containing an array of line messages
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()

        msg = line_list()
        msg.header = header
        msg.lines = lines_wrt_robot
        self.pub_lines.publish(msg)


if __name__ == "__main__":
    rospy.init_node('unet', log_level=rospy.INFO)

    lidar_process_node = Lidar_Process_Node()

    rospy.spin()