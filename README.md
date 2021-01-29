# ROS U-Net Node for Autonomous Lidar Navigation

Read raw point clouds and extract lines using a Convolutional Neural Network called U-Net!

![Node Pipeline](https://user-images.githubusercontent.com/35245591/101233100-410ebd00-3684-11eb-93b5-8ea502669e5d.png)

## Table of Contents
- [Details](#Details)
- [U-Net](#U-Net)
- [Pipeline](#Pipeline)
- [Usage](#Usage)
- [Dataset](#Dataset)
- [Training](#Training)
  - [Point Cloud Trimming](#Point-Cloud-Trimming)
  - [Resolution](#Resolution)
  - [Training Time](#Training-Time)
  - [Loss and Prediction Maps](#Loss-and-Prediction-Maps)
- [Acknowledgements](#Acknowledgements)

## Details
The code is separated into three sections for each Lidar frame:

- First, the point cloud goes through [pre-processing](https://github.com/aaronzberger/CMU_UNet_Node/blob/main/src/pre_process.py), where it is converted from an unordered array of \[X, Y, Z] pairs to a \[400 x 400] image with 24 channels (to represent the height of each pixel).
- Next, the \[1, 24, 400, 400] Tensor is [passed through](https://github.com/aaronzberger/CMU_UNet_Node/blob/966c0ca2701703849b61900425df3c33d7be1dee/src/main.py#L55-L60) the trained U-Net model, which outputs a prediction map.
- Lastly, the prediction map goes through [post-processing](https://github.com/aaronzberger/CMU_UNet_Node/blob/main/src/post_process.py), where a [Probabalistic Hough Transform](https://github.com/aaronzberger/CMU_UNet_Node/blob/966c0ca2701703849b61900425df3c33d7be1dee/src/post_process.py#L213) is applied, followed by [clustering](https://github.com/aaronzberger/CMU_UNet_Node/blob/966c0ca2701703849b61900425df3c33d7be1dee/src/post_process.py#L218-L236) and [selection](https://github.com/aaronzberger/CMU_UNet_Node/blob/966c0ca2701703849b61900425df3c33d7be1dee/src/post_process.py#L238-L249).

These final lines are then published via the `/unet_lines` ROS topic.

## U-Net
U-Net is a Convolutional Neural Network architecture built for Biomedical Image Segementation (specifically, 
segmentation of neuronal structures in electron microscopic stacks). 

> "The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization."

The precise architecture is shown here:

![U-Net Architecture](https://user-images.githubusercontent.com/35245591/101233308-e37b7000-3685-11eb-8318-eedc7b904ef5.png)

The official paper for U-Net can be found [here](https://arxiv.org/abs/1505.04597).

## Pipeline
This node is part of a larger autonomous navigation pipeline that is currently being developed. 

This node is the __VISION__ node represented in the full pipeline below:

![Full Pipeline](https://user-images.githubusercontent.com/35245591/101234307-10cb1c80-368c-11eb-99de-7afccb2e8909.png)

Each node is a separate ROS node, each receiving and publishing relevant data.

Code locations for the other nodes are listed below:
- [__ROW TRACKING__](https://github.com/aaronzberger/CMU_EKF_Node)
- [__PATH PLANNING__](https://github.com/aaronzberger/CMU_Path_Planning_Node)
- __DRIVING__ (Not yet on Github)

## Usage
It's very easy to get this node up and running! Simply complete these steps:
- Replace the [first line](https://github.com/aaronzberger/CMU_UNet_Node/blob/main/src/main.py#L1) of `main.py` with your Python interpreter path
- Add in the path to your trained state dictionary [here](https://github.com/aaronzberger/CMU_UNet_Node/blob/b9bf561f837066faad402f198aecc72eda709062/src/main.py#L35)
- If you wish to save images of the [point clouds](https://github.com/aaronzberger/CMU_UNet_Node/blob/b9bf561f837066faad402f198aecc72eda709062/src/main.py#L50) or [prediction maps](https://github.com/aaronzberger/CMU_UNet_Node/blob/b9bf561f837066faad402f198aecc72eda709062/src/main.py#L70), add in paths
- Make sure you have all necessary packages installed
- Start op your `roscore` and run:
  
  `rosrun cmu_unet main.py`

## Dataset
A sizeable dataset containing point clouds and masks (ground truths) is required for training.

Python scripts for extracting point clouds from bags and saving them as `.np` files are provided [here](https://github.com/jnmacdnld/ag_lidar_navigation/tree/bev/srcs).

A custom labeler for labeling the point clouds with ground truth lines is also provided in the code above.

## Training
The code for training, labeling, and testing loss functions is available [here](https://github.com/aaronzberger/CMU_Lidar_Navigation).

Check out that repository for details on training with different loss functions.

#### Point Cloud Trimming
In pre-processing, the point clouds need to be trimmed to ensure all points can fit within the \[24, 400, 400] image. This means there must be minimums and maximums for width, length, and height.

- Width represents the X direction of the robot (forward and backwards)
- Length represents the Y direction of the robot (side to side)
- Height represents the Z direction of the robot (up and down)

These mins and maxes are [specified as tuples](https://github.com/aaronzberger/CMU_UNet_Node/blob/b9bf561f837066faad402f198aecc72eda709062/src/main.py#L24-L25) in `main.py` and should be adjusted for your use.

For example, if you wish to only count points ahead of the robot, you may use a width range of (0.0, 10.0). If you wish to use nearly all points ahead and behind the robot, you may use a width range of (-10.0, 10.0).

#### Resolution
Point clouds are received as raw point arrays, where as distance increases, so does the sparsity of points. Since we are representing the point clouds in images in the Birds Eye View representation, each \[X, Y, Z] pair must be matched to exactly one pixel (and channel). Therefore, there may be two points that map to the same pixel if their Cartesian coordinates are close enough.

We can therefore see that resolution may affect the performance of the model. Using larger images (say \[600, 600, 32]), we may increase the accuracy of the model by mapping closer points to separate pixels. Keep in mind the size of the image also affects the speed and memory usage of the model.

Depending on your [Point Cloud Trimming](#Point-Cloud-Trimming), you may wish to decrease or increase the size of the model. For example, if you only use a small area around the robot, it may only be necessary to use a small image size (say, \[256, 256, 18]).

The pixel mapping is done [here](https://github.com/aaronzberger/CMU_UNet_Node/blob/b9bf561f837066faad402f198aecc72eda709062/src/pre_process.py#L36-L55), in pre-processing.

#### Training Time
Training on an NVIDIA RTX 2080 with 2,087 point clouds, with a batch size of 4, each epoch takes around 2.9 mins.

## Acknowledgements
- John Macdonald for Lidar pre-processing and labeling
- Olaf Ronneberger, Philipp Fischer, Thomas Brox for the [U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- Jon Binney for [`numpy_pc2.py`](https://github.com/dimatura/pypcd/blob/master/pypcd/numpy_pc2.py)
