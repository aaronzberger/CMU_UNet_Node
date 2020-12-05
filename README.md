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
- [Credits](#Credits)

## Details

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
- __PATH PLANNING__ (Not yet on Github)
- __DRIVING__ (Not yet on Github)
## Usage

## Dataset

## Training

## Credits
