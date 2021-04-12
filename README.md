# LPCC-Net: Local point cloud completion network for 3D object detection

## Introduction
This repository contains the pytorch implementation of **LPCC-NET** introducted in ICME21 paper **"LPCC-NET: RGB GUIDED LOCAL POINT CLOUD COMPLETION FOR OUTDOOR 3D
OBJECT DETECTION."** From which, we proposed an RGB-guided local point cloud completion network, that aims to improve off-the-shelf 3D object detectors by selectively densifying the collected point clouds.And our proposed method directly predicts the existence of points in 3D space around input points. Also, we create a semi-dense labeled local points completion dataset and design a new loss for training the network in a semi-supervised manner. Extensive experiments show that the proposed method can produce reasonable and accurate dense 3D point clouds from sparse inputs, improving off-the-shelf 3D object detectors on the KITTI 3D detection benchmark.

<img src='./docs/intro.png' width=600>




