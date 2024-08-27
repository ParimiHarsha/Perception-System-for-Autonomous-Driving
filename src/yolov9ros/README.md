# YOLOv9 for Autonomous Driving with Custom Classes

## ⚠️&nbsp;&nbsp;Cautions
> This repository currently under development

## Preface:

YOLOv9, released in February 2024, marks a significant advancement in the YOLO (You Only Look Once) series, a family of object detection models that have revolutionized the field of computer vision. According to the project research team, YOLOv9 achieves a higher mAP than existing popular YOLO models such as YOLOv8, YOLOv7, and YOLOv5, when benchmarked against the MS COCO dataset.

## Problem Statement:

YOLOv9 was trained on the [COCO Dataset](https://cocodataset.org/#home) which has 80 classes. It excels at detecting those classes. Our goal is to add new custom classes to the existing COCO classes and detect all classes effectively.

Our specific use case is in the field of autonomous driving (AD). We aim to detect additional classes of cones and traffic signs. Specifically, we want to detect the following 15 classes in addition to the existing 80 COCO classes:

```python
0: 'Speed limit: 70'
1: 'Speed limit: 75'
2: 'Speed limit: 30'
3: 'Speed limit: 35'
4: 'Speed limit: 40'
5: 'Speed limit: 45'
6: 'Speed limit: 50'
7: 'Speed limit: 55'
8: 'Speed limit: 60'
9: 'Speed limit: 65'
14: 'cones'
```
We have curated datasets to help us train for the above classes:

- For the COCO classes, we used the [COCO Minitrain](https://github.com/giddyyupp/coco-minitrain) dataset. This is a curated mini training set (25K images ≈ 20% of `train2017`) for COCO, useful for hyperparameter tuning and reducing the cost of ablation experiments.
- For the signs and traffic lights, we created a dataset by capturing images across Texas. GitHub repo for the dataset and instructions: [Traffic-Sign-and-Light-Detection](https://github.com/ava-share/Traffic-Sign-and-Light-Detection)
- For the cones, we used the publicly available dataset from Roboflow: [Traffic Cones Dataset](https://universe.roboflow.com/robotica-xftin/traffic-cones-4laxg)
- Custom a nnotated dataset for cones and signs 

## Overview

This project contains scripts to perform object detection using YOLOv9 on images and publish it in ROS. The project includes:

- Extracting images from a ROS bag file.
- Running inference on extracted images using YOLOv9.
- Publishing detected objects as 3D bounding boxes.

## Prerequisites

- ROS (Robot Operating System) installed and properly set up.
- Python 3.x
- Required Python packages:
  - `cv2`
  - `rosbag`
  - `cv_bridge`
  - `torch`
  - `ultralytics`
  - `ros_numpy`
  - `jsk_recognition_msgs`
  - `PIL` (Pillow)
  - `numpy`
## Trained Model
https://drive.google.com/file/d/1UAX-7jSXQJcyRdumn8iXmwjfJxxyC9Tw/view?usp=sharing
## Setup

### Install ROS and Dependencies

1. Follow the official [ROS installation guide](http://wiki.ros.org/ROS/Installation) for your operating system.
2. Install required ROS packages:

   ```sh
   sudo apt-get install ros-<distro>-cv-bridge ros-<distro>-sensor-msgs ros-<distro>-rosbag ros-<distro>-image-transport ros-<distro>-jsk-recognition-msgs
   ```

   Replace `<distro>` with your ROS distribution (e.g., `noetic`).

3. Install required Python packages:

   ```sh
   pip install opencv-python-headless torch ultralytics ros_numpy pillow numpy
   ```

### Clone the Repository

```sh
git clone https://github.com/ParimiHarsha/YOLOv9-for-Autonomous-Driving-with-Custom-Classes.git
```

## Scripts

### 1. Extract Images from a ROS Bag

The script `bag_to_images.py` extracts images from a specified topic in a ROS bag file and saves them to an output directory.

#### Usage

```sh
python bag_to_images.py
```

#### Script Arguments

- `bag_file`: Path to the ROS bag file.
- `image_topic`: ROS topic from which to extract images.
- `output_dir`: Directory where extracted images will be saved.
- `frequency`: Frequency of image extraction (default: 5).

### 2. Run Inference on ROS Image Topic

The script `inference_ros.py` subscribes to a ROS image topic, performs inference using YOLOv9, and publishes detected objects as 3D bounding boxes.

#### Usage

```sh
python inference_ros.py --weights yolov9c.pt --reg_weights weights/resnet18.pkl --model_select resnet18
```

#### Script Arguments

- `weights`: Path to the YOLOv9 weights file.
- `reg_weights`: Path to the regression weights file.
- `model_select`: Model selection for regression.

### 3. Object Detection Live

The script `objectDetectionLive.py` subscribes to a ROS image topic, performs live object detection using YOLOv9, and publishes the results.

#### Usage

```sh
rosrun <package_name> objectDetectionLive.py
```

## Running the Project

1. **Extract Images from a ROS Bag**

   ```sh
   python bag_to_images.py
   ```

2. **Run Inference on Extracted Images**

   ```sh
   python inference_ros.py --weights yolov9c.pt --reg_weights weights/resnet18.pkl --model_select resnet18
   ```

3. **Live Object Detection**

   ```sh
   rosrun <package_name> objectDetectionLive.py
   ```

## Troubleshooting

- Ensure ROS is properly sourced:

  ```sh
  source /opt/ros/<distro>/setup.bash
  ```

- Ensure the paths to the weight files and model files are correct.
- Ensure all required packages are installed and compatible with your ROS version.


## Methodology

Using the LabelBox tool we have annotated images for our custom classes, `cones` and the `traffic signs`. Then we have trained the YOLOv9 model to detect the new classes along with the existing coco classes.

## References

- [Train YOLOv9 Model](https://blog.roboflow.com/train-yolov9-model/)
- [Fine-tuning YOLOv9](https://learnopencv.com/fine-tuning-yolov9/#aioseo-experiment-1-freezing-the-backbone-lower-learning-rate-at-0-001)
- [Adding Classes to YOLOv8](https://y-t-g.github.io/tutorials/yolov8n-add-classes/)
- [COCO Dataset](https://cocodataset.org/#home)
- [COCO Minitrain](https://github.com/giddyyupp/coco-minitrain)
- [Cone Detection Dataset](https://github.com/ikatsamenis/Cone-Detection)
