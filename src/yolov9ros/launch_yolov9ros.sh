#!/bin/bash

# Source ROS and Catkin workspace
source /opt/ros/noetic/setup.bash
source /devel/setup.bash

# Activate the Conda environment for yolov9ros
conda activate yolov9ros_env

# Navigate to the yolov9ros package
cd ~/Document/ros_workspace/src/yolov9ros/src

# Run yolo_detection_node.py and transform_and_fuse.py in separate terminals
gnome-terminal -- bash -c "python src/yolov9ros/src/yolo_detection_node.py; exec bash"
gnome-terminal -- bash -c "python src/yolov9ros/src/transform_and_fuse.py; exec bash"
