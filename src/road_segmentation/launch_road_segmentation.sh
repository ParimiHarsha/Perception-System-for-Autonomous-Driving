#!/bin/bash

# Source ROS and Catkin workspace
source /opt/ros/noetic/setup.bash
source devel/setup.bash

# Activate the Conda environment for road_segmentation
conda activate roadseg

# Navigate to the road_segmentation package
cd ~/Document/ros_workspace/src/road_segmentation/src

echo pwd
# Run SAMv2Ros.py and road_segmentation_3d.py in separate terminals
gnome-terminal -- bash -c "python src/road_segmentation/src/SAMv2Ros.py; exec bash"
gnome-terminal -- bash -c "python src/road_segmentation/src/road_segmentation_3d.py; exec bash"