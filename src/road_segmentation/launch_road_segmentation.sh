#!/bin/bash

# Source ROS and Catkin workspace
# source /opt/ros/noetic/setup.bash
source devel/setup.bash

# Navigate to the road_segmentation package
# cd ~/Document/ros_workspace/src/road_segmentation/src

# Run SAMv2Ros.py and road_segmentation_3d.py in separate tabs of the same terminal, and activate the conda environment within each
gnome-terminal --tab -- bash -c "conda activate sam; python src/road_segmentation/src/SAMv2Ros.py; exec bash" \
               --tab -- bash -c "conda activate sam; python src/road_segmentation/src/road_segmentation_3d.py; exec bash"
