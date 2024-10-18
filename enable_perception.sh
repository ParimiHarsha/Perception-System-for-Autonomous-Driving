#!/bin/bash

# Source the ROS workspace
echo "Sourcing the ROS workspace"
source devel/setup.bash

# Activate YOLOv9 conda environment and launch YOLOv9 nodes in a new terminal tab
echo "Launching YOLOv9 object detection in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/anaconda3/etc/profile.d/conda.sh && 
                                 echo "Activating YOLOv9 conda environment..." &&
                                 conda activate yolov9 && 
                                 echo "Launching YOLOv9 object detection..." &&
                                 roslaunch yolov9ros run_object_detection.launch; exec bash'

# Activate SAM2 conda environment and launch SAM2 nodes in a new terminal tab
echo "Launching SAM2 segmentation in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/anaconda3/etc/profile.d/conda.sh && 
                                 echo "Launching SAM2 segmentation..." &&
                                 conda activate sam && 
                                 echo "Launching SAM2 segmentation..." &&
                                 roslaunch road_segmentation run_segmentation.launch; exec bash'

# Wait for all processes to finish
wait

# Close the current terminal tab 
echo "Closing the current tab..."
kill -9 $PPID