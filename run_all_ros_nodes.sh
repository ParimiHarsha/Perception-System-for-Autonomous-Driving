#!/bin/bash

# Run the road_segmentation launch script in a new terminal
gnome-terminal -- bash -c "./src/road_segmentation/launch_road_segmentation.sh; exec bash"

# Run the yolov9ros launch script in a new terminal
gnome-terminal -- bash -c "./src/yolov9ors/launch_yolov9ros.sh; exec bash"
