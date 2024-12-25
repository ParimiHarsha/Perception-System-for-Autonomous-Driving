#!/bin/bash

# Export PYTHONPATH for custom directories
echo "Setting PYTHONPATH..."
export PYTHONPATH=/home/dev/Documents/Autonomous-Driving-Perception-System:$PYTHONPATH

# Source the ROS workspace
echo "Sourcing the ROS workspace"
source devel/setup.bash

# Activate YOLOv9 conda environment and launch YOLOv9 nodes in a new terminal tab
echo "Launching YOLOv9 object detection in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/miniconda3/etc/profile.d/conda.sh && 
                                 echo "Activating YOLOv9 conda environment..." &&
                                 conda activate sam && 
                                 echo "Launching YOLOv9 object detection..." &&
                                 taskset -c 2,3 roslaunch yolov9_ros yolov9_ros.launch; exec bash'

# Activate SAM2 conda environment and launch SAM2 nodes in a new terminal tab
echo "Launching SAM2 segmentation in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/miniconda3/etc/profile.d/conda.sh && 
                                 echo "Activating SAM2 conda environment..." &&
                                 conda activate sam && 
                                 echo "Launching SAM2 segmentation..." &&
                                 taskset -c 4,5 roslaunch sam2_ros sam2_ros.launch; exec bash'

# Activate Sphereformer conda environment and launch it in a new terminal tab
echo "Launching Sphereformer Lidar Segmentation in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/miniconda3/etc/profile.d/conda.sh && 
                                 echo "Launching Sphereformer conda environment..." &&
                                 conda activate test_lidar1 && 
                                 echo "Launching Sphereformer Lidar segmentation..." &&
                                 taskset -c 6,7 roslaunch sphereformer_ros sphereformer_ros.launch; exec bash'

# Activate UltraFast conda environment and launch it in a new terminal tab
echo "Launching UltraFast Lane Detection in a new terminal tab..."
gnome-terminal --tab -- bash -c 'source /home/dev/miniconda3/etc/profile.d/conda.sh && 
                                 echo "Launching UltraFast conda environment..." &&
                                 conda activate lane-det && 
                                 echo "Launching UltraFast Lane Detection..." &&
                                 taskset -c 8,9 roslaunch ultrafastv2_ros ultrafastv2_ros.launch; exec bash'



# Wait for all processes to finish
wait

echo "Perception system has been successfully enabled."