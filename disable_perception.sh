#!/bin/bash

# Function to kill processes by name or PID
kill_processes() {
    process_name=$1
    echo "  Killing processes related to $process_name..."
    # Find PIDs for processes matching the name or specific pattern and kill them
    pids=$(pgrep -f "$process_name")
    if [ -z "$pids" ]; then
        echo "  No processes found for $process_name."
    else
        kill -9 $pids
    fi
}

# Print statement for starting the disable process
echo "Disabling perception system..."

# Kill YOLOv9 related processes
echo "Terminating YOLOv9 processes..."
kill_processes "yolo_detection_node.py"
kill_processes "transform_and_fuse.py"

# Kill SAM2 related processes
echo "Terminating SAM2 processes..."
kill_processes "SAMv2Ros.py"
kill_processes "road_segmentation_3d.py"

# Kill Sphereformer related processes
echo "Terminating Sphereformer Lidar Segmentation processes..."
kill_processes "lidar_segmentation.py"

# Kill UltraFast related processes
echo "Terminating UltraFast Lane Detection processes..."
kill_processes "ulfast.py"
kill_processes "lineto3d.py"

# Kill roslaunch processes if any are left running
echo "Terminating roslaunch processes..."
kill_processes "roslaunch"

# Print statement for ending the disable process
echo "Perception system has been successfully disabled."

