# Autonomous-Driving-Perception-System
A ready-to-use system for autonomous driving, featuring YOLOv9 for object detection and SAMv2 for road segmentation. Fully integrated with ROS for seamless deployment in robotic systems. Ideal for developers and researchers focused on autonomous vehicle technologies.

## Features

Object Detection: YOLOv9 for real-time and accurate object detection.
Road Segmentation: SAMv2 for robust and precise road detection.
ROS Integration: Ready to use with ROS for easy integration into robotic systems.

## Installation

### Prerequisites

ROS: Ensure ROS is installed on your system. For installation instructions, visit ROS Wiki.
Python: Requires Python 3.10 or later.
Dependencies: Required Python libraries are listed in the environment.yaml file.

### Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/ParimiHarsha/Autonomous-Driving-Perception-System.git
cd Autonomous-Driving-Perception-System
```



Steps to install

1. Clone the repo

2. cd to road_segmentation

3. install the samv2 repo , clone it and install the dependencies

4. cd back to top directory

5. catkin build

6. activate the conda env

7. run the SAMv2Ros.py script and then run road_segmentation.3d.py script (Install any other packages that it might need)

