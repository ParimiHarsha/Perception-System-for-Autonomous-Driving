# Autonomous-Driving-Perception-System

A ready-to-use system for autonomous driving, featuring YOLOv9 for object detection and SAMv2 for road segmentation. Fully integrated with ROS for seamless deployment in robotic systems. Ideal for developers and researchers focused on autonomous vehicle technologies.

## Features

- **Object Detection:** Utilizes YOLOv9 for real-time, accurate object detection.
- **Road Segmentation:** Employs SAMv2 for robust and precise road segmentation.
- **ROS Integration:** Fully integrated with ROS for seamless deployment and easy integration into robotic systems.

## Installation

### Prerequisites

Before installing the Autonomous Driving Perception System, ensure the following prerequisites are met:

- **ROS:** Make sure ROS-Noetic is installed on your system. For installation instructions, visit the [ROS Wiki](http://wiki.ros.org/ROS/
Installation).
- **Python:** Python 3.10 or later is required.
- **Dependencies:** All necessary Python libraries and dependencies are listed in the `environment.yaml` file.

## Installation Steps

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/ParimiHarsha/Autonomous-Driving-Perception-System.git
cd Autonomous-Driving-Perception-System
```

### Navigate to the Road Segmentation Directory

```bash
cd road_segmentation
```

### Install the SAMv2 Repository

#### Install the SAMv2

```bash
cd segment-anything-2 & pip install -e .
```

Install the necessary dependencies for SAMv2 by following the instructions provided in the SAMv2 repository.

### Return to the Top-Level Directory

```bash
cd ..
```

### Build the ROS Workspace: Use catkin to build the workspace and source it

```bash
catkin build
source devel/setup.bash
```

### Activate the Conda Environment: Activate the environment specified in the environment.yaml file

```bash
conda env create -f environment.yaml
conda activate perception_env
```

### Run the Required Road Segmentation Scripts

First, run the SAMv2Ros.py script:

```bash
python road_segmentation/src/SAMv2Ros.py
```

Then, execute the road_segmentation.3d.py script:

```bash
python road_segmentation/src/road_segmentation_3d.py
```

### Run the required Object Detection Scipts

First, run the yolo_detection_node.py script:

```bash
python yolov9ros/src/yolo_detection_node.py
```

Then, execute the transform_and_fuse.py script:

```bash
python yolov9ros/src/transform_and_fuse.py
```

If any additional packages are required during this process, install them as prompted.

## Usage

Once the installation is complete, the system can be deployed within a ROS environment. Use the provided scripts to start the perception modules for object detection and road segmentation. Ensure your ROS nodes are correctly configured and that the necessary topics are being published and subscribed to within your ROS ecosystem.

## Contributing

Contributions to the Autonomous-Driving-Perception-System are welcome. Please ensure that you follow the standard development practices and submit pull requests with detailed descriptions of the changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions, issues, or suggestions, please open an issue on the GitHub repository or contact the repository owner.
