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

## Installation

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/ParimiHarsha/Autonomous-Driving-Perception-System.git
cd Autonomous-Driving-Perception-System
```

### Build the ROS Workspace: Use catkin to build the workspace and source it

```bash
catkin build
source devel/setup.bash
```

### Navigate to the Road Segmentation Directory

```bash
cd src/road_segmentation/src
```

#### Clone and install the SAMv2

```bash
conda create -n "sam" python=3.10
conda activate sam
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

#### Download Checkpoints

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

Refer to the [SAMv2 github](https://github.com/facebookresearch/segment-anything-2) README file for more details/issues.

### Return to the Top-Level Directory

```bash
cd ../../..
```
### Download the Trained YOLOv9 model

```bash
cd src/yolov9ros/
```
Download the [trained model](https://drive.google.com/file/d/1UAX-7jSXQJcyRdumn8iXmwjfJxxyC9Tw/view?usp=sharing) here. And store it in yolov9ros folder

### Run the required Road Segmentation Scripts

```bash
python src/road_segmentation/src/SAMv2Ros.py
```

```bash
python src/road_segmentation/src/road_segmentation_3d.py
```

### Run the required Object Detection Scipts

```bash
conda env create -f src/yolov9ros/src/environment.yaml
conda activate perception_env
```

```bash
python src/yolov9ros/src/yolo_detection_node.py
```

```bash
python src/yolov9ros/src/transform_and_fuse.py
```

If any additional packages are required during this process, install them as prompted.

## Usage

Once the installation is complete, the system can be deployed within a ROS environment. Use the provided scripts to start the perception modules for object detection and road segmentation. Ensure your ROS nodes are correctly configured and that the necessary topics are being published and subscribed to within your ROS ecosystem.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For any questions, issues, or suggestions, please open an issue on the GitHub repository or contact the repository owner.
