# Perception System for Autonomous Driving

![Demo Video](https://github.com/user-attachments/assets/e6f0876c-9d8a-4220-9c69-085cf4ed74b0)

A ready-to-use system for autonomous driving, featuring YOLOv9 for object detection, SAMv2 for road segmentation, Ultrafast lane detection, and Sphereformer LiDAR segmentation. Fully integrated with ROS for seamless deployment in robotic systems. Ideal for developers and researchers focused on autonomous vehicle technologies.

---

## Features
- **Object Detection**: Utilizes YOLOv9 for real-time, accurate object detection.
- **Road Segmentation**: Employs SAMv2 for robust and precise road segmentation.
- **Lane Detection**: Implements Ultrafast lane detection for high-speed lane recognition.
- **LiDAR Segmentation**: Uses Sphereformer for advanced 3D LiDAR-based segmentation.
- **ROS Integration**: Fully integrated with ROS for seamless deployment and easy integration into robotic systems.

---

## Installation

### Prerequisites

Before installing the Autonomous Driving Perception System, ensure the following prerequisites are met:
- **ROS**: Make sure ROS-Noetic is installed on your system. For installation instructions, visit the [ROS Wiki](http://wiki.ros.org/ROS/Installation).
- **Python**: Python 3.10 or later is required.
- **Dependencies**: All necessary Python libraries and dependencies are listed in the corresponding `environment.yaml` files.

### Clone the Repository with Submodules
```bash
git clone --recursive https://github.com/ParimiHarsha/Autonomous-Driving-Perception-System.git
cd Autonomous-Driving-Perception-System
```

### Build the ROS Workspace
```bash
catkin build
source devel/setup.bash
```

### Create and Activate Environments
Each perception module requires its own environment:

#### SAMv2 Installation
```bash
conda create -n sam python=3.10
conda activate sam
cd src/road_segmentation/src/segment-anything-2
pip install -e .
cd checkpoints && ./download_ckpts.sh
```

#### YOLOv9 Installation
```bash
conda env create -f src/yolov9ros/src/environment.yaml
conda activate yolo_env
```
##### Download the Trained YOLOv9 model

```bash
cd src/yolov9ros/
```
Download the [trained model](https://drive.google.com/file/d/1UAX-7jSXQJcyRdumn8iXmwjfJxxyC9Tw/view?usp=sharing) here. And save it in yolov9_ros folder

#### Ultrafast Lane Detection and Sphereformer
Follow the respective `README.md` files in their directories for installation details.


## Running the Perception System

The entire perception stack can be launched with a single command:

### Enable Perception
```bash
./enable_perception.bash
```

### Disable Perception
```bash
./disable_perception.bash
```

These scripts handle launching and shutting down all perception nodes, including object detection, road segmentation, lane detection, and LiDAR segmentation.

---

## Usage
Once installed, the system can be deployed within a ROS environment. The `enable_perception.bash` script will launch all perception nodes, while individual components can be run manually using their respective launch files and scripts. Ensure the appropriate topics are correctly published and subscribed to within your ROS setup.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact
For questions, issues, or suggestions, please open an issue on the GitHub repository or contact the repository owner.