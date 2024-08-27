# type: ignore
"""
YOLO Object Detection Node for ROS

This module implements a ROS node for real-time object detection using the YOLO model.
The detected objects are published as bounding box coordinates with class labels and
confidence scores. Additionally, the module supports optional video recording of the
detection results.

Classes:
    Detect: A class for handling YOLO object detection in ROS.

Configuration parameters:
    weights (str): Path to the YOLO model weights file.
    img_size (int): Size to which input images are resized for detection.
    conf_thres (float): Confidence threshold for filtering detections.
    device (torch.device): Device to run the model on (CUDA if available, otherwise CPU).
    view_img (bool): Flag to enable publishing detected images.
    write_file (bool): Flag to enable video recording of detections.

Usage:
    Run the module as a python script using. Ensure the ROS environment is set up correctly
    and the required topics are available.

Example:
    python yolo_detection_node.py

"""

from typing import List

import cv2
import numpy as np
import ros_numpy
import rospy
import torch
import yaml
from geometry_msgs.msg import Point32
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from ultralytics import YOLO

from yolov9ros.msg import BboxCentersClass

# Configuration parameters
weights: str = (
    "/home/avalocal/Documents/yolov9_ros/src/yolov9ros/src/runs/detect/train3/weights/best.pt"
)
img_size: int = 640
conf_thres: float = 0.5

# Initialize CUDA device early
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Is CUDA available", torch.cuda.is_available())
if device != torch.device("cpu"):
    torch.cuda.init()  # Ensure CUDA is initialized early

view_img: bool = True
write_file: bool = False  # Set this flag to control whether to write the video file

# Average Class Dimensions
with open("class_averages.yaml", "r", encoding="utf-8") as file:
    average_dimensions = yaml.safe_load(file)


class Detect:
    def __init__(self) -> None:
        self.model = YOLO(weights).to(device)
        self.model.conf = 0.5
        if device != torch.device("cpu"):
            self.model.half()
        self.names: List[str] = self.model.names
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color",
            Image,
            self.camera_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.image_pub = rospy.Publisher("~published_image", Image, queue_size=1)
        self.bboxInfo_pub = rospy.Publisher("~bboxInfo", BboxCentersClass, queue_size=1)

        # Initialize VideoWriter if write_file is True
        if write_file:
            self.video_writer = cv2.VideoWriter(
                "video_output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # Assuming 30 FPS, change if necessary
                (img_size, img_size),
            )
            if not self.video_writer.isOpened():
                rospy.logerr("Failed to open video writer")

        rospy.on_shutdown(self.cleanup)  # Register cleanup function
        rospy.spin()

    def camera_callback(self, data: Image) -> None:
        img: np.ndarray = ros_numpy.numpify(data)  # Image size is (772, 1032, 3)
        img_resized: np.ndarray = cv2.resize(
            img, (img_size, img_size)
        )  # Image resized to (640, 640)
        img_rgb: np.ndarray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize and prepare the tensor
        img_tensor: torch.Tensor = torch.from_numpy(img_rgb).to(device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            detections = self.model(img_tensor)[0]
            bboxes: np.ndarray = detections.boxes.xyxy.cpu().numpy().astype(int)
            class_ids: np.ndarray = detections.boxes.cls.cpu().numpy().astype(int)
            confidences: np.ndarray = detections.boxes.conf.cpu().numpy()

            # Filter out detections below the confidence threshold
            filtered_indices = [
                i for i, conf in enumerate(confidences) if conf > conf_thres
            ]
            filtered_bboxes = bboxes[filtered_indices]
            filtered_class_ids = class_ids[filtered_indices]
            filtered_confidences = confidences[filtered_indices]

            for bbox, class_id, conf in zip(
                filtered_bboxes, filtered_class_ids, filtered_confidences
            ):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label: str = f"{self.names[class_id]}: {conf:.2f}"
                cv2.putText(
                    img_resized,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 225),
                    2,
                )

            self.publish_center_class(
                detections.boxes.data[filtered_indices],
                data.header.stamp,
            )

            if view_img:
                self.publish_image(img_resized, data.header.stamp)

            # Write frame to video file if write_file is True
            if write_file and self.video_writer.isOpened():
                self.video_writer.write(img_resized)
                rospy.loginfo("Frame written to video")

    def publish_center_class(self, detections: List[float], stamp: rospy.Time) -> None:
        msg = BboxCentersClass()
        msg.header.stamp = stamp
        msg.CenterClass = []
        for bbox in detections:
            x1, y1, x2, y2, conf, cls = bbox
            if conf > conf_thres:
                x_center: float = (x1 + x2) / 2 * (1032 / 640)
                y_center: float = (y1 + y2) / 2 * (772 / 640)
                point = Point32(x=x_center, y=y_center, z=cls)
                msg.CenterClass.append(point)
                rospy.loginfo(
                    "Appended bounding box center: (%f, %f, %s)",
                    x_center,
                    y_center,
                    average_dimensions[int(cls.item())]["name"],
                )
        self.bboxInfo_pub.publish(msg)

    def publish_image(self, img: np.ndarray, stamp: rospy.Time) -> None:
        img_pil: PILImage = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        msg: Image = Image()
        msg.header.stamp = stamp
        msg.height = img_pil.height
        msg.width = img_pil.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img_pil.width
        msg.data = np.array(img_pil).tobytes()
        self.image_pub.publish(msg)

    def cleanup(self) -> None:
        if write_file and self.video_writer.isOpened():
            self.video_writer.release()
            rospy.loginfo("Video writer released")


if __name__ == "__main__":
    rospy.init_node("yoloLiveNode")
    Detect()
