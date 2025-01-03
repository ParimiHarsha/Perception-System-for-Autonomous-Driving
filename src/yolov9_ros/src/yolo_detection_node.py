#!/usr/bin/env python3
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

import os
from typing import List

import cv2
import numpy as np
import ros_numpy
import rospy
import torch
import yaml
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ultralytics import YOLO

from src.configs import CAMERA_TOPIC, YOLO_BBOX_TOPIC, YOLO_IMAGE_TOPIC
from yolov9_ros.msg import Bbox, BboxList

# Initialize CUDA device early
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.3, device=torch.device("cuda:0"))
if device == "cpu":
    print("Using CPU and not GPU")
if device != torch.device("cpu"):
    torch.cuda.init()  # Ensure CUDA is initialized early

# Get the paths of the current script and other required files
base_path = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_path, "..", "best.pt")
class_averages_path = os.path.join(base_path, "class_averages.yaml")
suppressed_classes_path = os.path.join(base_path, "suppressed_classes.yaml")

# Configuration parameters
img_size = 640
conf_thres = 0.4
view_img = True
write_file = False  # Set this flag to control whether to write the video file

# Average Class Dimensions
with open(class_averages_path, "r", encoding="utf-8") as file:
    average_dimensions = yaml.safe_load(file)

with open(suppressed_classes_path, "r", encoding="utf-8") as file:
    suppressed_classes = yaml.safe_load(file)["suppressed_classes"]


class Detect:
    def __init__(self) -> None:
        self.model = YOLO(weights_path).to(device)
        self.model.conf = 0.5
        self.names: List[str] = self.model.names
        self.image_sub = rospy.Subscriber(
            CAMERA_TOPIC,
            Image,
            self.camera_callback,
            queue_size=1,
        )
        self.image_pub = rospy.Publisher(YOLO_IMAGE_TOPIC, Image, queue_size=1)
        self.bboxInfo_pub = rospy.Publisher(
            YOLO_BBOX_TOPIC, BboxList, queue_size=1
        )

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

    # Add the classify_traffic_light function
    def classify_traffic_light(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define color ranges for yellow, green, red
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([90, 255, 255])

        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        # Masking
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Count the number of pixels for each color
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)

        # Determine the color with the maximum number of pixels
        if yellow_pixels > green_pixels and yellow_pixels > red_pixels:
            return "Yellow"
        elif green_pixels > yellow_pixels and green_pixels > red_pixels:
            return "Green"
        elif red_pixels > yellow_pixels and red_pixels > green_pixels:
            return "Red"
        else:
            return "Unknown"

    def camera_callback(self, data: Image) -> None:
        img: np.ndarray = ros_numpy.numpify(data)  # Image size is (772, 1032, 3)
        img_resized: np.ndarray = cv2.resize(
            img, (img_size, img_size)
        )  # Image resized to (640, 640)
        img_without_green_box = img_resized.copy()
        img_rgb: np.ndarray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize and prepare the tensor
        img_tensor: torch.Tensor = (
            torch.from_numpy(img_rgb).to(device, non_blocking=True).float()
        )
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
                label: str = f"{self.names[class_id]}: {conf:.2f}"

                # Suppress irrelevant classes
                if self.names[class_id] in suppressed_classes:
                    continue

                # If the detected object is a traffic light and confidence is greater than 50%
                # Then make the bounding box
                if self.names[class_id] == "traffic light" and conf > 0.5:
                    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    roi = img_without_green_box[y1:y2, x1:x2]
                    color = self.classify_traffic_light(roi)
                    cv2.putText(
                        img_resized,
                        color,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),  # Black
                        2,
                    )

                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_resized,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 225),
                    2,
                )

            self.publish_bboxes(
                detections.boxes.data[filtered_indices],
                data.header.stamp,
            )
            if view_img:
                self.publish_image(img_resized, data.header.stamp)

            # Write frame to video file if write_file is True
            if write_file and self.video_writer.isOpened():
                self.video_writer.write(img_resized)
                rospy.loginfo("Frame written to video")

    def publish_bboxes(self, detections: torch.Tensor, stamp: rospy.Time) -> None:
        # Ensure the detections data is in the expected format
        msg = BboxList()  # Create an instance of the BboxCentersClass message
        msg.header = Header()
        msg.header.stamp = stamp  # Add timestamp from the original ROS message
        msg.Bboxes = []  # Initialize the list for bounding boxes

        for bbox in detections:
            # Parse detection data
            x1, y1, x2, y2, conf, cls = bbox  # Convert tensor to list

            if conf > conf_thres:  # Filter detections based on confidence
                # Scale bounding box coordinates back to the original image dimensions
                x_min = x1 * (1032 / 640)
                y_min = y1 * (772 / 640)
                x_max = x2 * (1032 / 640)
                y_max = y2 * (772 / 640)

                # Create a BboxCenter message for the bounding box
                bbox_msg = Bbox()
                bbox_msg.x_min = x_min
                bbox_msg.y_min = y_min
                bbox_msg.x_max = x_max
                bbox_msg.y_max = y_max
                bbox_msg.confidence = float(conf)  # Confidence as float
                bbox_msg.class_id = int(cls)  # Class ID as integer

                msg.Bboxes.append(bbox_msg)  # Append the bbox message to the list

        # Publish the message
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
    rospy.init_node("YOLOv9")
    Detect()
