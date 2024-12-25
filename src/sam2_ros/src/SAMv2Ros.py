#!/usr/bin/env python3
import os

import cv2
import numpy as np

np.float = np.float64
from functools import wraps

import message_filters
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image

from sam2_ros.msg import DetectedRoadArea
from src.configs import (
    CAMERA_TOPIC,
    LEFT_CONTOUR_TOPIC,
    RIGHT_CONTOUR_TOPIC,
    SEGMENTATION_MASK_TOPIC,
    YOLO_BBOX_TOPIC,
)
from yolov9_ros.msg import BboxList


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = rospy.Time.now().to_sec()
        result = func(*args, **kwargs)
        end_time = rospy.Time.now().to_sec()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")
torch.cuda.set_per_process_memory_fraction(0.4, device=torch.device("cuda:0"))

# Load SAM2 model
base_path = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")
sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Define points for initial segmentation prompt
point_coords = np.array([[400, 700], [550, 700], [650, 700]])
input_labels = [1, 1, 1]
MIN_CONTOUR_AREA = 30000.0


def process_image(image, detected_objects, publish_image=False):
    h_original, w_original = image.shape[:2]
    center_x = int(w_original * 0.75)

    # Predict masks using SAM2 model
    with torch.cuda.amp.autocast():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=input_labels, multimask_output=False
        )

    road_mask = (masks[0] > 0).astype(np.uint8)
    # Create a unified mask for road and bounding boxes
    unified_mask = road_mask.copy()

    # Iterate over detected objects to adjust the mask
    for x_min, y_min, x_max, y_max in detected_objects:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Create a bounding box mask
        bbox_mask = np.zeros_like(road_mask, dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        # Ensure bounding boxes do not override road regions
        unified_mask = cv2.bitwise_and(unified_mask, cv2.bitwise_not(bbox_mask))

    # Fill gaps in the road mask
    kernel = np.ones((6, 6), np.uint8)
    road_mask_cleaned = cv2.morphologyEx(unified_mask, cv2.MORPH_CLOSE, kernel)

    # Subtract object mask from road mask
    # road_mask_cleaned = cv2.bitwise_or(road_mask, cv2.bitwise_not(object_mask_dilated))

    # Fill gaps in the road mask
    # road_mask_filled = cv2.morphologyEx(road_mask_dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        road_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # # Dilate object mask to cover edges
    # kernel = np.ones((6, 6), np.uint8)
    # dilated_mask = cv2.dilate(road_mask, kernel, iterations=1)
    # filled_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    # contours, _ = cv2.findContours(filled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        rospy.logwarn("No contours found.")
        return None, None, None

    road_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(road_contour) < MIN_CONTOUR_AREA:
        rospy.logwarn("The contour area is below the threshold.")
        return None, None, None

    # Filter road contour points
    MIN_Y_COORD = int(0.8 * h_original)
    road_contour = [point for point in road_contour if point[0][1] < MIN_Y_COORD]

    if len(road_contour) < 2:
        rospy.logwarn(
            "Filtered road contour is too small after removing bottom points."
        )
        return None, None, None

    contour_points = np.array(road_contour).reshape(-1, 2)
    horizon_point = contour_points[np.argmin(contour_points[:, 1])]

    (
        left_boundary_points,
        right_boundary_points,
    ) = classify_boundaries_using_horizontal_bins(
        contour_points[contour_points[:, 1] < center_x], h_original
    )

    for point in contour_points[contour_points[:, 1] >= center_x]:
        (
            left_boundary_points
            if point[0] < horizon_point[0]
            else right_boundary_points
        ).append(point)

    overlay = create_overlay(
        image,
        road_mask,
        left_boundary_points,
        right_boundary_points,
        publish_image,
    )

    return overlay, left_boundary_points, right_boundary_points


def classify_boundaries_using_horizontal_bins(
    contour_points, frame_height, num_bins=50
):
    if len(contour_points) == 0:
        return np.array([]), np.array([])

    y_min = contour_points[:, 1].min()
    bin_height = (frame_height - y_min) // num_bins
    left_boundary_points, right_boundary_points = [], []

    for i in range(num_bins):
        y_lower_limit = y_min + i * bin_height
        y_upper_limit = y_min + (i + 1) * bin_height

        bin_points = contour_points[
            (contour_points[:, 1] > y_lower_limit)
            & (contour_points[:, 1] <= y_upper_limit)
        ]

        if len(bin_points) > 0:
            mean_x = bin_points[:, 0].mean()
            for point in bin_points:
                (
                    left_boundary_points if point[0] < mean_x else right_boundary_points
                ).append(point)

    return np.array(left_boundary_points), np.array(right_boundary_points)


def create_overlay(
    image, binary_mask_np, left_boundary_points, right_boundary_points, publish_image
):
    if not publish_image:
        return None

    overlay = image.copy()
    mask = binary_mask_np.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
    overlay = cv2.addWeighted(
        overlay.astype(np.uint8),
        0.8,
        mask * np.array([0, 255, 0], dtype=np.uint8),
        0.2,
        0,
    )

    for point in point_coords:
        cv2.circle(overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1)
    for point in left_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2)
    for point in right_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2)

    return overlay


class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        self.image_pub = rospy.Publisher(SEGMENTATION_MASK_TOPIC, Image, queue_size=1)

        self.image_sub = message_filters.Subscriber(
            CAMERA_TOPIC,
            Image,
            queue_size=1,
        )
        self.yolo_sub = message_filters.Subscriber(
            YOLO_BBOX_TOPIC, BboxList, queue_size=1
        )
        # Time synchronizer for lidar, image, and radar data
        ts = message_filters.ApproximateTimeSynchronizer(
            [
                self.image_sub,
                self.yolo_sub,
            ],
            15,
            0.4,
        )
        ts.registerCallback(self.callback)

        self.left_boundary_pub = rospy.Publisher(
            LEFT_CONTOUR_TOPIC, DetectedRoadArea, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            RIGHT_CONTOUR_TOPIC, DetectedRoadArea, queue_size=1
        )

        self.ros_image = None
        self.publish_image = True
        self.detected_objects = []
        rospy.Timer(rospy.Duration(0.1), self.process_loop)

    def callback(self, ros_image, bboxes):
        self.ros_image = ros_image
        self.detected_objects = [
            (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max) for bbox in bboxes.Bboxes
        ]

    def objects_on_road(self, objects, road_mask):
        """
        Check if any detected objects overlap with the road mask.
        :param objects: List of bounding boxes [(x_min, y_min, x_max, y_max)].
        :param road_mask: Binary mask of the road (numpy array).
        :return: True if any objects overlap with the road mask.
        """
        for x_min, y_min, x_max, y_max in objects:
            x_min, x_max = max(0, x_min), min(road_mask.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(road_mask.shape[0], y_max)
            roi = road_mask[y_min:y_max, x_min:x_max]
            if np.any(roi > 0):  # Check if there's overlap
                return True
        return False

    @timer
    def process_loop(self, event):
        """Process the image if available, called periodically by a ROS Timer."""
        if self.ros_image:
            self.image_callback()

    @timer
    def image_callback(self):
        img = ros_numpy.numpify(self.ros_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay, left_boundary, right_boundary = process_image(
            img,
            self.detected_objects,
            self.publish_image,
        )

        if self.publish_image and overlay is not None:
            self.publish_image_topic(self.ros_image, overlay)

        # Publish boundary points
        if left_boundary is not None:
            self.publish_boundary(
                left_boundary, self.left_boundary_pub, self.ros_image.header.stamp
            )

        if right_boundary is not None:
            self.publish_boundary(
                right_boundary, self.right_boundary_pub, self.ros_image.header.stamp
            )

    def publish_image_topic(self, ros_image, overlay):
        msg = Image()
        msg.header.stamp = ros_image.header.stamp
        msg.height, msg.width = overlay.shape[:2]
        msg.encoding = "rgb8"
        msg.data = overlay.tobytes()
        self.image_pub.publish(msg)

    def publish_boundary(self, boundary, publisher, stamp):
        boundary_msg = DetectedRoadArea()
        boundary_msg.header.stamp = stamp
        boundary_msg.RoadArea.data = [float(point) for point in boundary.flatten()]
        publisher.publish(boundary_msg)


if __name__ == "__main__":
    rospy.init_node("SAM2", anonymous=False)
    RoadSegmentation()
    rospy.spin()
