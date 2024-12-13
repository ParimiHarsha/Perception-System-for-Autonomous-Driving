#!/usr/bin/env python3

import sys
import os

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import rospy
import cv2
import math
import torch
from sensor_msgs.msg import Image

# from cv_bridge import CvBridge
from UltraFast.ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from lane_detection.msg import LanePoint, LanePoints
import numpy as np
import ros_numpy


class LaneDetectionNode:
    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__),
            "/home/dev/Documents/Autonomous-Driving-Perception-System/src/ultrafastv2_ros/src/UltraFast/models/tusimple_18.pth",
        )
        self.model_type = ModelType.TUSIMPLE

        # Determine device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.use_gpu = True  # To ensure GPU usage
        else:
            self.use_gpu = False

        self.lane_detector = UltrafastLaneDetector(
            self.model_path, self.model_type, self.use_gpu
        )

        # Subscribers
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.image_callback
        )

        # Publishers
        self.image_pub = rospy.Publisher("/lane_detection/output", Image, queue_size=1)
        print("Publishing /lane_detection/output\n")
        self.lane_points_pub = rospy.Publisher(
            "/lane_detection/detected_lane_points", LanePoints, queue_size=1
        )
        print("Publishing /lane_detection/detected_lane_points\n")
        self.left_lane_boundary_pub = rospy.Publisher(
            "/lane_detection/current_lane_left_boundary", LanePoints, queue_size=1
        )
        print("Publishing /lane_detection/current_lane_left_boundary\n")
        self.right_lane_boundary_pub = rospy.Publisher(
            "/lane_detection/current_lane_right_boundary", LanePoints, queue_size=1
        )
        print("Publishing /lane_detection/current_lane_right_boundary\n")

        self.last_time = rospy.Time.now()
        rospy.loginfo(f"Using GPU: {self.use_gpu}")

    def image_callback(self, data):
        img = ros_numpy.numpify(data)
        cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect lanes
        point_coords = [
            (680, 630),
            (820, 630),
            (960, 630),
        ]  # multiple reference points
        output_img, lanes_points, lanes_detected = self.lane_detector.detect_lanes(
            cv_image, point_coords=point_coords
        )

        # Publish lane points
        lane_points_msg = LanePoints()
        left_lane_boundary_points = []
        right_lane_boundary_points = []

        min_left_avg_dist = float("inf")
        min_right_avg_dist = float("inf")
        closest_left_lane_id = None
        closest_right_lane_id = None

        # Find closest points on left and right lane boundaries
        for lane_id, lane in enumerate(lanes_points):
            left_distances = [0] * len(point_coords)
            right_distances = [0] * len(point_coords)
            left_counts = [0] * len(point_coords)
            right_counts = [0] * len(point_coords)

            for point in lane:
                if point[0] != 0 or point[1] != 0:
                    for i, ref_point in enumerate(point_coords):
                        dist = math.sqrt(
                            (point[0] - ref_point[0]) ** 2
                            + (point[1] - ref_point[1]) ** 2
                        )
                        if point[0] < ref_point[0]:
                            left_distances[i] += dist
                            left_counts[i] += 1
                        elif point[0] > ref_point[0]:
                            right_distances[i] += dist
                            right_counts[i] += 1

            avg_left_dist = sum(
                (
                    left_distances[i] / left_counts[i]
                    if left_counts[i] > 0
                    else float("inf")
                )
                for i in range(len(point_coords))
            ) / len(point_coords)
            if avg_left_dist < min_left_avg_dist:
                min_left_avg_dist = avg_left_dist
                closest_left_lane_id = lane_id

            avg_right_dist = sum(
                (
                    right_distances[i] / right_counts[i]
                    if right_counts[i] > 0
                    else float("inf")
                )
                for i in range(len(point_coords))
            ) / len(point_coords)
            if avg_right_dist < min_right_avg_dist:
                min_right_avg_dist = avg_right_dist
                closest_right_lane_id = lane_id

        # Collect the closest left and right boundary points
        if closest_left_lane_id is not None:
            for point in lanes_points[closest_left_lane_id]:
                if point[0] != 0 or point[1] != 0:
                    left_lane_boundary_points.append(
                        LanePoint(
                            x=int(point[0]),
                            y=int(point[1]),
                            lane_id=closest_left_lane_id,
                        )
                    )

        if closest_right_lane_id is not None:
            for point in lanes_points[closest_right_lane_id]:
                if point[0] != 0 or point[1] != 0:
                    right_lane_boundary_points.append(
                        LanePoint(
                            x=int(point[0]),
                            y=int(point[1]),
                            lane_id=closest_right_lane_id,
                        )
                    )

        # Publish lane boundary points
        left_lane_msg = LanePoints(
            header=data.header, points=left_lane_boundary_points
        )
        right_lane_msg = LanePoints(
            header=data.header, points=right_lane_boundary_points
        )
        self.left_lane_boundary_pub.publish(left_lane_msg)
        self.right_lane_boundary_pub.publish(right_lane_msg)

        # Convert OpenCV image back to ROS Image message manually
        output_img_msg = Image(
            height=output_img.shape[0],
            width=output_img.shape[1],
            encoding="rgb8",
            is_bigendian=0,
            step=output_img.shape[1] * 3,
            data=np.array(output_img).tobytes(),
        )

        # Publish the output image
        self.image_pub.publish(output_img_msg)


if __name__ == "__main__":
    rospy.init_node("lane_detection_node", anonymous=True)
    lane_detection_node = LaneDetectionNode()
    rospy.spin()
