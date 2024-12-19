#!/usr/bin/env python3

import math
import os

import cv2
import numpy as np
import ros_numpy
import rospy
import torch
from sensor_msgs.msg import Image
from UltraFast import ModelType, UltrafastLaneDetector

from ultrafastv2_ros.msg import LanePoint, LanePoints


class LaneDetectionNode:
    def __init__(self):
        # Model configuration
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tusimple_18.pth"
        )
        self.model_type = ModelType.TUSIMPLE
        self.use_gpu = torch.cuda.is_available()

        self.lane_detector = UltrafastLaneDetector(
            self.model_path, self.model_type, self.use_gpu
        )

        # ROS setup
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.image_callback
        )
        self.image_pub = rospy.Publisher("/lane_detection/output", Image, queue_size=1)
        self.left_lane_boundary_pub = rospy.Publisher(
            "/lane_detection/current_lane_left_boundary", LanePoints, queue_size=1
        )
        self.right_lane_boundary_pub = rospy.Publisher(
            "/lane_detection/current_lane_right_boundary", LanePoints, queue_size=1
        )

        rospy.loginfo(f"LaneDetectionNode initialized. Using GPU: {self.use_gpu}")

    def image_callback(self, data):
        # Convert ROS image to OpenCV format
        img = ros_numpy.numpify(data)
        cv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect lanes
        reference_points = [(680, 630), (820, 630), (960, 630)]
        output_img, lanes_points, _ = self.lane_detector.detect_lanes(
            cv_image, point_coords=reference_points
        )

        # Process lane points
        left_lane_points, right_lane_points = self.get_closest_lane_boundaries(
            lanes_points, reference_points
        )

        # Publish lane boundaries
        self.publish_lane_points(data.header, left_lane_points, right_lane_points)

        # Publish the output image
        self.publish_output_image(output_img)

    def get_closest_lane_boundaries(self, lanes_points, reference_points):
        left_lane_points = []
        right_lane_points = []

        min_left_avg_dist, min_right_avg_dist = float("inf"), float("inf")
        closest_left_lane_id, closest_right_lane_id = None, None

        for lane_id, lane in enumerate(lanes_points):
            left_distances, right_distances = [0] * len(reference_points), [0] * len(
                reference_points
            )
            left_counts, right_counts = [0] * len(reference_points), [0] * len(
                reference_points
            )

            for point in lane:
                if point[0] == 0 and point[1] == 0:
                    continue
                for i, ref_point in enumerate(reference_points):
                    dist = math.sqrt(
                        (point[0] - ref_point[0]) ** 2 + (point[1] - ref_point[1]) ** 2
                    )
                    if point[0] < ref_point[0]:
                        left_distances[i] += dist
                        left_counts[i] += 1
                    elif point[0] > ref_point[0]:
                        right_distances[i] += dist
                        right_counts[i] += 1

            avg_left_dist = sum(
                left_distances[i] / left_counts[i]
                if left_counts[i] > 0
                else float("inf")
                for i in range(len(reference_points))
            ) / len(reference_points)
            if avg_left_dist < min_left_avg_dist:
                min_left_avg_dist = avg_left_dist
                closest_left_lane_id = lane_id

            avg_right_dist = sum(
                right_distances[i] / right_counts[i]
                if right_counts[i] > 0
                else float("inf")
                for i in range(len(reference_points))
            ) / len(reference_points)
            if avg_right_dist < min_right_avg_dist:
                min_right_avg_dist = avg_right_dist
                closest_right_lane_id = lane_id

        if closest_left_lane_id is not None:
            left_lane_points = [
                LanePoint(x=int(p[0]), y=int(p[1]), lane_id=closest_left_lane_id)
                for p in lanes_points[closest_left_lane_id]
                if p[0] != 0 or p[1] != 0
            ]

        if closest_right_lane_id is not None:
            right_lane_points = [
                LanePoint(x=int(p[0]), y=int(p[1]), lane_id=closest_right_lane_id)
                for p in lanes_points[closest_right_lane_id]
                if p[0] != 0 or p[1] != 0
            ]

        return left_lane_points, right_lane_points

    def publish_lane_points(self, header, left_lane_points, right_lane_points):
        self.left_lane_boundary_pub.publish(
            LanePoints(header=header, points=left_lane_points)
        )
        self.right_lane_boundary_pub.publish(
            LanePoints(header=header, points=right_lane_points)
        )
        rospy.loginfo("Published left and right lane boundaries.")

    def publish_output_image(self, output_img):
        output_img_msg = Image(
            height=output_img.shape[0],
            width=output_img.shape[1],
            encoding="rgb8",
            is_bigendian=0,
            step=output_img.shape[1] * 3,
            data=np.array(output_img).tobytes(),
        )
        self.image_pub.publish(output_img_msg)
        rospy.loginfo("Published output image.")


if __name__ == "__main__":
    rospy.init_node("lane_detection_node", anonymous=True)
    LaneDetectionNode()
    rospy.spin()
