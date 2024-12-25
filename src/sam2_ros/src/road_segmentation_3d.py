#!/usr/bin/env python3

from functools import wraps

import message_filters
import numpy as np

np.float = np.float64
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from scipy.spatial import KDTree
from sensor_msgs.msg import PointCloud2

from sam2_ros.msg import DetectedRoadArea
from src.configs import (LEFT_BOUNDARY, LEFT_CONTOUR_TOPIC, LIDAR_TOPIC, PROJ,
                         RIGHT_BOUNDARY, RIGHT_CONTOUR_TOPIC, T1)

# Define limits
lim_x, lim_y, lim_z, pixel_lim = [20, 50], [-10, 10], [-3.5, 1], 5


def inverse_rigid_transformation(arr: np.ndarray) -> np.ndarray:
    Rt = arr[:3, :3].T
    tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


T_vel_cam = inverse_rigid_transformation(T1)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = rospy.Time.now().to_sec()
        result = func(*args, **kwargs)
        end_time = rospy.Time.now().to_sec()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class RoadSegmentation3D:
    def __init__(self):
        rospy.init_node("segmentationTO3d")

        # Publishers
        self.left_boundary_pub = rospy.Publisher(
            LEFT_BOUNDARY, PointCloud2, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            RIGHT_BOUNDARY, PointCloud2, queue_size=1
        )

        # Define the point fields
        self.fields = [
            pc2.PointField(
                name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1
            ),
            pc2.PointField(
                name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1
            ),
        ]

        # Subscribers
        self.sub_lidar = message_filters.Subscriber(LIDAR_TOPIC, PointCloud2)
        self.left_boundary_sub = message_filters.Subscriber(
            LEFT_CONTOUR_TOPIC, DetectedRoadArea
        )
        self.right_boundary_sub = message_filters.Subscriber(
            RIGHT_CONTOUR_TOPIC, DetectedRoadArea
        )

        # Synchronize topics
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.left_boundary_sub, self.right_boundary_sub],
            queue_size=10,
            slop=0.3,
            allow_headerless=False,
        )
        ts.registerCallback(self.callback)

        # Variables to store incoming data
        self.msgLidar = None
        self.msgLeftBoundary = None
        self.msgRightBoundary = None

        # Timer to trigger processing loop
        rospy.Timer(rospy.Duration(0.1), self.process_loop)
        rospy.loginfo("Node initialized and timer set.")

    def callback(self, msgLidar, msgLeftBoundary, msgRightBoundary):
        self.msgLidar = msgLidar
        self.msgLeftBoundary = msgLeftBoundary
        self.msgRightBoundary = msgRightBoundary

    @timer
    def process_loop(self, event):
        if not self.msgLidar or not self.msgLeftBoundary or not self.msgRightBoundary:
            print("Some topic is missing")
            return

        pc_arr, u, v = self.process_pointcloud(self.msgLidar)
        left_boundary_3d = self.find_matching_points_kdtree(
            np.array(self.msgLeftBoundary.RoadArea.data).reshape(-1, 2), u, v, pc_arr
        )
        right_boundary_3d = self.find_matching_points_kdtree(
            np.array(self.msgRightBoundary.RoadArea.data).reshape(-1, 2), u, v, pc_arr
        )

        if left_boundary_3d.size > 0:
            self.create_cloud(left_boundary_3d, self.left_boundary_pub, self.msgLidar)
        if right_boundary_3d.size > 0:
            self.create_cloud(right_boundary_3d, self.right_boundary_pub, self.msgLidar)

    def process_pointcloud(self, msgLidar):
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T
        pc_arr = self.crop_pointcloud(points)

        # Apply transformation and projection
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr.T))
        uv1 = torch.matmul(torch.tensor(PROJ), m1)
        u, v = (uv1[:2, :] / uv1[2, :]).numpy()
        return pc_arr, u, v

    def create_cloud(self, points_3d, publisher, msgLidar):
        header = msgLidar.header
        pointcloud = pc2.create_cloud(header, self.fields, points_3d)
        publisher.publish(pointcloud)
        rospy.loginfo("Published point cloud with %d points.", len(points_3d))

    def find_matching_points_kdtree(self, boundary_points, u, v, pc_arr):
        tree = KDTree(np.column_stack((u, v)))
        idx = []
        for contour_point in boundary_points:
            matches = tree.query_ball_point(contour_point, pixel_lim)
            idx.extend(matches)
        return pc_arr[np.array(idx)] if idx else np.empty((0, 4))

    def crop_pointcloud(self, pointcloud):
        mask = (
            (pointcloud[:, 0] >= lim_x[0])
            & (pointcloud[:, 0] <= lim_x[1])
            & (pointcloud[:, 1] >= lim_y[0])
            & (pointcloud[:, 1] <= lim_y[1])
            & (pointcloud[:, 2] >= lim_z[0])
            & (pointcloud[:, 2] <= lim_z[1])
        )
        return pointcloud[mask]


if __name__ == "__main__":
    RoadSegmentation3D()
    rospy.spin()
