#!/usr/bin/env python3

from functools import wraps

import numpy as np
import ros_numpy
import rospy
import torch
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

from src.configs import LIDAR_2D_PROJ_TOPIC, LIDAR_TOPIC, PROJ, T1

# Define limits
lim_x, lim_y, lim_z = [20, 50], [-10, 10], [-3.5, 1]


def inverse_rigid_transformation(arr: np.ndarray) -> np.ndarray:
    rt = arr[:3, :3].T
    tt = -np.dot(rt, arr[:3, 3])
    return np.vstack((np.column_stack((rt, tt)), [0, 0, 0, 1]))


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


class LidarTo2DProjection:
    def __init__(self):
        rospy.init_node("lidar_to_2d_projection")

        # Publisher
        self.projection_pub = rospy.Publisher(
            LIDAR_2D_PROJ_TOPIC, Float32MultiArray, queue_size=1
        )

        # Subscriber
        self.sub_lidar = rospy.Subscriber(LIDAR_TOPIC, PointCloud2, self.lidar_callback)

        rospy.loginfo("Node initialized and ready to publish 2D projections.")

    def lidar_callback(self, msgLidar):
        pc_arr, u, v = self.process_pointcloud(msgLidar)
        self.publish_projection(u, v)

    @timer
    def process_pointcloud(self, msgLidar):
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T
        pc_arr = self.crop_pointcloud(points)

        # Apply transformation and projection
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr.T))
        uv1 = torch.matmul(torch.tensor(PROJ), m1)
        u, v = (uv1[:2, :] / uv1[2, :]).numpy()
        return pc_arr, u, v

    def publish_projection(self, u, v):
        projection_msg = Float32MultiArray()
        projection_msg.data = np.column_stack((u, v)).flatten().tolist()
        self.projection_pub.publish(projection_msg)
        rospy.loginfo("Published 2D projections with %d points.", len(u))

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
    LidarTo2DProjection()
    rospy.spin()
