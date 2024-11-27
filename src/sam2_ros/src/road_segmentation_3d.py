#!/usr/bin/env python3

import message_filters
import numpy as np

np.float = np.float64
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import torch
from sam2_ros.msg import DetectedRoadArea
from scipy.spatial import KDTree
from functools import wraps

# Define calibration and transformation matrices
proj = np.array(
    [
        [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
        [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
        [0, 0.0, 1.0, 0],
    ]
)

# proj = np.array([
#     [3508.080811 / 2, 0.000000, 1061.300000 / 2, 0],
#     [0.000000, 3543.697510 / 2, 736.416491 / 2, 0],
#     [0.000000, 0.000000, 1.000000, 0],]
# )
T1 = np.array(
    [
        [
            -4.8076040039157775e-03,
            1.1565175070195832e-02,
            9.9992156375854679e-01,
            1.3626313209533691e00,
        ],
        [
            -9.9997444266988167e-01,
            -5.3469003551928074e-03,
            -4.7460155553246119e-03,
            2.0700573921203613e-02,
        ],
        [
            5.2915924636425249e-03,
            -9.9991882539643562e-01,
            1.1590585274754983e-02,
            -9.1730421781539917e-01,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

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
            "/road_segment_3d/left_boundary", PointCloud2, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            "/road_segment_3d/right_boundary", PointCloud2, queue_size=1
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
        self.pcdSub = message_filters.Subscriber(
            "/lidar_tc/velodyne_points", PointCloud2
        )
        self.left_boundary_sub = message_filters.Subscriber(
            "/left_boundary", DetectedRoadArea
        )
        self.right_boundary_sub = message_filters.Subscriber(
            "/right_boundary", DetectedRoadArea
        )

        # Synchronize topics
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
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
            print("something is missing")
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
        uv1 = torch.matmul(torch.tensor(proj), m1)
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
