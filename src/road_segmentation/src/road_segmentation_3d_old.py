#!/usr/bin/env python3

import message_filters
import numpy as np

np.float = np.float64
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import time
import torch
from road_segmentation.msg import DetectedRoadArea
from scipy.spatial import KDTree

rect = np.array(
    [
        [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
        [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
        [0, 0.0, 1.0, 0],
    ]
)

# Camera to lidar extrinsic transformation matrix
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


def inverse_rigid_transformation(arr: np.ndarray) -> np.ndarray:
    """Compute the inverse of a rigid transformation matrix."""
    Rt = arr[:3, :3].T
    tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


T_vel_cam = inverse_rigid_transformation(T1)

# Define boundaries and limits
lim_x = [0, 50]
lim_y = [-10, 10]
lim_z = [-3.5, 1]
pixel_lim = 5


class RoadSegmentation3D:
    def __init__(self):
        rospy.init_node("segmentationTO3d")

        # Publishers for the left and right boundary point clouds
        self.left_boundary_pub = rospy.Publisher(
            "/road_segment_3d/left_boundary", PointCloud2, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            "/road_segment_3d/right_boundary", PointCloud2, queue_size=1
        )

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

        self.pcdSub = message_filters.Subscriber(
            "/lidar_tc/velodyne_points", PointCloud2
        )
        self.left_boundary_sub = message_filters.Subscriber(
            "/left_boundary", DetectedRoadArea
        )
        self.right_boundary_sub = message_filters.Subscriber(
            "/right_boundary", DetectedRoadArea
        )

        # Set up ApproximateTimeSynchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
            queue_size=10,
            slop=0.3,
            allow_headerless=False,
        )
        ts.registerCallback(self.segmentation_callback)
        rospy.loginfo("TimeSynchronizer initialized and callback registered.")
        rospy.spin()

    def create_cloud(
        self, points_3d: np.ndarray, publisher: rospy.Publisher, msgLidar: PointCloud2
    ):
        """Create and publish a PointCloud2 message from 3D points."""
        header = msgLidar.header
        pointcloud = pc2.create_cloud(header, self.fields, points_3d)
        publisher.publish(pointcloud)
        rospy.loginfo("Published point cloud with %d points.", len(points_3d))

    def segmentation_callback(
        self,
        msgLidar: PointCloud2,
        msgLeftBoundary: DetectedRoadArea,
        msgRightBoundary: DetectedRoadArea,
    ):
        """Callback function to process lidar and contour data for road segmentation."""
        starttime = time.time()
        # Extract left and right boundary points from messages
        left_boundary_points = np.array(msgLeftBoundary.RoadArea.data).reshape(-1, 2)
        right_boundary_points = np.array(msgRightBoundary.RoadArea.data).reshape(-1, 2)

        # Convert Lidar point cloud message to numpy array
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T

        # Crop point cloud to reduce computational expense
        pc_arr = self.crop_pointcloud(points)

        # Apply the transformation
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr.T))
        uv1 = torch.matmul(torch.tensor(rect), m1)
        u, v = (uv1[:2, :] / uv1[2, :]).numpy()

        # Find matching points between the lidar and contour points
        left_boundary_3d = self.find_matching_points_kdtree(
            left_boundary_points, u, v, pc_arr
        )
        right_boundary_3d = self.find_matching_points_kdtree(
            right_boundary_points, u, v, pc_arr
        )

        # Publish left and right boundary points if available
        if left_boundary_3d.size > 0:
            self.create_cloud(left_boundary_3d, self.left_boundary_pub, msgLidar)

        if right_boundary_3d.size > 0:
            self.create_cloud(right_boundary_3d, self.right_boundary_pub, msgLidar)

        print("Processing time: %.3f seconds", time.time() - starttime)
        # rospy.loginfo("Processing time: %.3f seconds", time.time() - starttime)

    def find_matching_points(
        self,
        boundary_points: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        pc_arr: np.ndarray,
    ) -> np.ndarray:
        """Find 3D points corresponding to boundary points."""
        idx = np.array([])

        # result = np.where(arr > 0, arr, 0)
        # result = arr[arr > 0]
        for contour_point in boundary_points:
            matches = np.where(
                (u + pixel_lim >= contour_point[0])
                & (u - pixel_lim <= contour_point[0])
                & (v + pixel_lim >= contour_point[1])
                & (v - pixel_lim <= contour_point[1])
            )[0]
            idx = np.concatenate(
                (idx, matches)
            )  # Use numpy's concatenate for efficiency

        return pc_arr[idx.astype(int)] if idx.size > 0 else np.empty((0, 4))

    def find_matching_points_kdtree(
        self,
        boundary_points: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        pc_arr: np.ndarray,
    ) -> np.ndarray:
        """Find 3D points corresponding to boundary points."""
        tree = KDTree(np.column_stack((u, v)))
        idx = []

        for contour_point in boundary_points:
            matches = tree.query_ball_point(contour_point, pixel_lim)
            idx.extend(matches)

        return pc_arr[np.array(idx)] if idx else np.empty((0, 4))

    def crop_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """Crop the point cloud to only include points within the defined 3D limits."""
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