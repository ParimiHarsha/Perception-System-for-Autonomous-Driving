'''
# cython: language_level=3



from libc.stdlib cimport malloc, free

# Regular Python imports
import numpy as np
np.float = np.float64
import message_filters
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2  # Make sure to import this correctly
from sensor_msgs.msg import PointCloud2  # Regular import for PointCloud2
import time
import torch
from road_segmentation.msg import DetectedRoadArea
from scipy.spatial import KDTree

# Make sure to initialize numpy
cnp.import_array()

# Your Numpy arrays
rect = np.array([
    [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
    [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
    [0, 0.0, 1.0, 0],
], dtype=np.float64)

T1 = np.array([
    [-4.8076040039157775e-03, 1.1565175070195832e-02, 9.9992156375854679e-01, 1.3626313209533691e00],
    [-9.9997444266988167e-01, -5.3469003551928074e-03, -4.7460155553246119e-03, 2.0700573921203613e-02],
    [5.2915924636425249e-03, -9.9991882539643562e-01, 1.1590585274754983e-02, -9.1730421781539917e-01],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

#lim_x = np.array([2.5, 50], dtype=np.float64)
lim_x = np.array([-100, 100], dtype=np.float64)
lim_y = np.array([-10, 10], dtype=np.float64)
lim_z = np.array([-3.5, 1], dtype=np.float64)
pixel_lim = 5

cdef cnp.ndarray inverse_rigid_transformation(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    """Compute the inverse of a rigid transformation matrix."""
    cdef cnp.ndarray Rt = arr[:3, :3].T
    cdef cnp.ndarray tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))

cdef class RoadSegmentation3D:
    cdef object left_boundary_pub  # Define left boundary publisher
    cdef object right_boundary_pub  # Define right boundary publisher
    cdef object fields # Define fields
    cdef object pcdSub  # Define PointCloud2 subscriber
    cdef object left_boundary_sub  # Define left boundary subscriber
    cdef object right_boundary_sub  # Define right boundary subscriber
    cdef object ts  # Time Synchronizer
    cdef object header  # Header for PointCloud2
    cdef object pointcloud  # PointCloud2 message

    def __init__(self):
        rospy.init_node("segmentationTO3d")

        print("left_boundary_pub initialized.")
        self.left_boundary_pub = rospy.Publisher("/road_segment_3d/left_boundary", PointCloud2, queue_size=1)
        self.right_boundary_pub = rospy.Publisher("/road_segment_3d/right_boundary", PointCloud2, queue_size=1)

        self.fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]

        self.pcdSub = message_filters.Subscriber("/lidar_tc/velodyne_points", PointCloud2)
        self.left_boundary_sub = message_filters.Subscriber("/left_boundary", DetectedRoadArea)
        self.right_boundary_sub = message_filters.Subscriber("/right_boundary", DetectedRoadArea)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
            queue_size=10,
            slop=0.3,
            allow_headerless=False,
        )
        ts.registerCallback(self.segmentation_callback)
        rospy.loginfo("TimeSynchronizer initialized and callback registered.")
        rospy.spin()

    cpdef void create_cloud(self, cnp.ndarray points_3d, object publisher, object msgLidar):
        """Create and publish a PointCloud2 message from 3D points."""
        cdef object header = msgLidar.header
        pointcloud = pc2.create_cloud(header, self.fields, points_3d)
        publisher.publish(pointcloud)
        rospy.loginfo("Published point cloud with %d points.", len(points_3d))

    cpdef void segmentation_callback(self, object msgLidar, object msgLeftBoundary, object msgRightBoundary):
        """Callback function to process lidar and contour data for road segmentation."""
        starttime = time.time()

        cdef cnp.ndarray[cnp.float64_t, ndim=2] left_boundary_points = np.array(msgLeftBoundary.RoadArea.data).reshape(-1, 2)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right_boundary_points = np.array(msgRightBoundary.RoadArea.data).reshape(-1, 2)

        cdef cnp.ndarray points = ros_numpy.numpify(msgLidar)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] points_arr = np.vstack((points["x"], points["y"], points["z"], np.ones(points["x"].shape[0]))).T

        cdef cnp.ndarray[cnp.float64_t, ndim=2] pc_arr = self.crop_pointcloud(points_arr)

        cdef cnp.ndarray[cnp.float64_t, ndim=2] T_vel_cam = inverse_rigid_transformation(np.array(T1, dtype=np.float64))

        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr.T))
        uv1 = torch.matmul(torch.tensor(rect), m1)
        cdef cnp.ndarray u = (uv1[0, :] / uv1[2, :]).numpy()
        cdef cnp.ndarray v = (uv1[1, :] / uv1[2, :]).numpy()

        cdef cnp.ndarray[cnp.float64_t, ndim=2] left_boundary_3d = self.find_matching_points_kdtree(left_boundary_points, u, v, pc_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right_boundary_3d = self.find_matching_points_kdtree(right_boundary_points, u, v, pc_arr)

        if left_boundary_3d.size > 0:
            self.create_cloud(left_boundary_3d, self.left_boundary_pub, msgLidar)

        if right_boundary_3d.size > 0:
            self.create_cloud(right_boundary_3d, self.right_boundary_pub, msgLidar)

        rospy.loginfo("Processing time: %.3f seconds", time.time() - starttime)

    cpdef cnp.ndarray find_matching_points_kdtree(self, cnp.ndarray[cnp.float64_t, ndim=2] boundary_points, cnp.ndarray[cnp.float64_t, ndim=1] u, cnp.ndarray[cnp.float64_t, ndim=1] v, cnp.ndarray[cnp.float64_t, ndim=2] pc_arr):
        tree = KDTree(np.column_stack((u, v)))
        cdef list idx = []

        for contour_point in boundary_points:
            matches = tree.query_ball_point(contour_point, pixel_lim)
            idx.extend(matches)

        return pc_arr[np.array(idx)] if idx else np.empty((0, 4))

    cpdef cnp.ndarray crop_pointcloud(self, cnp.ndarray[cnp.float64_t, ndim=2] pointcloud):
        """Crop the pointcloud based on x, y, and z limits."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_x_typed = np.array(lim_x, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_y_typed = np.array(lim_y, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_z_typed = np.array(lim_z, dtype=np.float64)

        cdef cnp.ndarray mask = (
            (pointcloud[:, 0] >= lim_x_typed[0]) & (pointcloud[:, 0] <= lim_x_typed[1]) &
            (pointcloud[:, 1] >= lim_y_typed[0]) & (pointcloud[:, 1] <= lim_y_typed[1]) &
            (pointcloud[:, 2] >= lim_z_typed[0]) & (pointcloud[:, 2] <= lim_z_typed[1])
        )
        return pointcloud[mask]
'''

# cython: language_level=3

# Cython cimports
cimport numpy as cnp

from libc.stdlib cimport malloc, free
import numpy as np
np.float = np.float64
import message_filters
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import time
import torch
from road_segmentation.msg import DetectedRoadArea
from scipy.spatial import KDTree

# Initialize numpy arrays
cnp.import_array()

# Your Numpy arrays
cdef cnp.ndarray rect = np.array([
    [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
    [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
    [0, 0.0, 1.0, 0],
], dtype=np.float64)

cdef cnp.ndarray T1 = np.array([
    [-4.8076040039157775e-03, 1.1565175070195832e-02, 9.9992156375854679e-01, 1.3626313209533691e00],
    [-9.9997444266988167e-01, -5.3469003551928074e-03, -4.7460155553246119e-03, 2.0700573921203613e-02],
    [5.2915924636425249e-03, -9.9991882539643562e-01, 1.1590585274754983e-02, -9.1730421781539917e-01],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# Constants for pointcloud cropping
cdef cnp.ndarray lim_x = np.array([2.5, 50], dtype=np.float64)
cdef cnp.ndarray lim_y = np.array([-10, 10], dtype=np.float64)
cdef cnp.ndarray lim_z = np.array([-3.5, 1], dtype=np.float64)
cdef int pixel_lim = 5

cdef cnp.ndarray inverse_rigid_transformation(cnp.ndarray[cnp.float64_t, ndim=2] arr):
    """Compute the inverse of a rigid transformation matrix."""
    cdef cnp.ndarray Rt = arr[:3, :3].T
    cdef cnp.ndarray tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))

cdef class RoadSegmentation3D:
    cdef object left_boundary_pub  # Publisher for left boundary
    cdef object right_boundary_pub  # Publisher for right boundary
    cdef object fields  # Point cloud fields
    cdef object pcdSub  # Point cloud subscriber
    cdef object left_boundary_sub  # Left boundary subscriber
    cdef object right_boundary_sub  # Right boundary subscriber
    cdef object ts  # Time synchronizer
    cdef object header  # Header for PointCloud2
    cdef object pointcloud  # PointCloud2 message

    def __init__(self):
        rospy.init_node("segmentationTO3d")
        print("left_boundary_pub initialized.")
        self.left_boundary_pub = rospy.Publisher("/road_segment_3d/left_boundary", PointCloud2, queue_size=1)
        self.right_boundary_pub = rospy.Publisher("/road_segment_3d/right_boundary", PointCloud2, queue_size=1)

        self.fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]

        self.pcdSub = message_filters.Subscriber("/lidar_tc/velodyne_points", PointCloud2)
        self.left_boundary_sub = message_filters.Subscriber("/left_boundary", DetectedRoadArea)
        self.right_boundary_sub = message_filters.Subscriber("/right_boundary", DetectedRoadArea)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
            queue_size=10,
            slop=0.3,
            allow_headerless=False,
        )
        ts.registerCallback(self.segmentation_callback)
        rospy.loginfo("TimeSynchronizer initialized and callback registered.")
        rospy.spin()

    cpdef void create_cloud(self, cnp.ndarray points_3d, object publisher, object msgLidar):
        """Create and publish a PointCloud2 message from 3D points."""
        cdef object header = msgLidar.header
        pointcloud = pc2.create_cloud(header, self.fields, points_3d)
        publisher.publish(pointcloud)
        rospy.loginfo("Published point cloud with %d points.", len(points_3d))

    cpdef void segmentation_callback(self, object msgLidar, object msgLeftBoundary, object msgRightBoundary):
        """Callback function to process lidar and contour data for road segmentation."""
        starttime = time.time()

        # Convert left and right boundaries to arrays and reshape
        cdef cnp.ndarray[cnp.float64_t, ndim=2] left_boundary_points = np.array(msgLeftBoundary.RoadArea.data).reshape(-1, 2)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right_boundary_points = np.array(msgRightBoundary.RoadArea.data).reshape(-1, 2)

        # Convert the PointCloud2 message to a NumPy array
        cdef cnp.ndarray points = ros_numpy.numpify(msgLidar)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] points_arr = np.vstack((points["x"], points["y"], points["z"], np.ones(points["x"].shape[0]))).T

        # Crop the point cloud based on limits
        cdef cnp.ndarray[cnp.float64_t, ndim=2] pc_arr = self.crop_pointcloud(points_arr)

        # Compute the transformation from camera to lidar
        cdef cnp.ndarray[cnp.float64_t, ndim=2] T_vel_cam = inverse_rigid_transformation(np.array(T1, dtype=np.float64))

        # Perform the matrix operations for camera-to-lidar transformation
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr.T))
        uv1 = torch.matmul(torch.tensor(rect), m1)
        cdef cnp.ndarray u = (uv1[0, :] / uv1[2, :]).numpy()
        cdef cnp.ndarray v = (uv1[1, :] / uv1[2, :]).numpy()

        # Find matching points in the 3D point cloud for both boundaries
        cdef cnp.ndarray[cnp.float64_t, ndim=2] left_boundary_3d = self.find_matching_points_kdtree(left_boundary_points, u, v, pc_arr)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] right_boundary_3d = self.find_matching_points_kdtree(right_boundary_points, u, v, pc_arr)

        # Publish the results
        if left_boundary_3d.size > 0:
            self.create_cloud(left_boundary_3d, self.left_boundary_pub, msgLidar)

        if right_boundary_3d.size > 0:
            self.create_cloud(right_boundary_3d, self.right_boundary_pub, msgLidar)

        rospy.loginfo("Processing time: %.3f seconds", time.time() - starttime)

    cpdef cnp.ndarray find_matching_points_kdtree(self, cnp.ndarray[cnp.float64_t, ndim=2] boundary_points, cnp.ndarray[cnp.float64_t, ndim=1] u, cnp.ndarray[cnp.float64_t, ndim=1] v, cnp.ndarray[cnp.float64_t, ndim=2] pc_arr):
        """Find matching points using KD-tree."""
        cdef object tree = KDTree(np.column_stack((u, v)))
        cdef list idx = []

        # Use cdef for loop to iterate over boundary points
        cdef int i
        for i in range(boundary_points.shape[0]):
            contour_point = boundary_points[i]
            matches = tree.query_ball_point(contour_point, pixel_lim)
            idx.extend(matches)

        return pc_arr[np.array(idx)] if idx else np.empty((0, 4))
    
    # cpdef cnp.ndarray crop_pointcloud(self, cnp.ndarray[cnp.float64_t, ndim=2] pointcloud):
    #     cdef int n = pointcloud.shape[0]
    #     cdef int m = 0
    #     cdef cnp.ndarray filtered = np.empty((n, 4), dtype=np.float64)
        
    #     for i in range(n):
    #         if (pointcloud[i, 0] >= lim_x[0] and pointcloud[i, 0] <= lim_x[1] and
    #             pointcloud[i, 1] >= lim_y[0] and pointcloud[i, 1] <= lim_y[1] and
    #             pointcloud[i, 2] >= lim_z[0] and pointcloud[i, 2] <= lim_z[1]):
    #             filtered[m] = pointcloud[i]
    #             m += 1
    #         print(filtered[:m])
    #     return filtered[:m]  # Return only the valid entries
    cpdef cnp.ndarray crop_pointcloud(self, cnp.ndarray[cnp.float64_t, ndim=2] pointcloud):
        """Crop the pointcloud based on x, y, and z limits."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_x_typed = lim_x
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_y_typed = lim_y
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lim_z_typed = lim_z

        cdef cnp.ndarray mask = (
            (pointcloud[:, 0] >= lim_x_typed[0]) & (pointcloud[:, 0] <= lim_x_typed[1]) &
            (pointcloud[:, 1] >= lim_y_typed[0]) & (pointcloud[:, 1] <= lim_y_typed[1]) &
            (pointcloud[:, 2] >= lim_z_typed[0]) & (pointcloud[:, 2] <= lim_z_typed[1])
        )
        return pointcloud[mask]
