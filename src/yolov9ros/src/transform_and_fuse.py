# type: ignore
"""
This script transforms 2D bounding box coordinates from image space to 3D lidar space
and integrates radar object data. It uses ROS (Robot Operating System) to synchronize
and process data from lidar, image, and radar sensors. The result is a fused 3D bounding
box representation of detected objects which is published as a ROS message.

Usage:
- This script should be executed within a ROS environment where the required
 topics (`/lidar_tc/velodyne_points`,`/yoloLiveNode/bboxInfo`, `/radar_fc/as_tx/objects`)
 are being published.
- It assumes the presence of specific message types and sensor configurations
as defined in the imported message types.

Example:
    python transform_and_fuse.py
"""


from collections import defaultdict

import message_filters
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import torch
import yaml
from derived_object_msgs.msg import ObjectWithCovarianceArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
from yolov7ros.msg import BboxCentersClass

# Camera intrinsic parameters
rect = np.array(
    [
        [1757.3969095, 0.0, 548.469263, 0.0],
        [0.0, 1758.613861, 404.160806, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

# Camera to lidar extrinsic transformation matrix
T1 = np.array(
    [
        [
            -0.01286594650077832,
            -0.0460667467684005,
            0.9988555061983764,
            1.343301892280579,
        ],
        [
            -0.9971783142793244,
            -0.07329508411852753,
            -0.01622467796607624,
            0.2386326789855957,
        ],
        [
            0.07395861648032626,
            -0.9962457957182222,
            -0.04499375025580721,
            -0.7371386885643005,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Point cloud limits
lim_x, lim_y, lim_z = [2.5, 100], [-10, 10], [-3.5, 5]
pixel_lim = 10

# Radar Limit Cutoff
radar_limit = 50  # meters
close_distance_threshold = 7  # meters

# Average Class Dimensions
with open("class_averages.yaml", "r", encoding="utf-8") as file:
    average_dimensions = yaml.safe_load(file)


def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.
    """
    Rt = arr[:3, :3].T
    tt = -Rt @ arr[:3, 3]
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


# Inverse transformation matrix from camera to lidar
T_vel_cam = inverse_rigid_transformation(T1)


class TransformFuse:
    """
    Class to handle the transformation of detected bounding box coordinates
    from 2D image space to 3D lidar space.
    """

    def __init__(self):
        rospy.loginfo("Initializing TransformFuse node...")
        # Initialize ROS publishers
        self.bbox_publish = rospy.Publisher(
            "/fused_bbox", BoundingBoxArray, queue_size=1
        )

        # Point field configuration for PointCloud2
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

        # Initialize ROS subscribers
        self.sub_lidar = message_filters.Subscriber(
            "/lidar_tc/velodyne_points", PointCloud2
        )
        self.sub_image = message_filters.Subscriber(
            "/yoloLiveNode/bboxInfo", BboxCentersClass, queue_size=1
        )
        self.sub_radar = message_filters.Subscriber(
            "/radar_fc/as_tx/objects",
            ObjectWithCovarianceArray,
            queue_size=10,
            tcp_nodelay=True,
        )

        # Initialize header
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "lidar_tc"

        # Time synchronizer for lidar, image, and radar data
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_image, self.sub_radar], 15, 0.4
        )
        ts.registerCallback(self.callback)

        rospy.loginfo("Initialization complete. Spinning...")
        rospy.spin()

    def callback(self, msgLidar, msgPoint, msgRadar):
        """
        Callback function for synchronized lidar, image, and radar data.

        Args:
            msgLidar (sensor_msgs.msg.PointCloud2): Lidar point cloud message.
            msgPoint (yolov7ros.msg.BboxCentersClass): Bounding box centers from image detection.
            msgRadar (derived_object_msgs.msg.ObjectWithCovarianceArray): Radar objects message.
        """
        rospy.loginfo("Received synchronized messages.")

        # Convert lidar data to numpy array
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T

        # Crop point cloud and transform to camera frame
        pc_arr = self.crop_pointcloud(points)
        pc_arr_pick = np.transpose(pc_arr)
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr_pick))
        uv1 = torch.matmul(torch.tensor(rect), m1)
        uv1[:2, :] /= uv1[2, :]

        center_3d = []
        label = []
        u, v = uv1[0, :].numpy(), uv1[1, :].numpy()

        # Match bounding box centers with point cloud data
        for point in msgPoint.CenterClass:
            idx = np.where(
                (u + pixel_lim >= point.x)
                & (u - pixel_lim <= point.x)
                & (v + pixel_lim >= point.y)
                & (v - pixel_lim <= point.y)
            )[0]

            if idx.size > 0:
                for i in idx:
                    center_3d.append(
                        [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
                    )
                    label.append([point.z])
        rospy.loginfo(f"Found {len(center_3d)} camera detections.")

        # Publishing the bounding boxes
        bbox_array = BoundingBoxArray()
        center_3d = np.array(center_3d)

        # Filtering out duplicate camera detections
        unique_camera_indices = []
        if center_3d.size > 0:  # Check if center_3d is empty
            close_distance_threshold_camera = 0.5
            db = DBSCAN(eps=close_distance_threshold_camera, min_samples=1).fit(
                center_3d[:, :3]
            )

            for lab in set(db.labels_):
                indices = np.where(db.labels_ == lab)[0]
                unique_camera_indices.append(
                    indices[0]
                )  # Taking the first point in the cluster

        rospy.loginfo(f"Found {len(unique_camera_indices)} unique camera detections.")

        # Finding matching detections b/w camera and radar
        camera_detections = unique_camera_indices
        radar_detections = msgRadar.objects

        matched_pairs = []
        for i in camera_detections:
            for j, rad_det in enumerate(radar_detections):
                # cam_position = cam_det[:3]
                cam_position = center_3d[i][:3]
                rad_position = np.array(
                    [
                        rad_det.pose.pose.position.x,
                        rad_det.pose.pose.position.y,
                        rad_det.pose.pose.position.z,
                    ]
                )
                distance = np.linalg.norm(cam_position - rad_position)
                if distance < close_distance_threshold:
                    matched_pairs.append((i, j))

        rospy.loginfo(f"Matched pairs: {matched_pairs}")
        matched_dict = defaultdict(list)
        for i, j in matched_pairs:
            matched_dict[i].append(j)

        # Add 3D centers to bounding box array
        for i, box in enumerate(center_3d):  # camera detections
            if i in unique_camera_indices:
                class_ = int(label[i][0])
                bbox = BoundingBox()
                bbox.header.stamp = msgLidar.header.stamp
                bbox.header.frame_id = msgLidar.header.frame_id
                bbox.pose.position.x, bbox.pose.position.y, bbox.pose.position.z = box[
                    :3
                ]
                # bbox dimensions
                bbox.dimensions.x, bbox.dimensions.y, bbox.dimensions.z = (
                    average_dimensions[class_]["dimensions"][2],
                    average_dimensions[class_]["dimensions"][1],
                    average_dimensions[class_]["dimensions"][0],
                )
                bbox.pose.orientation.w = 1
                bbox.value = 1
                bbox.label = int(label[i][0])  # class number
                bbox_array.boxes.append(bbox)

        # Add radar objects to bounding box array
        for i, obj in enumerate(msgRadar.objects):  # radar detections
            if i not in [g for f, g in matched_pairs]:
                bbox = BoundingBox()
                bbox.header = msgRadar.header
                bbox.pose.position.x = obj.pose.pose.position.x
                bbox.pose.position.y = obj.pose.pose.position.y
                bbox.pose.position.z = obj.pose.pose.position.z
                bbox.dimensions.x = 1.5
                bbox.dimensions.y = 1.5
                bbox.dimensions.z = 1.5
                bbox_array.boxes.append(bbox)

        bbox_array.header.frame_id = msgLidar.header.frame_id
        self.bbox_publish.publish(bbox_array)
        rospy.loginfo(f"Published {len(bbox_array.boxes)} fused bounding boxes.\n")

    def compute_distance_matrix(self, camera_detections, radar_detections):
        """
        Calculate the distance matrix between camera and radar detections.

        Args:
            camera_detections (np.ndarray): Camera detection coordinates.
            radar_detections (list): List of radar detections.

        Returns:
            np.ndarray: Distance matrix.
        """
        num_camera = camera_detections.shape[0]
        num_radar = len(radar_detections)

        distance_matrix = np.zeros((num_camera, num_radar))
        for i in range(num_camera):
            for j in range(num_radar):
                distance_matrix[i, j] = np.linalg.norm(
                    camera_detections[i][0] - radar_detections[j].pose.pose.position.x
                )
        return distance_matrix

    def crop_pointcloud(self, pointcloud):
        """
        Crop the point cloud to the region of interest.

        Args:
            pointcloud (np.ndarray): Input point cloud data.

        Returns:
            np.ndarray: Cropped point cloud data.
        """
        mask = np.all(
            [
                (pointcloud[:, 0] >= lim_x[0]),
                (pointcloud[:, 0] <= lim_x[1]),
                (pointcloud[:, 1] >= lim_y[0]),
                (pointcloud[:, 1] <= lim_y[1]),
                (pointcloud[:, 2] >= lim_z[0]),
                (pointcloud[:, 2] <= lim_z[1]),
            ],
            axis=0,
        )
        return pointcloud[mask]


if __name__ == "__main__":
    rospy.init_node("TransformFuse")
    TransformFuse()
