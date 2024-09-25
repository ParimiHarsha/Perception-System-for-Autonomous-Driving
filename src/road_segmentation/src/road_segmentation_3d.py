import message_filters
import numpy as np
np.float = np.float64
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import torch
from sensor_msgs.msg import PointCloud2

from road_segmentation.msg import DetectedRoadArea

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


def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.
    """
    Rt = arr[:3, :3].T
    tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


T_vel_cam = inverse_rigid_transformation(T1)

# Define boundaries and limits
lim_x = [2.5, 75]
lim_y = [-15, 15]
lim_z = [-3.5, 5]
height, width = 772, 1024
pixel_lim = 5


class RoadSegmentation3d:
    def __init__(self):
        rospy.init_node("segmentationTO3d")

        # self.segmentation_pub = rospy.Publisher(
        #     "/road_segment_3d", PointCloud2, queue_size=1
        # )
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
        # self.segmentation_sub = message_filters.Subscriber(
        #     "/contour_points", DetectedRoadArea
        # )
        self.left_boundary_sub = message_filters.Subscriber(
            "/left_boundary", DetectedRoadArea
        )
        self.right_boundary_sub = message_filters.Subscriber(
            "/right_boundary", DetectedRoadArea
        )

        self.header = std_msgs.msg.Header()
        self.header.frame_id = "lidar_tc"

        # Set up ApproximateTimeSynchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            # [self.pcdSub, self.segmentation_sub],
             [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
            queue_size=10,  # Reduced queue size for faster processing
            slop=10,  # Adjusted slop for stricter synchronization
            allow_headerless=True,  # Ensure headers are present
        )
        ts.registerCallback(self.segmentation_callback)
        rospy.loginfo('TimeSynchronizer initialized and callback registered.')
        rospy.spin()

    # def create_cloud(self, points_3d):
    #     """
    #     Create and publish a PointCloud2 message from 3D points.
    #     """
    #     self.header.stamp = rospy.Time.now()
    #     pointcloud = pc2.create_cloud(self.header, self.fields, points_3d)
    #     self.segmentation_pub.publish(pointcloud)
    #     rospy.loginfo("Published point cloud with %d points.", len(points_3d))

    def create_cloud(self, points_3d, publisher):
        """
        Create and publish a PointCloud2 message from 3D points.
        """
        self.header.stamp = rospy.Time.now()
        pointcloud = pc2.create_cloud(self.header, self.fields, points_3d)
        publisher.publish(pointcloud)
        rospy.loginfo("Published point cloud with %d points.", len(points_3d))


    def segmentation_callback(self, msgLidar, msgLeftBoundary, msgRightBoundary):
        """
        Callback function to process lidar and contour data for road segmentation.
        """
        rospy.loginfo('Inside callback function.')
        # contour_points = np.array(msgContours.RoadArea.data).reshape(-1, 2)
        # Extract left and right boundary points from messages
        left_boundary_points = np.array(msgLeftBoundary.RoadArea.data).reshape(-1, 2)
        right_boundary_points = np.array(msgRightBoundary.RoadArea.data).reshape(-1, 2)


        # Convert Lidar point cloud message to numpy array
        pc = ros_numpy.numpify(msgLidar)
        points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T

        # Crop point cloud to reduce computational expense
        pc_arr = self.crop_pointcloud(points)
        pc_arr_pick = np.transpose(pc_arr)

        # Apply the transformation
        m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr_pick))
        uv1 = torch.matmul(torch.tensor(rect), m1)
        uv1[:2, :] /= uv1[2, :]
        u, v = uv1[0, :].numpy(), uv1[1, :].numpy()

        # Find matching points between the lidar and contour points
        # line_3d = []
        # for contour_point in contour_points:
        #     idx = np.where(
        #         (u + pixel_lim >= contour_point[0])
        #         & (u - pixel_lim <= contour_point[0])
        #         & (v + pixel_lim >= contour_point[1])
        #         & (v - pixel_lim <= contour_point[1])
        #     )[0]

        #     if idx.size > 0:
        #         for i in idx:
        #             line_3d.append(
        #                 [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
        #             )

        # if line_3d:
        #     self.create_cloud(line_3d)
        #     rospy.loginfo(f"Published {len(line_3d)} points in the point cloud.")

        # Process left boundary points
        left_boundary_3d = []
        for contour_point in left_boundary_points:
            idx = np.where(
                (u + pixel_lim >= contour_point[0])
                & (u - pixel_lim <= contour_point[0])
                & (v + pixel_lim >= contour_point[1])
                & (v - pixel_lim <= contour_point[1])
            )[0]

            if idx.size > 0:
                for i in idx:
                    left_boundary_3d.append(
                        [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
                    )

        # Process right boundary points
        right_boundary_3d = []
        for contour_point in right_boundary_points:
            idx = np.where(
                (u + pixel_lim >= contour_point[0])
                & (u - pixel_lim <= contour_point[0])
                & (v + pixel_lim >= contour_point[1])
                & (v - pixel_lim <= contour_point[1])
            )[0]

            if idx.size > 0:
                for i in idx:
                    right_boundary_3d.append(
                        [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
                    )

        # Publish left boundary points
        if left_boundary_3d:
            self.create_cloud(left_boundary_3d, self.left_boundary_pub)
            rospy.loginfo(f"Published {len(left_boundary_3d)} points in the left boundary point cloud.")

        # Publish right boundary points
        if right_boundary_3d:
            self.create_cloud(right_boundary_3d, self.right_boundary_pub)
            rospy.loginfo(f"Published {len(right_boundary_3d)} points in the right boundary point cloud.")



    def crop_pointcloud(self, pointcloud):
        """
        Crop the point cloud to only include points within the defined 3D limits.
        """
        mask = np.where(
            (pointcloud[:, 0] >= lim_x[0])
            & (pointcloud[:, 0] <= lim_x[1])
            & (pointcloud[:, 1] >= lim_y[0])
            & (pointcloud[:, 1] <= lim_y[1])
            & (pointcloud[:, 2] >= lim_z[0])
            & (pointcloud[:, 2] <= lim_z[1])
        )
        return pointcloud[mask]


if __name__ == "__main__":
    RoadSegmentation3d()








# import message_filters
# import numpy as np
# np.float = np.float64
# import ros_numpy
# import rospy
# import sensor_msgs.point_cloud2 as pc2
# import std_msgs.msg
# import torch
# from sensor_msgs.msg import PointCloud2

# from road_segmentation.msg import DetectedRoadArea

# # Camera intrinsic parameters
# rect = np.array(
#     [
#         [1757.3969095, 0.0, 548.469263, 0.0],
#         [0.0, 1758.613861, 404.160806, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#     ]
# )

# # Camera to lidar extrinsic transformation matrix
# T1 = np.array(
#     [
#         [
#             -0.01286594650077832,
#             -0.0460667467684005,
#             0.9988555061983764,
#             1.343301892280579,
#         ],
#         [
#             -0.9971783142793244,
#             -0.07329508411852753,
#             -0.01622467796607624,
#             0.2386326789855957,
#         ],
#         [
#             0.07395861648032626,
#             -0.9962457957182222,
#             -0.04499375025580721,
#             -0.7371386885643005,
#         ],
#         [0.0, 0.0, 0.0, 1.0],
#     ]
# )


# def inverse_rigid_transformation(arr):
#     """
#     Compute the inverse of a rigid transformation matrix.
#     """
#     Rt = arr[:3, :3].T
#     tt = -np.dot(Rt, arr[:3, 3])
#     return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


# T_vel_cam = inverse_rigid_transformation(T1)

# # Define boundaries and limits
# lim_x = [2.5, 75]
# lim_y = [-15, 15]
# lim_z = [-3.5, 5]
# height, width = 772, 1024
# pixel_lim = 5


# class RoadSegmentation3d:
#     def __init__(self):
#         rospy.init_node("segmentationTO3d")

#         # Publishers for the left and right boundary point clouds
#         self.left_boundary_pub = rospy.Publisher(
#             "/road_segment_3d/left_boundary", PointCloud2, queue_size=1
#         )
#         self.right_boundary_pub = rospy.Publisher(
#             "/road_segment_3d/right_boundary", PointCloud2, queue_size=1
#         )
#         self.fields = [
#             pc2.PointField(
#                 name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1
#             ),
#             pc2.PointField(
#                 name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1
#             ),
#             pc2.PointField(
#                 name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1
#             ),
#             pc2.PointField(
#                 name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1
#             ),
#         ]

#         self.pcdSub = message_filters.Subscriber(
#             "/lidar_tc/velodyne_points", PointCloud2
#         )
#         self.left_boundary_sub = message_filters.Subscriber(
#             "/left_boundary", DetectedRoadArea
#         )
#         self.right_boundary_sub = message_filters.Subscriber(
#             "/right_boundary", DetectedRoadArea
#         )

#         self.header = std_msgs.msg.Header()
#         self.header.frame_id = "lidar_tc"

#         # Set up ApproximateTimeSynchronizer
#         ts = message_filters.ApproximateTimeSynchronizer(
#             [self.pcdSub, self.left_boundary_sub, self.right_boundary_sub],
#             queue_size=10,  # Reduced queue size for faster processing
#             slop=10,  # Adjusted slop for stricter synchronization
#             allow_headerless=True,  # Ensure headers are present
#         )
#         ts.registerCallback(self.segmentation_callback)
#         rospy.loginfo('TimeSynchronizer initialized and callback registered.')
#         rospy.spin()

#     def create_cloud(self, points_3d, publisher):
#         """
#         Create and publish a PointCloud2 message from 3D points.
#         """
#         self.header.stamp = rospy.Time.now()
#         pointcloud = pc2.create_cloud(self.header, self.fields, points_3d)
#         publisher.publish(pointcloud)
#         rospy.loginfo("Published point cloud with %d points.", len(points_3d))

#     def segmentation_callback(self, msgLidar, msgLeftBoundary, msgRightBoundary):
#         """
#         Callback function to process lidar and contour data for road segmentation.
#         """
#         rospy.loginfo('Inside callback function.')

#         # Extract left and right boundary points from messages
#         left_boundary_points = np.array(msgLeftBoundary.RoadArea.data).reshape(-1, 2)
#         right_boundary_points = np.array(msgRightBoundary.RoadArea.data).reshape(-1, 2)

#         # Convert Lidar point cloud message to numpy array
#         pc = ros_numpy.numpify(msgLidar)
#         points = np.vstack((pc["x"], pc["y"], pc["z"], np.ones(pc["x"].shape[0]))).T

#         # Crop point cloud to reduce computational expense
#         pc_arr = self.crop_pointcloud(points)
#         pc_arr_pick = np.transpose(pc_arr)

#         # Apply the transformation
#         m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr_pick))
#         uv1 = torch.matmul(torch.tensor(rect), m1)
#         uv1[:2, :] /= uv1[2, :]
#         u, v = uv1[0, :].numpy(), uv1[1, :].numpy()

#         # Process left boundary points
#         left_boundary_3d = []
#         for contour_point in left_boundary_points:
#             idx = np.where(
#                 (u + pixel_lim >= contour_point[0])
#                 & (u - pixel_lim <= contour_point[0])
#                 & (v + pixel_lim >= contour_point[1])
#                 & (v - pixel_lim <= contour_point[1])
#             )[0]

#             if idx.size > 0:
#                 for i in idx:
#                     left_boundary_3d.append(
#                         [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
#                     )

#         # Process right boundary points
#         right_boundary_3d = []
#         for contour_point in right_boundary_points:
#             idx = np.where(
#                 (u + pixel_lim >= contour_point[0])
#                 & (u - pixel_lim <= contour_point[0])
#                 & (v + pixel_lim >= contour_point[1])
#                 & (v - pixel_lim <= contour_point[1])
#             )[0]

#             if idx.size > 0:
#                 for i in idx:
#                     right_boundary_3d.append(
#                         [pc_arr_pick[0][i], pc_arr_pick[1][i], pc_arr_pick[2][i], 1]
#                     )

#         # Publish left boundary points
#         if left_boundary_3d:
#             self.create_cloud(left_boundary_3d, self.left_boundary_pub)
#             rospy.loginfo(f"Published {len(left_boundary_3d)} points in the left boundary point cloud.")

#         # Publish right boundary points
#         if right_boundary_3d:
#             self.create_cloud(right_boundary_3d, self.right_boundary_pub)
#             rospy.loginfo(f"Published {len(right_boundary_3d)} points in the right boundary point cloud.")

#     def crop_pointcloud(self, pointcloud):
#         """
#         Crop the point cloud to only include points within the defined 3D limits.
#         """
#         mask = np.where(
#             (pointcloud[:, 0] >= lim_x[0])
#             & (pointcloud[:, 0] <= lim_x[1])
#             & (pointcloud[:, 1] >= lim_y[0])
#             & (pointcloud[:, 1] <= lim_y[1])
#             & (pointcloud[:, 2] >= lim_z[0])
#             & (pointcloud[:, 2] <= lim_z[1])
#         )
#         return pointcloud[mask]


# if __name__ == "__main__":
#     RoadSegmentation3d()
