#!/usr/bin/env python3

import message_filters
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from sensor_msgs.msg import PointCloud2

from src.configs import (LEFT_LANE_BOUNDARY_TOPIC, LEFT_LANE_TOPIC,
                         LIDAR_TOPIC, PROJ, RIGHT_LANE_BOUNDARY_TOPIC,
                         RIGHT_LANE_TOPIC, T1)
from ultrafastv2_ros.msg import LanePoints


def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.
    """
    Rt = arr[:3, :3].T
    tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))


T_vel_cam = inverse_rigid_transformation(T1)

lim_x = [2.5, 75]
lim_y = [-15, 15]
lim_z = [-3.5, 1]
height, width = 772, 1024
pixel_lim = 5


class realCoor:
    def __init__(self):
        print("Running the new_lineto3d transformation script\n")

        self.left_pcl_pub = rospy.Publisher(
            LEFT_LANE_BOUNDARY_TOPIC, PointCloud2, queue_size=1
        )
        print("Publishing /Left_Line3dPoints\n")
        self.right_pcl_pub = rospy.Publisher(
            RIGHT_LANE_BOUNDARY_TOPIC, PointCloud2, queue_size=1
        )
        print("Publishing /Right_Line3dPoints\n")

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

        self.pcdSub = message_filters.Subscriber(LIDAR_TOPIC, PointCloud2)
        self.lane_pointcloud = PointCloud2()
        self.used_pointcloud = PointCloud2()
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "lidar_tc"
        pointcloud = []

        self.left_pointSub = message_filters.Subscriber(
            LEFT_LANE_TOPIC, LanePoints
        )
        self.right_pointSub = message_filters.Subscriber(
            RIGHT_LANE_TOPIC, LanePoints
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcdSub, self.left_pointSub, self.right_pointSub],
            10,
            0.4,
            allow_headerless=True,
        )
        ts.registerCallback(self.pcd_callback)

        self.vis = True
        rospy.spin()

    def create_cloud(self, line_3d, which):
        if which == "left":
            self.left_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.left_pcl_pub.publish(self.left_pointcloud)
        elif which == "right":
            self.right_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.right_pcl_pub.publish(self.right_pointcloud)

    def pcd_callback(self, msgLidar, msgLeftPoint, msgRightPoint):
        self.header = msgLidar.header
        if msgLeftPoint.points != [] or msgRightPoint.points != []:
            pc = ros_numpy.numpify(msgLidar)
            points = np.zeros((pc.shape[0], 4))
            points[:, 0] = pc["x"]
            points[:, 1] = pc["y"]
            points[:, 2] = pc["z"]
            points[:, 3] = 1

            pc_arr = self.crop_pointcloud(points)
            pc_arr_pick = np.transpose(pc_arr)

            m1 = np.matmul(T_vel_cam, pc_arr_pick)

            uv1 = np.matmul(PROJ, m1)

            uv1[0, :] = np.divide(uv1[0, :], uv1[2, :])
            uv1[1, :] = np.divide(uv1[1, :], uv1[2, :])

            self.process_points(msgLeftPoint, uv1, pc_arr_pick, "left")
            self.process_points(msgRightPoint, uv1, pc_arr_pick, "right")

    def process_points(self, msgPoint, uv1, pc_arr_pick, which):
        line_3d = []

        u = uv1[0, :]
        v = uv1[1, :]
        for point in msgPoint.points:
            idx = np.where(
                ((u + pixel_lim >= point.x) & (u - pixel_lim <= point.x))
                & ((v + pixel_lim >= point.y) & (v - pixel_lim <= point.y))
            )
            idx = np.array(idx)
            # print(idx)
            if idx.size > 0:
                for i in range(idx.size):
                    # print(point.lane_id)
                    line_3d.append(
                        (
                            [
                                (pc_arr_pick[0][idx[0, i]]),
                                (pc_arr_pick[1][idx[0, i]]),
                                (pc_arr_pick[2][idx[0, i]]),
                                1,
                            ]
                        )
                    )

        if self.vis == True and line_3d != [] and len(line_3d) > 2:
            # line_3d = self.test(line_3d)
            line_3d = np.array(line_3d)
            _, idx = np.unique(line_3d[:, 0:2], axis=0, return_index=True)
            line_3d_unique = line_3d[idx]
            self.create_cloud(line_3d_unique, which)

    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where(
            (pointcloud[:, 0] >= lim_x[0])
            & (pointcloud[:, 0] <= lim_x[1])
            & (pointcloud[:, 1] >= lim_y[0])
            & (pointcloud[:, 1] <= lim_y[1])
            & (pointcloud[:, 2] >= lim_z[0])
            & (pointcloud[:, 2] <= lim_z[1])
        )
        pointcloud = pointcloud[mask]
        return pointcloud

    def test(self, line):
        output = []
        line = np.array(line)
        indd = np.lexsort((line[:, 2], line[:, 1], line[:, 0]))
        line = line[indd]
        _, idx = np.unique(line[:, 0:2], axis=0, return_index=True)
        line = line[idx]
        # print("line::", line)
        x = -np.sort(-line[:, 0])
        y = -np.sort(-line[:, 1])
        z = -np.sort(-line[:, 2])
        # print("xyz", x,y,z)
        intensity = line[:, 3]
        npts = len(x)
        s = np.zeros(npts, dtype=float)

        xl = np.linspace(np.amax(x), np.amin(x), 500)
        yl = np.linspace(np.amax(y), np.amin(y), 500)
        zl = np.linspace(np.amax(z), np.amin(z), 500)

        # Create new interpolation function for each axis against the norm
        data = np.concatenate(
            (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1
        )

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)

        #  SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)

        linepts = vv[0] * np.mgrid[-20:20:2j][:, np.newaxis]

        linepts += datamean

        l = linepts[0, 0] - linepts[1, 0]
        m = linepts[0, 1] - linepts[1, 1]
        n = linepts[0, 2] - linepts[1, 2]

        t = (xl - linepts[0, 0]) / l
        xs = xl
        ys = (t) * m + linepts[0, 1]
        zs = (t) * n + linepts[0, 2]

        return np.transpose([xs, ys, zs, np.ones(len(xs)) * 1.1])


if __name__ == "__main__":
    rospy.init_node("LaneTO3D")
    realCoor()
