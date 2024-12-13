#!/usr/bin/env python3

'''
Author: Pardis Taghavi, Jonas Lossner
Modified by: Paavan Bagla
Texas A&M University - Fall 2024
'''

import std_msgs.msg
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import message_filters
from lane_detection.msg import LanePoints
import sensor_msgs.point_cloud2 as pc2
np.float = np.float64
import ros_numpy

# #Camera Matrix
# rect = np.array(
#     [
#         [1757.3969095, 0.0, 548.469263, 0.0],
#         [0.0, 1758.613861, 404.160806, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#     ]
# )

rect = np.array([ [3.5612204509314029e+03/2, 0., 9.9143998670769213e+02/2, 0.],
       [0, 3.5572532571086072e+03/2, 7.8349772942764150e+02/2, 0.],
       [0, 0., 1. ,0 ]])

proj = np.array(
    [
        [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
        [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
        [0, 0.0, 1.0, 0],
    ]
)

# #Camera-Lidar matrix 
# T1 = np.array(
#     [
#         [
#             -0.01286594650077832,
#             -0.0460667467684005,
#             0.9988555061983764,
#             -3.343301892280579,
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

# T1=np.array([ [-1.1946479179686226e-01, 1.0441412892891744e-02,
#        9.9278353150006082e-01, 1.3941261768341064e+00],
#        [-9.9271490860797629e-01, 1.4517090902935403e-02,
#        -1.1960921494254950e-01, 8.1298440694808960e-01],
#        [-1.5661217972733510e-02, -9.9984010270068480e-01,
#        8.6310650034898684e-03, -8.7750834226608276e-01], [0., 0., 0., 1.], ])

# T1=np.array([[ -4.8076040039157775e-03, 1.1565175070195832e-02,
#        9.9992156375854679e-01, 1.3626313209533691e+00],
#        [-9.9997444266988167e-01, -5.3469003551928074e-03,
#        -4.7460155553246119e-03, 2.0700573921203613e-02],
#        [5.2915924636425249e-03, -9.9991882539643562e-01,
#        1.1590585274754983e-02, -9.1730421781539917e-01], [0., 0., 0., 1.] ])


T1=np.array([[  -0.0022981,  0.0123123,  0.9999216, 1.3626313209533691e+00],
       [-0.9792343,  0.2026763, -0.0047461, 2.0700573921203613e-02],
       [-0.2027189, -0.9791684,  0.0115908, -9.1730421781539917e-01], [0., 0., 0., 1.] ])
T1=np.array([[ -4.8076040039157775e-03, 1.1565175070195832e-02,
       9.9992156375854679e-01, 1.3626313209533691e+00],
       [-9.9997444266988167e-01, -5.3469003551928074e-03,
       -4.7460155553246119e-03, 2.0700573921203613e-02],
       [5.2915924636425249e-03, -9.9991882539643562e-01,
       1.1590585274754983e-02, -9.1730421781539917e-01], [0., 0., 0., 1.] ])
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
##################################################################################################
# def inverse_rigid_transformation(arr):
#     irt=np.zeros_like(arr)
#     Rt=np.transpose(arr[:3,:3])
#     tt=-np.matmul(Rt,arr[:3,3])
#     irt[:3,:3]=Rt
#     irt[0,3]=tt[0]
#     irt[1,3]=tt[1]
#     irt[2,3]=tt[2]
#     irt[3,3]=1
#     return irt

def inverse_rigid_transformation(arr):
    """
    Compute the inverse of a rigid transformation matrix.
    """
    Rt = arr[:3, :3].T
    tt = -np.dot(Rt, arr[:3, 3])
    return np.vstack((np.column_stack((Rt, tt)), [0, 0, 0, 1]))

##################################################################################################

T_vel_cam=inverse_rigid_transformation(T1)

##############################################################################################
# lim_x=[2, 50]
# lim_y=[-15,15]
# lim_z=[-5,5]
# pixel_lim=8


lim_x = [2.5, 75]
lim_y = [-15, 15]
lim_z = [-3.5, 1]
height, width = 772, 1024
pixel_lim = 5

##############################################################################################

class realCoor():
    
    def __init__(self):
        print("Running the new_lineto3d transformation script\n")

        self.left_pcl_pub = rospy.Publisher("/Left_Line3dPoints", PointCloud2, queue_size=1)
        print("Publishing /Left_Line3dPoints\n")
        self.right_pcl_pub = rospy.Publisher("/Right_Line3dPoints", PointCloud2, queue_size=1)
        print("Publishing /Right_Line3dPoints\n")

        self.fields = [pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='y', offset=4,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='z', offset=8,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='intensity', offset=12,datatype=pc2.PointField.FLOAT32, count=1)]

        self.pcdSub=message_filters.Subscriber("/lidar_tc/velodyne_points", PointCloud2)#/kitti/velo/pointcloud
        self.lane_pointcloud = PointCloud2() #rospy.subscriber
        self.used_pointcloud = PointCloud2() #rospy.subscriber
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'lidar_tc'
        pointcloud=[]
        
        self.left_pointSub = message_filters.Subscriber("/lane_detection/current_lane_left_boundary", LanePoints)
        self.right_pointSub = message_filters.Subscriber("/lane_detection/current_lane_right_boundary", LanePoints)

        ts = message_filters.ApproximateTimeSynchronizer([self.pcdSub, self.left_pointSub, self.right_pointSub], 10, 0.4, allow_headerless=True)
        ts.registerCallback(self.pcd_callback)

        self.vis=True
        rospy.spin()
    
    def create_cloud(self, line_3d, which):
        #self.header = std_msgs.msg.Header()
        #self.header.frame_id = 'lidar_tc'
        #self.header.stamp = rospy.Time.now()
        
        if which == 'left':
            self.left_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.left_pcl_pub.publish(self.left_pointcloud)
        elif which == 'right':
            self.right_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.right_pcl_pub.publish(self.right_pointcloud)

    def pcd_callback(self, msgLidar, msgLeftPoint, msgRightPoint):
        print("in Callback") # This code never prints this in console which means it never goes into the callback
        self.header = msgLidar.header
        #start_time_pcd = rospy.Time.now().to_sec()
        # print('in callback')
        if msgLeftPoint.points!=[] or msgRightPoint.points!=[]:

            pc = ros_numpy.numpify(msgLidar)
            points=np.zeros((pc.shape[0],4))
            points[:,0]=pc['x']
            points[:,1]=pc['y']
            points[:,2]=pc['z']
            points[:,3]=1

            pc_arr=self.crop_pointcloud(points) #to reduce computational expense
            pc_arr_pick=np.transpose(pc_arr)       

            m1=np.matmul(T_vel_cam,pc_arr_pick)#4*N
            #m2= np.matmul(R0_rect,m1) 
            uv1= np.matmul(rect,m1) #4*N        

            uv1[0,:]=  np.divide(uv1[0,:],uv1[2,:])
            uv1[1,:]=  np.divide(uv1[1,:],uv1[2,:])
            #-
            # m1 = torch.matmul(torch.tensor(T_vel_cam), torch.tensor(pc_arr_pick))
            # uv1 = torch.matmul(torch.tensor(rect), m1)
            # uv1[:2, :] /= uv1[2, :]
            # u, v = uv1[0, :].numpy(), uv1[1, :].numpy()

            self.process_points(msgLeftPoint, uv1, pc_arr_pick, 'left')
            self.process_points(msgRightPoint, uv1, pc_arr_pick, 'right')

    def process_points(self, msgPoint, uv1, pc_arr_pick, which):
            line_3d=[]

            u=uv1[0,:]
            v=uv1[1,:]
            for point in msgPoint.points:
                idx = np.where(((u+pixel_lim>= point.x) & (u-pixel_lim<=point.x)) & ((v+pixel_lim>=point.y) & (v-pixel_lim<=point.y)))
                idx = np.array(idx)
                #print(idx)
                if idx.size > 0:
                    for i in range(idx.size):
                        # print(point.lane_id)
                        line_3d.append(([(pc_arr_pick[0][idx[0, i]]), (pc_arr_pick[1][idx[0, i]]), (pc_arr_pick[2][idx[0, i]]), 1]))

            if self.vis == True and line_3d!=[] and len(line_3d) > 2:
                # line_3d = self.test(line_3d)
                line_3d = np.array(line_3d)
                _, idx = np.unique(line_3d[:, 0:2], axis=0, return_index=True)
                line_3d_unique = line_3d[idx]
                self.create_cloud(line_3d_unique, which)
     
    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >=lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
        pointcloud = pointcloud[mask]
        return pointcloud
        
    def test(self, line):
        output = []
        line = np.array(line)
        indd=np.lexsort((line[:,2],line[:,1],line[:,0]))
        line=line[indd]
        _, idx=np.unique(line[:,0:2], axis=0, return_index=True)
        line =line[idx]
        #print("line::", line)
        x = -np.sort(-line[:,0])
        y =  -np.sort(-line[:,1])
        z =  -np.sort(-line[:,2])
        # print("xyz", x,y,z)
        intensity = line[:,3]
        npts = len(x)
        s = np.zeros(npts, dtype=float)
        
        xl=np.linspace(np.amax(x), np.amin(x), 500)
        yl=np.linspace(np.amax(y), np.amin(y), 500)
        zl=np.linspace(np.amax(z), np.amin(z),500)

        # Create new interpolation function for each axis against the norm 
        data = np.concatenate((x[:, np.newaxis], y[:, np.newaxis],  z[:, np.newaxis]),  axis=1)

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)

        #  SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)

        linepts = vv[0] * np.mgrid[-20:20:2j][:, np.newaxis]

        linepts += datamean

        l=linepts[0,0]-linepts[1,0]
        m=linepts[0,1]-linepts[1,1]
        n=linepts[0,2]-linepts[1,2]

        t= (xl-linepts[0,0])/l
        xs=xl
        ys=(t)*m+linepts[0,1]
        zs=(t)*n+linepts[0,2]
        
        return np.transpose([xs, ys, zs,np.ones(len(xs)) * 1.1])


if __name__=='__main__':
    rospy.init_node("LaneTO3D")
    realCoor()
