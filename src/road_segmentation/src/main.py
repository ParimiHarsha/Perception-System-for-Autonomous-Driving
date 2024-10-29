# main.py

from SAMv2Ros import RoadSegmentation
import rospy
#import torch

from road_segmentation_3d import RoadSegmentation3D

# if __name__ == "__main__":
#     segmentation = RoadSegmentation3D()

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('road_segmentation_node',anonymous=False)
    road_segmenter = RoadSegmentation()
    # Keep the node running until interrupted
    rospy.spin()