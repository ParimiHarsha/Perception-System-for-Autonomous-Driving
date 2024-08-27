import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray


def callback(bounding_box_array):
    marker_array = MarkerArray()
    rospy.loginfo(
        "Received BoundingBoxArray with {} boxes".format(len(bounding_box_array.boxes))
    )
    for i, bbox in enumerate(bounding_box_array.boxes):
        marker = Marker()
        marker.header = bbox.header
        marker.ns = "fused_bounding_boxes"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = bbox.pose.position.x
        marker.pose.position.y = bbox.pose.position.y
        marker.pose.position.z = bbox.pose.position.z
        marker.pose.orientation.x = bbox.pose.orientation.x
        marker.pose.orientation.y = bbox.pose.orientation.y
        marker.pose.orientation.z = bbox.pose.orientation.z
        marker.pose.orientation.w = bbox.pose.orientation.w
        marker.scale.x = bbox.dimensions.x
        marker.scale.y = bbox.dimensions.y
        marker.scale.z = bbox.dimensions.z
        marker.color.a = 0.5  # Set transparency
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker_array.markers.append(marker)
        rospy.loginfo("Added marker {} with ID {}".format(i, marker.id))
    marker_pub.publish(marker_array)
    rospy.loginfo(
        "Published MarkerArray with {} markers".format(len(marker_array.markers))
    )


if __name__ == "__main__":
    rospy.init_node("bounding_box_to_marker")
    marker_pub = rospy.Publisher("/fused_bbox_marker", MarkerArray, queue_size=10)
    rospy.Subscriber(
        "/fused_bbox", BoundingBoxArray, callback
    )  # Replace with your actual topic and message type
    rospy.loginfo("Node initialized and subscriber set up")
    rospy.spin()
    rospy.loginfo("Shutting down node")
