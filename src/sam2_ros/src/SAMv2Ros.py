#!/usr/bin/env python3
import os
import cv2
import numpy as np

np.float = np.float64
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2 #src/sam2_ros/src/segment-anything-2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image, CompressedImage
from sam2_ros.msg import DetectedRoadArea
from functools import wraps
from yolov9_ros.msg import BboxCentersClass


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = rospy.Time.now().to_sec()  # time.time()
        result = func(*args, **kwargs)
        end_time = rospy.Time.now().to_sec()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model
base_path = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")
# sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_small.pt")
sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Define points for initial segmentation prompt
point_coords = np.array([[450, 700], [550, 700], [650, 700]])
input_labels = [1, 1, 1]
MIN_CONTOUR_AREA = 30000.0


def process_image(image, publish_image=False):
    h_original, w_original = image.shape[:2]
    center_x = int(w_original * 0.75)

    # Predict masks using SAM2 model
    with torch.cuda.amp.autocast():  ###@torch.no_grad():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=input_labels, multimask_output=False
        )
    binary_mask_np = (masks[0] > 0).astype(np.uint8)
    kernel = np.ones((6, 6), np.uint8)
    dilated_mask = cv2.dilate(binary_mask_np, kernel, iterations=1)
    # Fill holes in the mask
    filled_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(filled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        rospy.logwarn("No contours found.")
        return None, None, None

    road_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(road_contour) < MIN_CONTOUR_AREA:
        rospy.logwarn("The contour area is below the threshold.")
        return None, None, None

    # Filter road contour points
    MIN_Y_COORD = int(0.8 * h_original)
    road_contour = [point for point in road_contour if point[0][1] < MIN_Y_COORD]

    if len(road_contour) < 2:
        rospy.logwarn(
            "Filtered road contour is too small after removing bottom points."
        )
        return None, None, None

    contour_points = np.array(road_contour).reshape(-1, 2)
    horizon_point = contour_points[np.argmin(contour_points[:, 1])]

    left_boundary_points, right_boundary_points = (
        classify_boundaries_using_horizontal_bins(
            contour_points[contour_points[:, 1] < center_x], h_original
        )
    )

    for point in contour_points[contour_points[:, 1] >= center_x]:
        (
            left_boundary_points
            if point[0] < horizon_point[0]
            else right_boundary_points
        ).append(point)

    overlay = create_overlay(
        image,
        binary_mask_np,
        left_boundary_points,
        right_boundary_points,
        publish_image,
    )

    return overlay, left_boundary_points, right_boundary_points


def classify_boundaries_using_horizontal_bins(
    contour_points, frame_height, num_bins=50
):
    if len(contour_points) == 0:
        return np.array([]), np.array([])

    y_min = contour_points[:, 1].min()
    bin_height = (frame_height - y_min) // num_bins
    left_boundary_points, right_boundary_points = [], []

    for i in range(num_bins):
        y_lower_limit = y_min + i * bin_height
        y_upper_limit = y_min + (i + 1) * bin_height

        bin_points = contour_points[
            (contour_points[:, 1] > y_lower_limit)
            & (contour_points[:, 1] <= y_upper_limit)
        ]

        if len(bin_points) > 0:
            mean_x = bin_points[:, 0].mean()
            for point in bin_points:
                (
                    left_boundary_points if point[0] < mean_x else right_boundary_points
                ).append(point)

    return np.array(left_boundary_points), np.array(right_boundary_points)


def create_overlay(
    image, binary_mask_np, left_boundary_points, right_boundary_points, publish_image
):
    if not publish_image:
        return None

    overlay = image.copy()
    mask = binary_mask_np.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
    overlay = cv2.addWeighted(
        overlay.astype(np.uint8),
        0.8,
        mask * np.array([0, 255, 0], dtype=np.uint8),
        0.2,
        0,
    )

    for point in point_coords:
        cv2.circle(overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1)
    for point in left_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2)
    for point in right_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2)

    return overlay


class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        # self.image_pub = rospy.Publisher("/road_segmentation/compressed", CompressedImage, queue_size=1)
        self.image_pub = rospy.Publisher("/road_segmentation", Image, queue_size=1)

        self.image_sub = rospy.Subscriber(
            # "resized/camera_fl/image_color/compressed",CompressedImage,
            "resized/camera_fl/image_color",
            Image,
            # self.image_callback,
            self.callback,
            queue_size=1,
            # buff_size=2**26,
            # tcp_nodelay=True
        )

        self.left_boundary_pub = rospy.Publisher(
            "/left_boundary", DetectedRoadArea, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            "/right_boundary", DetectedRoadArea, queue_size=1
        )
        self.yolo_sub = rospy.Subscriber(
            "/yolo_detection_node/bboxInfo", BboxCentersClass,self.yolo_callback, queue_size=1
        )
        self.ros_image = None
        self.publish_image = True
        rospy.Timer(rospy.Duration(0.1), self.process_loop)

    @timer
    def callback(self, ros_image):
        self.ros_image = ros_image

    def yolo_callback(self, data):
        """
        Callback for YOLO detection results.
        Extracts bounding boxes from the YOLO detection messages.
        """
        # self.detected_objects = [
        #     (box.xmin, box.ymin, box.xmax, box.ymax) for box in data.bounding_boxes
        # ]
        self.detected_objects = [point for point in data.Bboxes]
        rospy.loginfo(f"Updated YOLO detections: {self.detected_objects}")

    def objects_on_road(objects, road_mask):
        """
        Check if any detected objects overlap with the road mask.
        :param objects: List of bounding boxes [(x_min, y_min, x_max, y_max)].
        :param road_mask: Binary mask of the road (numpy array).
        :return: True if any objects overlap with the road mask.
        """
        for (x_min, y_min, x_max, y_max) in objects:
            # Extract the region of interest from the mask
            roi = road_mask[y_min:y_max, x_min:x_max]
            if np.any(roi > 0):  # Check if there's overlap
                return True
        return False

    @timer
    def process_loop(self, event):
        """Process the image if available, called periodically by a ROS Timer."""
        if self.ros_image:
            self.image_callback()

    @timer
    def image_callback(self):
        print(
            "time delay",
            self.ros_image.header.stamp.to_sec() - rospy.Time.now().to_sec(),
        )
        img = ros_numpy.numpify(self.ros_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.fromstring(ros_image.data,np.uint8)
        # img = cv2.imdecode(img, cv2.COLOR_BGR2RGB)
        overlay, left_boundary, right_boundary = process_image(img, self.publish_image)

        if self.publish_image and overlay is not None:
            self.publish_image_topic(self.ros_image, overlay)

        # Publish boundary points
        if left_boundary is not None:
            self.publish_boundary(
                left_boundary, self.left_boundary_pub, self.ros_image.header.stamp
            )

        if right_boundary is not None:
            self.publish_boundary(
                right_boundary, self.right_boundary_pub, self.ros_image.header.stamp
            )

    def publish_image_topic(self, ros_image, overlay):

        msg = Image()
        msg.header.stamp = ros_image.header.stamp
        msg.height, msg.width = overlay.shape[:2]
        msg.encoding = "rgb8"
        # msg.format = "jpeg"
        msg.data = overlay.tobytes()
        # msg.data = np.array(cv2.imencode('.jpg', overlay)[1]).tostring()
        self.image_pub.publish(msg)

    def publish_boundary(self, boundary, publisher, stamp):
        boundary_msg = DetectedRoadArea()
        boundary_msg.header.stamp = stamp
        boundary_msg.RoadArea.data = [
            float(point) for point in boundary.flatten()
        ]  # not sure if for point in boundary flatten is neccesary
        publisher.publish(boundary_msg)


if __name__ == "__main__":
    rospy.init_node("road_segmentation_node", anonymous=False)

    RoadSegmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down road segmentation node.")