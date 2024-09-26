import os
import sys

import cv2
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

np.float = np.float64
import time

import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull
from sensor_msgs.msg import Image
from sklearn.decomposition import PCA
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from road_segmentation.msg import DetectedRoadArea

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model
# Set local path to checkpoints directory
base_path = os.path.dirname(
    os.path.abspath(__file__)
)  # Get the directory of the current script
checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")

sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def process_image(image, publish_image=False, num_clusters=1):
    rospy.loginfo("Starting image processing function.")

    # Convert the image to a torch tensor and move it to the device
    start_step = time.time()
    image_tensor = torch.from_numpy(image).to(device)
    h_orig, w_orig, _ = image_tensor.shape
    image_resized = cv2.resize(image, (w_orig, h_orig))
    image_resized_tensor = torch.from_numpy(image_resized).to(device)
    rospy.loginfo(
        f"Image conversion and resizing took: {time.time() - start_step} seconds."
    )
    h_resized, w_resized, _ = image_resized_tensor.shape

    # Adjust point coordinates based on resizing
    point_coords = np.array([[300, 700], [500, 700], [600, 500], [700, 700]])
    point_coords = torch.tensor(
        [
            [int(p[0] * w_resized / w_orig), int(p[1] * h_resized / h_orig)]
            for p in point_coords
        ],
        dtype=torch.int32,
        device=device,
    )
    input_labels = torch.tensor([1, 1, 1, 1], device=device)

    # Predict masks
    start_step = time.time()
    predictor.set_image(image_resized_tensor.cpu().numpy())
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.cpu().numpy(),
        point_labels=input_labels.cpu().numpy(),
        multimask_output=True,
    )
    rospy.loginfo(f"Mask prediction took: {time.time() - start_step} seconds.")
    rospy.loginfo(f"Masks predicted with scores: {scores}.")

    # Create binary mask
    start_step = time.time()
    masks_tensor = torch.from_numpy(masks).to(device)
    binary_mask = (masks_tensor.sum(dim=0) > 0).to(torch.uint8)
    binary_mask_np = binary_mask.cpu().numpy()

    # Find contours in the clustered mask
    contours, _ = cv2.findContours(
        binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = []
    MIN_CONTOUR_AREA = 30000.0  # Set a minimum area threshold (adjust as needed)
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    # Optionally, you can print the maximum and minimum areas to get an idea of the range
    if areas:
        for area in areas:
            if area > MIN_CONTOUR_AREA:
                print("GREAT AREA", area)

    # If there's no contour found, return an empty result
    if not contours:
        rospy.logwarn("No contours found in the binary mask.")
        return None, None, None

    # Use only the largest contour (representing the road)
    road_contour = max(contours, key=cv2.contourArea)

    # left_boundary_points, right_boundary_points = classify_boundaries(road_contour)
    contour_points = np.array(road_contour).reshape(-1, 2)

    # Identify the horizon point (the point where left and right meet)
    # For simplicity, let's assume it's the highest point in the contour
    horizon_index = np.argmin(
        contour_points[:, 1]
    )  # Find the point with the maximum y value
    horizon_point = contour_points[horizon_index]

    left_boundary_points = []
    right_boundary_points = []

    # Classify points based on the horizon point
    for point in contour_points:
        if point[0] < horizon_point[0]:  # Left of the horizon point
            left_boundary_points.append(point)
        else:  # Right of the horizon point
            right_boundary_points.append(point)

    # Convert left and right boundary points into contours format
    left_boundary_contour = (
        np.array(left_boundary_points).reshape((-1, 1, 2)).astype(np.int32)
    )
    right_boundary_contour = (
        np.array(right_boundary_points).reshape((-1, 1, 2)).astype(np.int32)
    )

    # Create contour overlay image if publishing is enabled
    overlay = None
    if publish_image:
        start_step = time.time()
        contour_overlay = image_resized_tensor.clone().cpu().numpy()

        # Prepare the mask for overlay
        mask = (
            masks_tensor.sum(dim=0)
            .cpu()
            .numpy()
            .reshape(h_resized, w_resized, 1)
            .repeat(3, axis=2)
        )
        overlay = cv2.addWeighted(
            contour_overlay.astype(np.uint8),
            0.8,
            (mask * np.array([0, 255, 0], dtype=np.uint8)).astype(np.uint8),
            0.2,
            0,
        )

        # Draw points on the overlay
        for point, label in zip(point_coords.cpu().numpy(), input_labels.cpu().numpy()):
            cv2.circle(
                overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1
            )
        for point in left_boundary_points:
            cv2.circle(
                overlay, tuple(point), radius=5, color=(0, 255, 0), thickness=-2
            )  # Green for left
        for point in right_boundary_points:
            cv2.circle(
                overlay, tuple(point), radius=5, color=(0, 0, 255), thickness=-2
            )  # Blue for right

    return overlay, left_boundary_contour, right_boundary_contour


class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        self.image_pub = rospy.Publisher("/road_segmentation", Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.image_callback
        )
        self.left_boundary_pub = rospy.Publisher(
            "/left_boundary", DetectedRoadArea, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            "/right_boundary", DetectedRoadArea, queue_size=1
        )
        self.publish_image = True  # Boolean flag for publishing the image

    def image_callback(self, ros_image):
        rospy.loginfo("Image received, processing...")
        img = ros_numpy.numpify(ros_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay, left_boundary, right_boundary = process_image(img, self.publish_image)

        # Publish processed overlay image if the flag is set
        if self.publish_image:
            rospy.loginfo("Publishing overlay image.")
            msg = Image()
            msg.header.stamp = ros_image.header.stamp
            msg.height, msg.width = overlay.shape[:2]
            msg.encoding = "rgb8"
            msg.data = np.array(overlay).tobytes()
            self.image_pub.publish(msg)

        # Publish left boundary points
        rospy.loginfo("Publishing left boundary points.")
        left_boundary_msg = DetectedRoadArea()
        left_boundary_msg.header.stamp = ros_image.header.stamp
        left_boundary_msg.RoadArea.data = [
            float(point) for point in np.array(left_boundary).flatten()
        ]
        left_boundary_msg.RoadArea.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=len(left_boundary) // 2,
                stride=len(left_boundary),
            ),
            MultiArrayDimension(label="width", size=2, stride=2),
        ]
        self.left_boundary_pub.publish(left_boundary_msg)

        # Publish right boundary points
        rospy.loginfo("Publishing right boundary points.")
        right_boundary_msg = DetectedRoadArea()
        right_boundary_msg.header.stamp = ros_image.header.stamp
        right_boundary_msg.RoadArea.data = [
            float(point) for point in np.array(right_boundary).flatten()
        ]
        right_boundary_msg.RoadArea.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=len(right_boundary) // 2,
                stride=len(right_boundary),
            ),
            MultiArrayDimension(label="width", size=2, stride=2),
        ]
        self.right_boundary_pub.publish(right_boundary_msg)


if __name__ == "__main__":
    rospy.init_node("road_segmentation_node", anonymous=False)
    rospy.loginfo("Road segmentation node started.")
    road_segmentation = RoadSegmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down road segmentation node.")
