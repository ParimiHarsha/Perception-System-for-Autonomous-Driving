#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import cython 
np.float = np.float64
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image, CompressedImage
from road_segmentation.msg import DetectedRoadArea
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model and move constants to device
base_path = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")
sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_small.pt")
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
#sam2_model.eval()  # Set to evaluation mode
predictor = SAM2ImagePredictor(sam2_model)

# Convert constants to torch tensors on device
point_coords = torch.tensor([[550, 700], [650, 700], [750, 700]], device=device)
input_labels = torch.tensor([1, 1, 1], device=device)
MIN_CONTOUR_AREA = 30000.0

@torch.no_grad()  # Key optimization: disable gradient computation
def process_image(image, publish_image=False):
    h_original, w_original = image.shape[:2]
    center_x = int(w_original * 0.75)

    # Predict masks using SAM2 model with automatic mixed precision
    with torch.cuda.amp.autocast():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=input_labels,
            multimask_output=False
        )
    binary_mask_np = (masks[0] > 0).astype(np.uint8)
    
    # Use more efficient contour approximation
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        rospy.logwarn("No contours found.")
        return None, None, None

    road_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(road_contour) < MIN_CONTOUR_AREA:
        rospy.logwarn("The contour area is below the threshold.")
        return None, None, None

    # Filter road contour points
    MIN_Y_COORD = int(0.98 * h_original)
    road_contour = np.array([point for point in road_contour if point[0][1] < MIN_Y_COORD])

    if len(road_contour) < 2:
        rospy.logwarn(
            "Filtered road contour is too small after removing bottom points."
        )
        return None, None, None

    contour_points = road_contour.reshape(-1, 2)
    horizon_point = contour_points[np.argmin(contour_points[:, 1])]

    # Use numpy operations for faster point classification
    upper_mask = contour_points[:, 1] < center_x
    upper_points = contour_points[upper_mask]
    
    left_boundary_points, right_boundary_points = classify_boundaries_using_horizontal_bins(
        upper_points, h_original
    )

    # Classify remaining points efficiently
    lower_points = contour_points[~upper_mask]
    lower_mask = lower_points[:, 0] < horizon_point[0]
    
    if len(left_boundary_points) > 0:
        left_boundary_points = np.vstack([left_boundary_points, lower_points[lower_mask]])
    else:
        left_boundary_points = lower_points[lower_mask]
        
    if len(right_boundary_points) > 0:
        right_boundary_points = np.vstack([right_boundary_points, lower_points[~lower_mask]])
    else:
        right_boundary_points = lower_points[~lower_mask]

    overlay = None
    if publish_image:
        overlay = create_overlay(
            image,
            binary_mask_np,
            left_boundary_points,
            right_boundary_points,
        )

    return overlay, left_boundary_points, right_boundary_points

@timer
def classify_boundaries_using_horizontal_bins(contour_points, frame_height, num_bins=50):
    if len(contour_points) == 0:
        return np.array([]), np.array([])

    y_min = contour_points[:, 1].min()
    bin_height = (frame_height - y_min) // num_bins
    
    # Use numpy operations for faster binning
    bin_indices = ((contour_points[:, 1] - y_min) // bin_height).astype(int)
    bins = np.unique(bin_indices)
    
    left_points = []
    right_points = []
    
    for bin_idx in bins:
        bin_mask = bin_indices == bin_idx
        bin_points = contour_points[bin_mask]
        if len(bin_points) > 0:
            mean_x = bin_points[:, 0].mean()
            left_mask = bin_points[:, 0] < mean_x
            left_points.append(bin_points[left_mask])
            right_points.append(bin_points[~left_mask])
    
    return (np.vstack(left_points) if left_points else np.array([]),
            np.vstack(right_points) if right_points else np.array([]))

def create_overlay(image, binary_mask_np, left_boundary_points, right_boundary_points):
    overlay = image
    mask = binary_mask_np.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
    overlay = cv2.addWeighted(
        overlay,
        0.8,
        mask * np.array([0, 255, 0], dtype=np.uint8),
        0.2,
        0,
    )

    # Draw points
    point_coords_np = point_coords.cpu().numpy()
    for point in point_coords_np:
        cv2.circle(overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1)
    for point in left_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2)
    for point in right_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2)

    return overlay

class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        self.image_pub = rospy.Publisher("/road_segmentation", Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "resized/camera_fl/image_color",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**26,
        )
        self.left_boundary_pub = rospy.Publisher(
            "/left_boundary", DetectedRoadArea, queue_size=1
        )
        self.right_boundary_pub = rospy.Publisher(
            "/right_boundary", DetectedRoadArea, queue_size=1
        )
        self.publish_image = True

    @timer
    def image_callback(self, ros_image):
        img = ros_numpy.numpify(ros_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay, left_boundary, right_boundary = process_image(img, self.publish_image)

        if self.publish_image and overlay is not None:
            msg = Image()
            msg.header.stamp = ros_image.header.stamp
            msg.height, msg.width = overlay.shape[:2]
            msg.encoding = "rgb8"
            msg.data = overlay.tobytes()
            self.image_pub.publish(msg)

        # Publish boundary points
        if left_boundary is not None:
            left_msg = DetectedRoadArea()
            left_msg.header.stamp = ros_image.header.stamp
            left_msg.RoadArea.data = left_boundary.flatten().tolist()
            self.left_boundary_pub.publish(left_msg)

        if right_boundary is not None:
            right_msg = DetectedRoadArea()
            right_msg.header.stamp = ros_image.header.stamp
            right_msg.RoadArea.data = right_boundary.flatten().tolist()
            self.right_boundary_pub.publish(right_msg)

if __name__ == "__main__":
    rospy.init_node("road_segmentation_node", anonymous=False)

    road_seg = RoadSegmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down road segmentation node.")