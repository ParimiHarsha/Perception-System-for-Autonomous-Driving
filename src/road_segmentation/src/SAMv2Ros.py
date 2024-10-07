import os

import cv2
import numpy as np

np.float = np.float64
import time

import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image
from std_msgs.msg import MultiArrayDimension
from scipy import stats
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN


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

# model = SAM("sam2_b.pt")

def fit_polynomial_to_boundary(boundary_points, degree=3):
    """Fit a polynomial curve to a set of boundary points and return smoothed points."""
    if len(boundary_points) < 2:
        return boundary_points  # Return original if not enough points for fitting

    # Separate x and y coordinates
    x = boundary_points[:, 0]
    y = boundary_points[:, 1]

    # Fit a polynomial curve of the specified degree
    polynomial_coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(polynomial_coefficients)

    # Generate smooth x values for the curve
    x_new = np.linspace(x.min(), x.max(), 100)  # Adjust number of points as needed
    y_smooth = polynomial(x_new)

    # Combine the x and smoothed y values
    smoothed_boundary = np.vstack((x_new, y_smooth)).T

    return smoothed_boundary.astype(np.int32)


def remove_boundary_outliers_ransac(points, direction='left'):
    """
    Remove outliers from a set of points using the RANSAC method.
    This function ensures that points classified in 'left' or 'right' boundaries do not cross over.

    Args:
        points (np.ndarray): Points in the format [x, y].
        direction (str): 'left' or 'right', indicating expected boundary position.
        
    Returns:
        np.ndarray: Points after removing outliers.
    """
    if len(points) < 3:  # Not enough points to detect outliers
        return points

    # Convert the list of points to a numpy array
    points = np.array(points)

    # Extract x and y coordinates
    x = points[:, 0].reshape(-1, 1)  # Independent variable for RANSAC
    y = points[:, 1]  # Dependent variable for RANSAC

    # Fit a RANSAC regressor model
    ransac = RANSACRegressor()
    ransac.fit(x, y)

    # Use the RANSAC inlier mask to filter out outliers
    inlier_mask = ransac.inlier_mask_

    # Return only the inlier points
    return points[inlier_mask]


def remove_boundary_outliers_dbscan(points, eps=1, min_samples=200):
    """
    Remove outliers from a set of points using the DBSCAN clustering algorithm.
    
    Args:
        points (np.ndarray): Points in the format [x, y].
        eps (float): The maximum distance between points to be considered in the same neighborhood.
        min_samples (int): The number of points required to form a dense region.
        
    Returns:
        np.ndarray: Points after removing outliers using DBSCAN.
    """
    if len(points) < min_samples:  # Not enough points to form a cluster
        return points

    points = np.array(points)
    # Apply DBSCAN clustering to identify outliers
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = dbscan.fit_predict(points)
    
    # Filter out points labeled as noise (-1 label in DBSCAN)
    valid_mask = labels != -1  # Points not labeled as noise

    # Ensure valid_mask is a boolean array
    valid_mask = np.asarray(valid_mask, dtype=bool)

    return points[valid_mask]


def classify_boundaries_using_horizontal_bins(contour_points, frame_height, num_bins=50):
    """
    Classify left and right boundaries using horizontal bins.

    Args:
        contour_points (np.ndarray): Contour points of the detected road area.
        frame_height (int): The height of the image frame.
        num_bins (int): Number of horizontal bins to create.

    Returns:
        left_boundary_points (np.ndarray): Points classified as the left boundary.
        right_boundary_points (np.ndarray): Points classified as the right boundary.
    """
    if len(contour_points) == 0:
        return np.array([]), np.array([])
    contour_points = contour_points[contour_points[:,1]<frame_height]
    # Determine the minimum y-coordinate of the contour (top-most point of the road region)
    y_min = contour_points[:, 1].min()

    # Calculate the vertical range for binning (from y_min to bottom of the frame)
    binning_range = frame_height - y_min
    bin_height = binning_range // num_bins

    left_boundary_points = []
    right_boundary_points = []

    # Create horizontal bins from y_min to the bottom of the frame
    for i in range(num_bins):
        y_lower_limit = y_min + i * bin_height
        y_upper_limit = y_min + (i + 1) * bin_height

        # Get points within the current horizontal bin
        bin_points = contour_points[(contour_points[:, 1] > y_lower_limit) & (contour_points[:, 1] <= y_upper_limit)]

        if len(bin_points) > 0:
            # Classify points based on x-coordinate
            for point in bin_points:
                x, y = point
                if x < (bin_points[:, 0].mean()):  # Using the mean x-coordinate as the reference
                    left_boundary_points.append(point)
                else:
                    right_boundary_points.append(point)

    return left_boundary_points, right_boundary_points


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

    # Calculate the center line on the Y-axis
    center_x = w_resized * 0.75

    point_coords = np.array([[300, 700], [500, 700], [700, 700]])
    point_coords = torch.tensor(
        [
            [int(p[0] * w_resized / w_orig), int(p[1] * h_resized / h_orig)]
            for p in point_coords
        ],
        dtype=torch.int32,
        device=device,
    )
    input_labels = torch.tensor([1, 1, 1], device=device)

    # Predict masks
    start_step = time.time()
    predictor.set_image(image_resized_tensor.cpu().numpy())
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.cpu().numpy(),
        point_labels=input_labels.cpu().numpy(),
        multimask_output=False,
    )

    # result = model(image_resized_tensor.cpu().numpy(), points=point_coords, labels=input_labels)
    # ultralytics_mask = result[0].masks.data.cpu().numpy()  # Convert to numpy array
    # ultralytics_mask = ultralytics_mask_np)  # Convert to tensor

    # ultralytics_mask = torch.from_numpy(result[0].masks).to(device)
    # print('result', result)
    rospy.loginfo(f"Mask prediction took: {time.time() - start_step} seconds.")
    rospy.loginfo(f"Masks predicted with scores: {scores}.")

    # Create binary mask
    start_step = time.time()
    masks_tensor = torch.from_numpy(masks).to(device)
    # masks_tensor = torch.from_numpy(ultralytics_mask).to(device)
    binary_mask = (masks_tensor.sum(dim=0) > 0).to(torch.uint8)
    binary_mask_np = binary_mask.cpu().numpy()

    # Find contours in the clustered mask
    contours, _ = cv2.findContours(
        binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
 
    MIN_CONTOUR_AREA = 30000.0  # Set a minimum area threshold (adjust as needed)
    # Set the minimum y-coordinate threshold for removing points at the bottom
    MIN_Y_COORD = int(0.98 * h_resized)  # Keep points above 90% of the image height

    # Use only the largest contour (representing the road)
    road_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(road_contour)<MIN_CONTOUR_AREA:
        print('The contour area is below the threshold')
        return None, None, None
    
    # Filter out points that are too close to the bottom of the frame
    road_contour = [
        point for point in road_contour if point[0][1] < MIN_Y_COORD
    ]
    if len(road_contour) < 2:  # If the filtered contour is too small
        rospy.logwarn("Filtered road contour is too small after removing bottom points.")
        return None, None, None
    
    contour_points = np.array(road_contour).reshape(-1, 2)

    # Identify the horizon point (the point where left and right meet)
    # For simplicity, let's assume it's the highest point in the contour
    horizon_index = np.argmin(
        contour_points[:, 1]
    )  # Find the point with the maximum y value
    horizon_point = contour_points[horizon_index]

    left_boundary_points = []
    right_boundary_points = []


    # Classify left and right boundary points using horizontal bins
    left_boundary_points, right_boundary_points = classify_boundaries_using_horizontal_bins(contour_points[contour_points[:, 1] < center_x], h_resized)

    # Classify points based on the horizon point
    for point in contour_points[contour_points[:, 1] >= center_x]:
        if point[0] < horizon_point[0]:  # Left of the horizon point
            left_boundary_points.append(point)
        else:  # Right of the horizon point
            right_boundary_points.append(point)
    print('Left boundary points', left_boundary_points)
    # Remove outliers
    # left_boundary_points = remove_boundary_outliers(left_boundary_points, direction='left')
    # right_boundary_points = remove_boundary_outliers(right_boundary_points, direction='right')
    # Remove outliers from left and right boundary points using RANSAC
    # left_boundary_points = remove_boundary_outliers_ransac(left_boundary_points, direction='left')
    # right_boundary_points = remove_boundary_outliers_ransac(right_boundary_points, direction='right')

    #Using DBSCAN
    # left_boundary_points = remove_boundary_outliers_dbscan(left_boundary_points, eps=2, min_samples=5)
    # right_boundary_points = remove_boundary_outliers_dbscan(right_boundary_points, eps=2, min_samples=5)


    # Convert to numpy arrays for spline fitting
    left_boundary_points = np.array(left_boundary_points)
    right_boundary_points = np.array(right_boundary_points)

    # Apply spline fitting to smooth the boundaries
    # Apply polynomial fitting to smooth the boundaries
    # smoothed_left_boundary = fit_polynomial_to_boundary(left_boundary_points, degree=3)
    # smoothed_right_boundary = fit_polynomial_to_boundary(right_boundary_points, degree=3)

    # Convert left and right boundary points into contours format
    left_boundary_contour = (
        np.array(left_boundary_points).reshape((-1, 1, 2)).astype(np.int32)
    )
    right_boundary_contour = (
        np.array(right_boundary_points).reshape((-1, 1, 2)).astype(np.int32)
    )

    # Prepare to draw the polynomial curve on the overlay
    # x_curve = np.linspace(sampled_points[:, 0].min(), sampled_points[:, 0].max(), 100)
    # y_curve = polynomial(x_curve)


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
                overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2
            )  # Green for left
        for point in right_boundary_points:
            cv2.circle(
                overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2
            )  # Blue for right

        # Draw horizontal lines to visualize bins
        num_bins = 50
        # Draw horizontal lines to visualize bins
        # Determine the height of the image
        y_min = contour_points[:, 1].min()
        binning_range = h_resized - y_min

        bin_height = binning_range // num_bins

        # for i in range(num_bins):
        #     y_position = (i + 1) * bin_height
        #     cv2.line(overlay, (0, y_min+y_position), (w_resized, y_min+y_position), (255, 255, 0), 1)  # Yellow lines
        #     y_position = int(y_min + i * bin_height)  # Calculate y position for each line
        #     cv2.line(overlay, (0, y_position), (w_resized, y_position), (255, 255, 0), 1)  # Yellow lines
            # Draw the bin index text
        #     cv2.putText(overlay, f"Bin {i+1}", (5, y_position - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.line(overlay, (0, int(center_x)), (w_resized, int(center_x)), (255, 255, 0), 3) 
    return overlay, left_boundary_points, right_boundary_points


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
