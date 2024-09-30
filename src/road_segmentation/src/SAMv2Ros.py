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

def classify_boundaries_using_polynomial_fit(contour_points, bottom_center, sample_points=10):
    """
    Classify left and right boundaries using polynomial fitting.
    
    Args:
        contour_points (np.ndarray): Contour points of the detected road area.
        sample_points (int): Number of points to sample along the road for fitting.

    Returns:
        left_boundary_points (np.ndarray): Points classified as the left boundary.
        right_boundary_points (np.ndarray): Points classified as the right boundary.
    """
    # Identify key points for polynomial fitting
    # h, w = contour_points[:, 1].max()//2 + 1, contour_points[:, 0].max() + 1  # height and width
    h, w = contour_points[:, 1].max() + 1, contour_points[:, 0].max() + 1  # height and width
    bottom_limit = int(2 * h / 3)  # Limit for sampling points (2/3 from the bottom)

    # bottom_center = (w // 2, h - 1)  # Bottom center of the frame
    
    # Sample points along the road contour
    sampled_points = [bottom_center]

    # Sample additional points evenly spaced along the height of the frame
    # for i in range(1, sample_points + 1):
    #     y_value = int((h - 1) * (i / (sample_points + 1)))  # Evenly spaced along the height
    #     # Find the x-coordinate of the contour point at this y-value
    #     x_values_at_y = contour_points[contour_points[:, 1] == y_value][:, 0]
    #     if len(x_values_at_y) > 0:
    #         sampled_x = int(np.mean(x_values_at_y))  # Average x value for the given y
    #         sampled_points.append((sampled_x, y_value))

    # Sample additional points evenly spaced along the height of the frame (below the 2/3 mark)
    sample_points = 2  # Number of additional points to sample
    for i in range(1, sample_points + 1):
        y_value = int((h - bottom_limit) * (i / (sample_points + 1))) + bottom_limit # Evenly spaced along the height below the limit
        # Find the x-coordinate of the contour point at this y-value
        x_values_at_y = contour_points[contour_points[:, 1] <= y_value][:, 0]  # Use `<=` to include points on the limit
        if len(x_values_at_y) > 0:
            sampled_x = int(np.mean(x_values_at_y))  # Average x value for the given y
            sampled_points.append((sampled_x, y_value))


    # Include the topmost point from the contour
    topmost_point = tuple(contour_points[contour_points[:, 1].argmin()])  # Topmost point
    sampled_points.append(topmost_point)

    # Convert to numpy array
    sampled_points = np.array(sampled_points)

    # Fit a polynomial to the sampled points
    x_sampled = sampled_points[:, 0]
    y_sampled = sampled_points[:, 1]

    # Fit a polynomial using the indices to preserve the order of sampled points
    order = np.argsort(np.arange(len(x_sampled)))  # Preserve original order of points
    coeffs = np.polyfit(x_sampled[order], y_sampled[order], deg=4)  # Fit a 4th degree polynomial
    polynomial = np.poly1d(coeffs)

    # coeffs = np.polyfit(x_sampled, y_sampled, deg=4)  # Fit a 3rd degree polynomial
    # polynomial = np.poly1d(coeffs)

    # Classify contour points based on their position relative to the polynomial curve
    left_boundary_points = []
    right_boundary_points = []

    for point in contour_points:
        x, y = point
        if y < polynomial(x):  # Below the polynomial curve
            left_boundary_points.append(point)
        else:  # Above the polynomial curve
            right_boundary_points.append(point)

    return np.array(left_boundary_points), np.array(right_boundary_points), polynomial, sampled_points


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
    center_x = w_resized // 2

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
        multimask_output=True,
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
        binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
 
    MIN_CONTOUR_AREA = 30000.0  # Set a minimum area threshold (adjust as needed)
    # Set the minimum y-coordinate threshold for removing points at the bottom
    MIN_Y_COORD = int(0.9 * h_resized)  # Keep points above 90% of the image height

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

    # Classify points based on the horizon point
    for point in contour_points:
        if point[0] < horizon_point[0]:  # Left of the horizon point
            left_boundary_points.append(point)
        else:  # Right of the horizon point
            right_boundary_points.append(point)
        # Use the two reference points to classify boundaries
    # left_boundary_points, right_boundary_points, polynomial, sampled_points = classify_boundaries_using_polynomial_fit(contour_points, (center_x, h_resized))
    # print('sampled pionts', sampled_points)
    # Convert to numpy arrays for spline fitting
    left_boundary_points = np.array(left_boundary_points)
    right_boundary_points = np.array(right_boundary_points)

    # Apply spline fitting to smooth the boundaries
    # Apply polynomial fitting to smooth the boundaries
    smoothed_left_boundary = fit_polynomial_to_boundary(left_boundary_points, degree=3)
    smoothed_right_boundary = fit_polynomial_to_boundary(right_boundary_points, degree=3)

    # Convert left and right boundary points into contours format
    left_boundary_contour = (
        np.array(smoothed_left_boundary).reshape((-1, 1, 2)).astype(np.int32)
    )
    right_boundary_contour = (
        np.array(smoothed_right_boundary).reshape((-1, 1, 2)).astype(np.int32)
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

        # Draw the fitted polynomial curve on the overlay
        # for i in range(len(x_curve) - 1):
        #     pt1 = (int(x_curve[i]), int(y_curve[i]))
        #     pt2 = (int(x_curve[i + 1]), int(y_curve[i + 1]))
        #     cv2.line(overlay, pt1, pt2, (255, 255, 0), thickness=2)  # Yellow for the curve

        # Draw the sampled points on the overlay
        # for point in sampled_points:
        #     cv2.circle(overlay, tuple(point), radius=5, color=(255,0, 0), thickness=-1)  # Red for sampled points


        # Draw the smoothed contours on the overlay
        cv2.polylines(overlay, [left_boundary_contour], isClosed=False, color=(0, 255, 0), thickness=2)  # Green for left
        cv2.polylines(overlay, [right_boundary_contour], isClosed=False, color=(0, 0, 255), thickness=2)  # Blue for right
        
        # Draw points on the overlay
        for point, label in zip(point_coords.cpu().numpy(), input_labels.cpu().numpy()):
            cv2.circle(
                overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1
            )
        # cv2.circle(overlay, left_ref, radius=10, color=(255, 255, 255), thickness=-1)
        # cv2.circle(overlay, right_ref, radius=10, color=(255, 255, 255), thickness=-1)
        # for point in left_boundary_points:
        #     cv2.circle(
        #         overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2
        #     )  # Green for left
        # for point in right_boundary_points:
        #     cv2.circle(
        #         overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2
        #     )  # Blue for right

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
