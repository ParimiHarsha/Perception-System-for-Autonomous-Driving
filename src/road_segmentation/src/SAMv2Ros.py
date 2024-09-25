import os
import sys

import cv2
import numpy as np

np.float = np.float64
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import time

from road_segmentation.msg import DetectedRoadArea

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model

# Set local path to checkpoints directory
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")

sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_base_plus.pt")
# sam2_checkpoint = "/home/avalocal/Documents/ros_workspace/src/road_segmentation/src/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


# Main function
def process_image(image, publish_image=False):
    rospy.loginfo("Starting image processing function.")

    # Time the conversion and resizing
    start_total = time.time()

    # Convert the image to a torch tensor and move it to the device
    start_step = time.time()
    image_tensor = torch.from_numpy(image).to(device)
    h_orig, w_orig, _ = image_tensor.shape
    image_resized = cv2.resize(image, (w_orig, h_orig))
    # image_resized = cv2.resize(image, (640, 640))
    image_resized_tensor = torch.from_numpy(image_resized).to(device)
    rospy.loginfo(f"Image conversion and resizing took: {time.time() - start_step} seconds.")
    h_resized, w_resized, _ = image_resized_tensor.shape

    # Adjust point coordinates based on resizing
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
    predictor.set_image(
        image_resized_tensor.cpu().numpy()
    )  # Assuming predictor needs numpy input
    start_time = rospy.Time.now()
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.cpu().numpy(),
        point_labels=input_labels.cpu().numpy(),
        multimask_output=True,
    )
    rospy.loginfo(f"Mask prediction took: {time.time() - start_step} seconds.")
    end_time = rospy.Time.now()
    rospy.loginfo(f"Masks predicted with scores: {scores}.")

    # Create binary mask
    start_step = time.time()
    masks_tensor = torch.from_numpy(masks).to(device)
    binary_mask = (masks_tensor.sum(dim=0) > 0).to(torch.uint8)
    binary_mask_np = binary_mask.cpu().numpy()
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    rospy.loginfo(f"Number of contour points: {sum(array.size for array in contours)}.")
    rospy.loginfo(f"Binary mask and contour finding took: {time.time() - start_step} seconds.")


    # Find the closest points for left and right boundaries
    left_point = np.array([600, 700])
    right_point = np.array([700, 700])

    left_boundary_points = []
    right_boundary_points = []

    # Iterate through contours and append points based on proximity
    for contour in contours:
        for point in contour:
            point = point[0]  # Extract (x, y) from the contour
            left_distance = np.linalg.norm(np.array(point) - left_point)
            right_distance = np.linalg.norm(np.array(point) - right_point)

            if left_distance < right_distance:
                left_boundary_points.append(point)
            else:
                right_boundary_points.append(point)

    rospy.loginfo(f"Number of points in left boundary: {len(left_boundary_points)}.")
    rospy.loginfo(f"Number of points in right boundary: {len(right_boundary_points)}.")


    # Create contour overlay image if publishing is enabled
    overlay = None
    if publish_image:
        start_step = time.time()
        contour_overlay = image_resized_tensor.clone().cpu().numpy()
        cv2.drawContours(
            contour_overlay, contours, -1, (255, 0, 0), 2
        )  # Draw contours in red

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
        
        rospy.loginfo("Overlay created and points drawn.")

        # Draw points on the overlay
        for point, label in zip(point_coords.cpu().numpy(), input_labels.cpu().numpy()):
            cv2.circle(
                overlay, tuple(point), radius=10, color=(255, 0, 0), thickness=-1
            )

        # Draw the left and right boundary points
        for point in left_boundary_points:
            cv2.circle(overlay, tuple(point), radius=5, color=(0, 255, 0), thickness=-1)  # Green for left
        for point in right_boundary_points:
            cv2.circle(overlay, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)  # Blue for right



    rospy.loginfo(
        f"Image processing execution time: {(end_time - start_time).to_sec()} seconds."
    )

    # Flatten and format contour points for publishing
    contour_points = [
        float(point) for contour in contours for point in contour.flatten()
    ]
    rospy.loginfo(f"Contour overlay creation took: {time.time() - start_step} seconds.")
    return overlay, contour_points, left_boundary_points, right_boundary_points


# def process_image(image, publish_image=False):
#     rospy.loginfo("Starting image processing function.")

#     # Time the conversion and resizing
#     start_total = time.time()

#     # Convert the image to a torch tensor and move it to the device
#     start_step = time.time()
#     image_tensor = torch.from_numpy(image).to(device)
#     h_orig, w_orig, _ = image_tensor.shape
#     image_resized = cv2.resize(image, (640, 640))
#     image_resized_tensor = torch.from_numpy(image_resized).to(device)
#     rospy.loginfo(f"Image conversion and resizing took: {time.time() - start_step} seconds.")
#     h_resized, w_resized, _ = image_resized_tensor.shape

#     # Adjust point coordinates based on resizing (from original to resized 640x640)
#     point_coords = np.array([[300, 700], [500, 700], [700, 700]])
#     point_coords = torch.tensor(
#         [
#             [int(p[0] * 640 / w_orig), int(p[1] * 640 / h_orig)]
#             for p in point_coords
#         ],
#         dtype=torch.int32,
#         device=device,
#     )
#     input_labels = torch.tensor([1, 1, 1], device=device)

#     # Adjust left_point and right_point similarly
#     left_point = np.array([600, 700])
#     right_point = np.array([700, 700])

#     left_point_resized = np.array(
#         [int(left_point[0] * 640 / w_orig), int(left_point[1] * 640 / h_orig)]
#     )
#     right_point_resized = np.array(
#         [int(right_point[0] * 640 / w_orig), int(right_point[1] * 640 / h_orig)]
#     )

#     # Predict masks
#     start_step = time.time()
#     predictor.set_image(
#         image_resized_tensor.cpu().numpy()
#     )  # Assuming predictor needs numpy input
#     start_time = rospy.Time.now()
#     masks, scores, _ = predictor.predict(
#         point_coords=point_coords.cpu().numpy(),
#         point_labels=input_labels.cpu().numpy(),
#         multimask_output=True,
#     )
#     rospy.loginfo(f"Mask prediction took: {time.time() - start_step} seconds.")
#     end_time = rospy.Time.now()
#     rospy.loginfo(f"Masks predicted with scores: {scores}.")

#     # Create binary mask
#     start_step = time.time()
#     masks_tensor = torch.from_numpy(masks).to(device)
#     binary_mask = (masks_tensor.sum(dim=0) > 0).to(torch.uint8)
#     binary_mask_np = binary_mask.cpu().numpy()
#     contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
#     rospy.loginfo(f"Number of contour points: {sum(array.size for array in contours)}.")
#     rospy.loginfo(f"Binary mask and contour finding took: {time.time() - start_step} seconds.")

#     # Find the closest points for left and right boundaries using resized left_point and right_point
#     left_boundary_points = []
#     right_boundary_points = []

#     for contour in contours:
#         for point in contour:
#             point = point[0]  # Extract (x, y) from the contour
#             left_distance = np.linalg.norm(np.array(point) - left_point_resized)
#             right_distance = np.linalg.norm(np.array(point) - right_point_resized)

#             if left_distance < right_distance:
#                 left_boundary_points.append(point)
#             else:
#                 right_boundary_points.append(point)

#     rospy.loginfo(f"Number of points in left boundary: {len(left_boundary_points)}.")
#     rospy.loginfo(f"Number of points in right boundary: {len(right_boundary_points)}.")

#     # Convert contour points back to original size
#     # contour_points_resized = [
#     #     [int(point[0] * w_orig / 640), int(point[1] * h_orig / 640)]
#     #     for contour in contours for point in contour
#     # ]
#     # Convert contour points back to original size
#     contour_points_resized = []
#     for contour in contours:
#         for point in contour:
#             x, y = point[0]  # Extract (x, y) from the point
#             x_orig = int(x * w_orig / 640)
#             y_orig = int(y * h_orig / 640)
#             contour_points_resized.append([x_orig, y_orig])

#     # Flatten and format contour points for publishing
#     contour_points = [
#         float(point) for contour in contour_points_resized for point in contour.flatten()
#     ]
    
#     return overlay, contour_points, left_boundary_points, right_boundary_points


class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        self.image_pub = rospy.Publisher("/road_segmentation", Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.image_callback
        )
        # self.contour_pub = rospy.Publisher(
        #     "/contour_points", DetectedRoadArea, queue_size=1
        # )
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
        overlay, contour_points, left_boundary, right_boundary = process_image(img, self.publish_image)

        # Publish processed overlay image if the flag is set
        if self.publish_image:
            rospy.loginfo("Publishing overlay image.")
            msg = Image()
            msg.header.stamp = ros_image.header.stamp
            msg.height, msg.width = overlay.shape[:2]
            msg.encoding = "rgb8"
            msg.data = np.array(overlay).tobytes()
            self.image_pub.publish(msg)

        # # Publish contour points
        # rospy.loginfo("Publishing contour points.")
        # area_msg = DetectedRoadArea()
        # area_msg.header.stamp = ros_image.header.stamp
        # area_msg.RoadArea.data = contour_points
        # area_msg.RoadArea.layout.dim = [
        #     MultiArrayDimension(
        #         label="height",
        #         size=len(contour_points) // 2,
        #         stride=len(contour_points),
        #     ),
        #     MultiArrayDimension(label="width", size=2, stride=2),
        # ]
        # self.contour_pub.publish(area_msg)

        # Publish left boundary points
        rospy.loginfo("Publishing left boundary points.")
        left_boundary_msg = DetectedRoadArea()
        left_boundary_msg.header.stamp = ros_image.header.stamp
        left_boundary_msg.RoadArea.data = [float(point) for point in np.array(left_boundary).flatten()]
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
        right_boundary_msg.RoadArea.data = [float(point) for point in np.array(right_boundary).flatten()]
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
