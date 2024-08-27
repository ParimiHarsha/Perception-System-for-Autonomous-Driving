import os
import sys

import cv2
import numpy as np
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from road_segmentation.msg import DetectedRoadArea

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model
sam2_checkpoint = "/home/avalocal/Documents/ros_workspace/src/road_segmentation/src/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


# Main function
def process_image(image, publish_image=False):
    rospy.loginfo("Starting image processing function.")

    # Convert the image to a torch tensor and move it to the device
    image_tensor = torch.from_numpy(image).to(device)
    h_orig, w_orig, _ = image_tensor.shape
    image_resized = cv2.resize(image, (w_orig, h_orig))
    # image_resized = cv2.resize(image, (640, 640))
    image_resized_tensor = torch.from_numpy(image_resized).to(device)

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
    predictor.set_image(
        image_resized_tensor.cpu().numpy()
    )  # Assuming predictor needs numpy input
    start_time = rospy.Time.now()
    masks, scores, _ = predictor.predict(
        point_coords=point_coords.cpu().numpy(),
        point_labels=input_labels.cpu().numpy(),
        multimask_output=True,
    )
    end_time = rospy.Time.now()
    rospy.loginfo(f"Masks predicted with scores: {scores}.")

    # Create binary mask
    masks_tensor = torch.from_numpy(masks).to(device)
    binary_mask = (masks_tensor.sum(dim=0) > 0).to(torch.uint8)
    binary_mask_np = binary_mask.cpu().numpy()
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rospy.loginfo(f"Number of contour points: {sum(array.size for array in contours)}.")

    # Create contour overlay image if publishing is enabled
    overlay = None
    if publish_image:
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

    rospy.loginfo(
        f"Image processing execution time: {(end_time - start_time).to_sec()} seconds."
    )

    # Flatten and format contour points for publishing
    contour_points = [
        float(point) for contour in contours for point in contour.flatten()
    ]

    return overlay, contour_points


class RoadSegmentation:
    def __init__(self):
        rospy.loginfo("Initializing RoadSegmentation class.")
        self.image_pub = rospy.Publisher("/road_segmentation", Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            "/resized/camera_fl/image_color", Image, self.image_callback
        )
        self.contour_pub = rospy.Publisher(
            "/contour_points", DetectedRoadArea, queue_size=1
        )
        self.publish_image = True  # Boolean flag for publishing the image

    def image_callback(self, ros_image):
        rospy.loginfo("Image received, processing...")
        img = ros_numpy.numpify(ros_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay, contour_points = process_image(img, self.publish_image)

        # Publish processed overlay image if the flag is set
        if self.publish_image:
            rospy.loginfo("Publishing overlay image.")
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height, msg.width = overlay.shape[:2]
            msg.encoding = "rgb8"
            msg.data = np.array(overlay).tobytes()
            self.image_pub.publish(msg)

        # Publish contour points
        rospy.loginfo("Publishing contour points.")
        area_msg = DetectedRoadArea()
        area_msg.header.stamp = ros_image.header.stamp
        area_msg.RoadArea.data = contour_points
        area_msg.RoadArea.layout.dim = [
            MultiArrayDimension(
                label="height",
                size=len(contour_points) // 2,
                stride=len(contour_points),
            ),
            MultiArrayDimension(label="width", size=2, stride=2),
        ]
        self.contour_pub.publish(area_msg)


if __name__ == "__main__":
    rospy.init_node("road_segmentation_node", anonymous=False)
    rospy.loginfo("Road segmentation node started.")
    road_segmentation = RoadSegmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down road segmentation node.")
