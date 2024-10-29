

# cython: language_level=3
# cython: profile=True
# cython: boundscheck=False, wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: binding=True


import numpy as np
cimport numpy as np
cimport cython

import os
import time
import cv2
np.float = np.float64
import ros_numpy
import rospy
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sensor_msgs.msg import Image
from road_segmentation.msg import DetectedRoadArea

# Set device to GPU if available, otherwise use CPU
cdef object device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rospy.loginfo(f"Using device: {device}")

# Load SAM2 model
cdef str base_path = os.path.dirname(os.path.abspath(__file__))
cdef str checkpoints_dir = os.path.join(base_path, "segment-anything-2", "checkpoints")
cdef str sam2_checkpoint = os.path.join(checkpoints_dir, "sam2_hiera_small.pt")
cdef str model_cfg = "sam2_hiera_s.yaml"
cdef object sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

cdef object predictor = SAM2ImagePredictor(sam2_model)

# Define points for initial segmentation prompt inside the function
@cython.boundscheck(False)
cdef tuple process_image(np.ndarray[np.uint8_t, ndim=3] image, bint publish_image=False):
    #start_time = time.time()
    
    h_original, w_original = image.shape[:2]
    cdef int center_x = int(w_original * 0.75)
    
    # Local point_coords array
    cdef np.ndarray[np.npy_int32, ndim=2] point_coords = np.array([[550, 700], [650, 700], [750, 700]], dtype=np.int32)
    cdef list input_labels = [1, 1, 1]
    
    # Predict masks using SAM2 model
    with torch.cuda.amp.autocast():###@torch.no_grad():
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=input_labels, multimask_output=False
        )
    #binary_mask_np = (masks[0] > 0).astype(np.uint8)

    cdef np.ndarray[np.uint8_t, ndim=2] binary_mask_np = (torch.from_numpy(masks).sum(dim=0) > 0).to(torch.uint8).cpu().numpy()
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("No contours found.")
        return None, None, None

    road_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(road_contour) < 30000.0:
        print("The contour area is below the threshold.")
        return None, None, None

    # Filter road contour points
    cdef int MIN_Y_COORD = int(0.98 * h_original)
    road_contour = road_contour[road_contour[:, 0, 1] < MIN_Y_COORD]


    if len(road_contour) < 2:
        print("Filtered road contour is too small after removing bottom points.")
        return None, None, None

    contour_points = np.array(road_contour).reshape(-1, 2)
    horizon_point = contour_points[np.argmin(contour_points[:, 1])]

    left_boundary_points, right_boundary_points = classify_boundaries_using_horizontal_bins(
        contour_points[contour_points[:, 1] < center_x], h_original
    )

    for point in contour_points[contour_points[:, 1] >= center_x]:
        (left_boundary_points if point[0] < horizon_point[0] else right_boundary_points).append(point)

    overlay = create_overlay(image, binary_mask_np, left_boundary_points, right_boundary_points, publish_image)
    
    #print(f"process image executed in {time.time() - start_time:.4f} seconds")

    return overlay, left_boundary_points, right_boundary_points

@cython.boundscheck(False)
cdef tuple classify_boundaries_using_horizontal_bins(np.ndarray[np.int32_t, ndim=2] contour_points, int frame_height, int num_bins=50):
    #start_time = time.time()
    if len(contour_points) == 0:
        return np.array([]), np.array([])

    y_min = contour_points[:, 1].min()
    bin_height = (frame_height - y_min) // num_bins
    left_boundary_points = np.zeros((len(contour_points), 2), dtype=np.int32)  # Pre-allocate arrays
    right_boundary_points = np.zeros((len(contour_points), 2), dtype=np.int32)

    left_count = 0
    right_count = 0

    for i in range(num_bins):
        y_lower_limit = y_min + i * bin_height
        y_upper_limit = y_min + (i + 1) * bin_height

        bin_points = contour_points[
            (contour_points[:, 1] > y_lower_limit) & (contour_points[:, 1] <= y_upper_limit)
        ]

        if len(bin_points) > 0:
            mean_x = bin_points[:, 0].mean()
            for point in bin_points:
                if point[0] < mean_x:
                    left_boundary_points[left_count] = point
                    left_count += 1
                else:
                    right_boundary_points[right_count] = point
                    right_count += 1

    # Truncate unused space in the boundary arrays
    left_boundary_points = left_boundary_points[:left_count]
    right_boundary_points = right_boundary_points[:right_count]

    #print(f"classify_boundaries executed in {time.time() - start_time:.4f} seconds")
    return left_boundary_points, right_boundary_points

@cython.boundscheck(False)
cdef np.ndarray create_overlay(np.ndarray[np.uint8_t, ndim=3] image, np.ndarray[np.uint8_t, ndim=2] binary_mask_np, np.ndarray[np.int32_t, ndim=2] left_boundary_points, np.ndarray[np.int32_t, ndim=2] right_boundary_points, bint publish_image):
    #start_time=time.time()
    if not publish_image:
        return None

    overlay = image  # No need to copy
    mask = binary_mask_np.reshape(image.shape[0], image.shape[1], 1).repeat(3, axis=2)
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 0.8, mask * np.array([0, 255, 0], dtype=np.uint8), 0.2, 0)

    # Draw boundary points
    for point in left_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 255, 0), thickness=-2)
    for point in right_boundary_points:
        cv2.circle(overlay, tuple(point), radius=3, color=(0, 0, 255), thickness=-2)
    # cv2.polylines(overlay, [left_boundary_points], isClosed=False, color=(0, 255, 0), thickness=2)
    # cv2.polylines(overlay, [right_boundary_points], isClosed=False, color=(0, 0, 255), thickness=2)

    #print(f"create_overlay executed in {time.time() - start_time:.4f} seconds")
    return overlay

cdef class RoadSegmentation:
    cdef object image_pub
    cdef object image_sub
    cdef object left_boundary_pub
    cdef object right_boundary_pub
    cdef bint publish_image

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
        self.left_boundary_pub = rospy.Publisher("/left_boundary", DetectedRoadArea, queue_size=1)
        self.right_boundary_pub = rospy.Publisher("/right_boundary", DetectedRoadArea, queue_size=1)
        self.publish_image = False

    @cython.boundscheck(False)
    cdef image_callback(self, ros_image):
            start = time.time()
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
            # Check if left_boundary is valid before publishing
            if left_boundary is not None and len(left_boundary) > 0:
                self.publish_boundary(left_boundary, self.left_boundary_pub, ros_image.header.stamp)
            
            # Check if right_boundary is valid before publishing
            if right_boundary is not None and len(right_boundary) > 0:
                self.publish_boundary(right_boundary, self.right_boundary_pub, ros_image.header.stamp)
            print(f"image_callback executed in {time.time() - start:.4f} seconds")

    cdef publish_boundary(self, np.ndarray[np.int32_t, ndim=2] boundary, object publisher, stamp):
        #start_time = time.time()
        boundary_msg = DetectedRoadArea()
        boundary_msg.header.stamp = stamp
        boundary_msg.RoadArea.data = [float(point) for point in boundary.flatten()]
        publisher.publish(boundary_msg)
        # print("Boundary published.")
        #print(f"publish_boundary executed in {time.time() - start_time:.4f} seconds")
# ROS node initialization
if __name__ == "__main__":
    rospy.init_node("road_segmentation_node", anonymous=False)
    RoadSegmentation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down road segmentation node.")



