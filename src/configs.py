import numpy as np

# Raw Sensor topics
CAMERA_TOPIC = "resized/camera_fl/image_color"
LIDAR_TOPIC = "/lidar_tc/velodyne_points"
RADAR_TRACKS_TOPIC = "/radar_fc/as_tx/radar_tracks"

# YOLO output topics
YOLO_IMAGE_TOPIC = "yolov9/published_image"
YOLO_BBOX_TOPIC = "/yolov9/bboxInfo"
FUSED_BBOX_TOPIC = "/fused_bbox"

# SAM output topics
SEGMENTATION_MASK_TOPIC = "/road_segmentation"
LEFT_CONTOUR_TOPIC = "/left_boundary"
RIGHT_CONTOUR_TOPIC = "/right_boundary"
LEFT_BOUNDARY = "/road_segment_3d/left_boundary"
RIGHT_BOUNDARY = "/road_segment_3d/right_boundary"

# SPHEREFORMER output topics
LIDAR_SEGMENTATION_TOPIC = "/colored_points"
LIDAR_BBOX_TOPIC = "/bounding_boxes"

# ULTRAFAST output topics
LANE_DETECTION_MASK_TOPIC = "/lane_detection/output"
LEFT_LANE_TOPIC = "/lane_detection/current_lane_left_boundary"
RIGHT_LANE_TOPIC = "/lane_detection/current_lane_right_boundary"
LEFT_LANE_BOUNDARY_TOPIC = "/Left_Line3dPoints"
RIGHT_LANE_BOUNDARY_TOPIC = "/Right_Line3dPoints"

# 2D-3D Transformation
LIDAR_2D_PROJ_TOPIC = '/lidar_2d_projection'

# Calibration Parameters
# Camera intrinsic parameters
PROJ = np.array(
    [
        [3.5612204509314029e03 / 2, 0.0, 9.9143998670769213e02 / 2, 0.0],
        [0, 3.5572532571086072e03 / 2, 7.8349772942764150e02 / 2, 0.0],
        [0, 0.0, 1.0, 0],
    ]
)

# Camera to lidar extrinsic transformation matrix
T1 = np.array(
    [
        [
            -4.8076040039157775e-03,
            1.1565175070195832e-02,
            9.9992156375854679e-01,
            1.3626313209533691e00,
        ],
        [
            -9.9997444266988167e-01,
            -5.3469003551928074e-03,
            -4.7460155553246119e-03,
            2.0700573921203613e-02,
        ],
        [
            5.2915924636425249e-03,
            -9.9991882539643562e-01,
            1.1590585274754983e-02,
            -9.1730421781539917e-01,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
