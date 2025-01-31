# #!/bin/bash

# # Script to start recording a rosbag with specific topics

# # Topics to record (using regex for wildcard matching)
# TOPICS='(/right_boundary_global|/left_boundary_global|/novatel/.*|/lidar_tc/velodyne_points|/radar_fc/as_tx/objects|/resized/camera_fl/camera_info|/resized/camera_fl/image_color|/ctrl_ref.*|/vehicle.*|/yoloLiveNode/bboxInfo|/yolo_detection_node/published_image|/local_path|/lat_ctrl_perf|/tf_static|/road_segment_3d/left_boundary|/road_segment_3d/right_boundary|/colored_points|/fused_bbox|/left_boundary|/right_boundary|/current_fsm_status)'

# # Output file name with timestamp
# BAG_NAME="recorded_bag_$(date +'%Y-%m-%d_%H-%M-%S').bag"

# # Directory to save the bag file (default: current directory)
# SAVE_DIR="/media/dev/T9/0129_rosbags"

# # Ensure the save directory exists
# mkdir -p "$SAVE_DIR"

# # Start recording
# echo "Starting rosbag recording..."
# echo "Recording topics: $TOPICS"
# echo "Saving to: $SAVE_DIR/$BAG_NAME"

# rosbag record -e --split --size=5000 -o "$SAVE_DIR/$BAG_NAME" "$TOPICS"


#!/bin/bash

# Script to start recording a rosbag with specific topics based on user input

# Define topic groups
declare -A TOPIC_SETS

# Raw sensor data
TOPIC_SETS[raw]="/resized/camera_fl/image_color /lidar_tc/velodyne_points /radar_fc/as_tx/radar_tracks"

# Perception outputs
TOPIC_SETS[perception]="/yolov9/published_image /yolov9/bboxInfo /fused_bbox /road_segmentation /road_segment_3d/left_boundary /road_segment_3d/right_boundary /colored_points /bounding_boxes"

# Planning topics
TOPIC_SETS[planning]="/local_path /lane_detection/output /lane_detection/current_lane_left_boundary /lane_detection/current_lane_right_boundary /Left_Line3dPoints /Right_Line3dPoints"

# Controls topics
TOPIC_SETS[controls]="/ctrl_ref.* /vehicle.* /lat_ctrl_perf /current_fsm_status"

# Outputs (a mix of perception & planning)
TOPIC_SETS[outputs]="/yoloLiveNode/bboxInfo /yolo_detection_node/published_image /tf_static /lidar_2d_projection"

# Default to "raw" if no argument is provided
CATEGORY=${1:-raw}

# Validate input category
if [[ -z "${TOPIC_SETS[$CATEGORY]}" ]]; then
    echo "Invalid category: $CATEGORY"
    echo "Valid categories: ${!TOPIC_SETS[@]}"
    exit 1
fi

# Get the topics for the selected category
TOPICS="${TOPIC_SETS[$CATEGORY]}"

# Output file name with timestamp
BAG_NAME="recorded_bag_${CATEGORY}_$(date +'%Y-%m-%d_%H-%M-%S').bag"

# Directory to save the bag file
SAVE_DIR="/media/dev/T9/0129_rosbags"

# Ensure the save directory exists
mkdir -p "$SAVE_DIR"

# Start recording
echo "Starting rosbag recording..."
echo "Category: $CATEGORY"
echo "Recording topics: $TOPICS"
echo "Saving to: $SAVE_DIR/$BAG_NAME"

rosbag record -e --split --size=5000 -o "$SAVE_DIR/$BAG_NAME" $TOPICS

#Usage ./record_rosbag.sh [argument]
