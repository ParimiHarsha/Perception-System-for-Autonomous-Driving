#!/usr/bin/env python3
import os
import struct
import time
from functools import wraps

import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import spconv.pytorch as spconv
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from semantic_kitti_ros import SemanticKITTI
from sensor_msgs.msg import PointCloud2, PointField
from sklearn.cluster import DBSCAN
from SphereFormer.util import config
from SphereFormer_changes.unet_spherical_transformer import Semantic as Model
from visualization_msgs.msg import Marker, MarkerArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(
    SCRIPT_DIR,
    "SphereFormer_changes/semantic_kitti_unet32_spherical_transformer.yaml",
)
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "SphereFormer/model_semantic_kitti.pth")

lim_x, lim_y, lim_z = [3, 80], [-20, 20], [-5, 10]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.3, device=torch.device("cuda:0"))


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = rospy.Time.now().to_sec()  # time.time()
        result = func(*args, **kwargs)
        end_time = rospy.Time.now().to_sec()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class PointCloudInference:
    def __init__(
        self,
        config_path,
        checkpoint_path,
    ):
        # Configuration and model initialization
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.cfg = config.load_cfg_from_cfg_file(self.config_path)
        self.model = self._load_model()
        self.semkitti_dataset = SemanticKITTI(split="val")
        self.current_marker_ids = set()

        # ROS publishers and subscribers
        self.pub = rospy.Publisher("/colored_points", PointCloud2, queue_size=1)
        self.bounding_box_pub = rospy.Publisher(
            "/bounding_boxes", MarkerArray, queue_size=1
        )
        rospy.Subscriber(
            "/lidar_tc/velodyne_points",
            PointCloud2,
            self.ros_callback,
            queue_size=1,
        )

    @timer
    def _load_model(self):
        """Load and initialize the model."""

        rospy.loginfo("Loading and initializing the model...")

        # Model configuration
        self.cfg.patch_size = np.array(
            [self.cfg.voxel_size[i] * self.cfg.patch_size for i in range(3)]
        ).astype(np.float32)
        window_size = self.cfg.patch_size * self.cfg.window_size
        window_size_sphere = np.array(self.cfg.window_size_sphere)

        model = Model(
            input_c=self.cfg.input_c,
            m=self.cfg.m,
            classes=self.cfg.classes,
            block_reps=self.cfg.block_reps,
            block_residual=self.cfg.block_residual,
            layers=self.cfg.layers,
            window_size=window_size,
            window_size_sphere=window_size_sphere,
            quant_size=window_size / self.cfg.quant_size_scale,
            quant_size_sphere=window_size_sphere / self.cfg.quant_size_scale,
            rel_query=self.cfg.rel_query,
            rel_key=self.cfg.rel_key,
            rel_value=self.cfg.rel_value,
            drop_path_rate=self.cfg.drop_path_rate,
            window_size_scale=self.cfg.window_size_scale,
            grad_checkpoint_layers=self.cfg.grad_checkpoint_layers,
            sphere_layers=self.cfg.sphere_layers,
            a=self.cfg.a,
        )

        # Load checkpoint
        rospy.loginfo("Loading model weights from checkpoint...")
        checkpoint = torch.load(self.checkpoint_path)
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        model.eval()
        return model

    @timer
    def ros_callback(self, msg):
        """ROS callback to process incoming PointCloud2 messages."""
        rospy.loginfo("Received a message, starting inference...")
        seg_points, output_labels = self.inference_from_ros_message(msg, self.model)
        rospy.loginfo("Inference complete. Processing results...")

        # Check if seg_points is structured and has the required dtype fields
        if seg_points is None or seg_points.dtype.names is None:
            rospy.logerr(
                "Invalid data structure in seg_points. Ensure the point cloud data is structured correctly."
            )
            return

        # Detect and cluster car points
        # car_indices = np.where(output_labels.cpu().numpy() == 0)
        # car_points = seg_points[car_indices]
        # car_points_2d = self._prepare_car_points(car_points)

        # Cluster and create bounding boxes if car points exist
        # bounding_boxes = self._cluster_points(car_points_2d)
        # self.publish_bounding_boxes(bounding_boxes, msg.header)

        # Convert labels to colors and create a colored point cloud
        colors = self.label_to_color(output_labels.cpu().numpy())
        colored_cloud = self.create_colored_pointcloud2(
            ros_numpy.msgify(PointCloud2, seg_points), colors, msg.header
        )
        self.pub.publish(colored_cloud)
        rospy.loginfo("Publishing the processed point cloud and bounding boxes.")

    @timer
    def inference_from_ros_message(self, ros_msg, model):
        pc = ros_numpy.numpify(ros_msg)
        pcd = np.zeros((pc.shape[0], 4))
        pcd[:, 0] = pc["x"]
        pcd[:, 1] = pc["y"]
        pcd[:, 2] = pc["z"]
        np_points = np.asarray(pcd)
        np_points = self.crop_pointcloud(np_points)
        p_points = np_points[:, 0:3]
        intensities = np_points[:, 3]
        binary_points, binary_labels = self.convert_to_binary_format(
            p_points, intensities
        )
        processed_data = self.semkitti_dataset.process_live_data(
            binary_points, binary_labels
        )
        batch_data = [processed_data]
        (coord, xyz, feat, target, offset, inds_reverse) = self.collation_fn_voxelmean(
            batch_data
        )
        inds_reverse = inds_reverse.to("cuda:0", non_blocking=True)
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()
        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
        coord, xyz, feat, target, offset = (
            coord.cuda(non_blocking=True),
            xyz.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, 1)
        assert batch.shape[0] == feat.shape[0]
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            a = time.time()
            output = model(sinput, xyz, batch)
            print("time for model", time.time() - a)
            output = output[inds_reverse, :]
            output = output.max(1)[1]
        points = np.zeros(
            np_points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.float32),
            ],
        )
        points["x"] = np_points[:, 0]
        points["y"] = np_points[:, 1]
        points["z"] = np_points[:, 2]
        return points, output

    def _prepare_car_points(self, car_points):
        """Convert structured array to 2D array for clustering."""
        if car_points.size == 0:
            return np.empty((0, 3))
        return np.array([list(point) for point in car_points[["x", "y", "z"]]])

    # def _cluster_points(self, car_points_2d):
    #     """Run DBSCAN clustering and create bounding boxes."""
    #     if car_points_2d.shape[0] > 0:
    #         clustering = DBSCAN(eps=1, min_samples=50).fit(car_points_2d)
    #         cluster_labels = clustering.labels_
    #         unique_labels = set(cluster_labels)
    #         return [
    #             ((np.min(cluster_points, axis=0)), (np.max(cluster_points, axis=0)))
    #             for label in unique_labels
    #             if label != -1
    #             for cluster_points in [car_points_2d[np.where(cluster_labels == label)]]
    #         ]
    #     return []

    def publish_bounding_boxes(self, bounding_boxes, header):
        """Publish bounding boxes as markers to RViz."""
        marker_array = MarkerArray()
        new_marker_ids = set()

        for i, (min_point, max_point) in enumerate(bounding_boxes):
            marker_id = i
            new_marker_ids.add(marker_id)
            marker = self._create_marker(marker_id, header, min_point, max_point)
            marker_array.markers.append(marker)

        # Add DELETE action for old markers
        for marker_id in self.current_marker_ids - new_marker_ids:
            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.ns = "bounding_boxes"
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        self.bounding_box_pub.publish(marker_array)
        self.current_marker_ids = new_marker_ids

    @timer
    def _create_marker(self, marker_id, header, min_point, max_point):
        """Create a single bounding box marker."""
        marker = Marker()
        marker.header = header
        marker.ns = "bounding_boxes"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (min_point[0] + max_point[0]) / 2
        marker.pose.position.y = (min_point[1] + max_point[1]) / 2
        marker.pose.position.z = (min_point[2] + max_point[2]) / 2
        marker.scale.x = max_point[0] - min_point[0]
        marker.scale.y = max_point[1] - min_point[1]
        marker.scale.z = max_point[2] - min_point[2]
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        return marker

    def _run_model_inference(self, coord, xyz, feat, offset, inds_reverse, np_points):
        """Run the model inference."""
        inds_reverse = inds_reverse.to(device, non_blocking=True)
        offset_ = offset.clone().to(device, non_blocking=True)
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o, device=device) for ii, o in enumerate(offset_)], 0
        ).long()
        coord = torch.cat([batch.unsqueeze(-1), coord.to(device)], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None)

        inputs = (
            coord.to(device, non_blocking=True),
            xyz.to(device, non_blocking=True),
            feat.to(device, non_blocking=True),
            offset.to(device, non_blocking=True),
            batch.to(device, non_blocking=True),
        )
        sinput = spconv.SparseConvTensor(inputs[2], inputs[0].int(), spatial_shape, 1)

        # Ensure model is on the correct device
        self.model = self.model.to(device)

        with torch.cuda.amp.autocast, torch.no_grad():
            output = self.model(sinput, xyz.to(device), batch)
        output_pred = output.argmax(1).view(-1)[inds_reverse]

        return np_points, output_pred

    def collation_fn_voxelmean(self, batch):
        coords, xyz, feats, labels, inds_recons = list(zip(*batch))
        inds_recons = list(inds_recons)
        accmulate_points_num = 0
        offset = []
        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)
        coords = torch.cat(coords)
        xyz = torch.cat(xyz)
        feats = torch.cat(feats)
        labels = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_recons = torch.cat(inds_recons)
        return coords, xyz, feats, labels, offset, inds_recons

    def crop_pointcloud(self, pointcloud):
        """Crop point cloud within the specified limits."""
        mask = np.where(
            (pointcloud[:, 0] >= lim_x[0])
            & (pointcloud[:, 0] <= lim_x[1])
            & (pointcloud[:, 1] >= lim_y[0])
            & (pointcloud[:, 1] <= lim_y[1])
            & (pointcloud[:, 2] >= lim_z[0])
            & (pointcloud[:, 2] <= lim_z[1])
        )
        return pointcloud[mask]

    def convert_to_binary_format(self, np_points, intensities):
        """Convert numpy points to binary format for inference."""
        np_points_with_intensity = np.hstack((np_points, intensities.reshape(-1, 1)))
        binary_points = np_points_with_intensity.astype(np.float32).tobytes()
        labels = np.zeros(np_points_with_intensity.shape[0], dtype=np.uint32)
        return np.asarray(binary_points), labels.tobytes()

    def create_colored_pointcloud2(self, original_cloud, colors, header):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        new_points = []
        for point, color in zip(
            pc2.read_points(
                original_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True
            ),
            colors,
        ):
            rgb_packed = struct.pack("BBBB", color[2], color[1], color[0], 255)
            rgb_float = struct.unpack("f", rgb_packed)[0]
            new_points.append([point[0], point[1], point[2], point[3], rgb_float])
        colored_cloud = pc2.create_cloud(header, fields, new_points)
        return colored_cloud

    def label_to_color(self, labels):
        """Convert segmentation labels to RGB color."""
        color_map = {
            0: [0, 0, 255],
            1: [0, 0, 0],
            10: [0, 0, 0],
            11: [0, 0, 0],
            13: [0, 0, 0],
            15: [0, 0, 0],
            16: [0, 0, 0],
            18: [0, 0, 0],
            20: [0, 0, 0],
            30: [0, 0, 0],
            31: [0, 0, 0],
            8: [255, 0, 0],
            40: [0, 0, 0],
            44: [0, 0, 0],
            48: [0, 0, 0],
            12: [0, 0, 0],
            50: [0, 0, 0],
            14: [0, 0, 0],
            52: [0, 0, 0],
            60: [0, 0, 0],
            70: [0, 0, 0],
            71: [0, 0, 0],
            72: [0, 0, 0],
            80: [0, 0, 0],
            81: [0, 0, 0],
            99: [0, 0, 0],
            252: [0, 0, 0],
            256: [0, 0, 0],
            253: [0, 0, 0],
            254: [0, 0, 0],
            255: [0, 0, 0],
            257: [0, 0, 0],
            258: [0, 0, 0],
            259: [0, 0, 0],
        }
        default_color = color_map[8]
        colors = np.array([color_map.get(label, default_color) for label in labels])
        return colors


if __name__ == "__main__":
    rospy.init_node("pointcloud_inference", anonymous=True)
    # checkpoint_path = "src/sphereformer_ros/src/SphereFormer/model_semantic_kitti.pth"
    inference_node = PointCloudInference(CONFIG_PATH, CHECKPOINT_PATH)
    rospy.spin()
