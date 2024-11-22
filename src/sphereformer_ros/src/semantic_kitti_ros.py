import os
import random
import glob
import numpy as np
import torch
import yaml
import pickle
from SphereFormer.util.data_util import data_prepare

#Elastic distortion
def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3
    bb = (np.abs(x).max(0)//gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag


class SemanticKITTI(torch.utils.data.Dataset):
    def __init__(self, 
        voxel_size=[0.1, 0.1, 0.1], 
        split='train', 
        return_ref=True, 
        # label_mapping="util/semantic-kitti.yaml", 
        label_mapping="/home/dev/Documents/Autonomous-Driving-Perception-System/src/sphereformer_ros/src/SphereFormer/util/semantic-kitti.yaml", 
        rotate_aug=False, 
        flip_aug=False, 
        scale_aug=False, 
        scale_params=[0.95, 1.05], 
        transform_aug=False, 
        trans_std=[0.1, 0.1, 0.1],
        elastic_aug=False, 
        elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        ignore_label=255, 
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
    ):
        super().__init__()
        self.num_classes = 19
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.return_ref = return_ref
        self.split = split
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.elastic_aug = elastic_aug
        self.elastic_gran, self.elastic_mag = elastic_params[0], elastic_params[1]
        self.use_tta = use_tta
        self.vote_num = vote_num
        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size
           

    def process_live_data(self, binary_points, binary_labels):
        # Convert binary data back to numpy arrays
        raw_points = np.frombuffer(binary_points, dtype=np.float32).reshape(-1, 4)
        raw_labels = np.frombuffer(binary_labels, dtype=np.uint32).reshape(-1, 1)

        # Apply augmentations and transformations as done in __getitem__
        # Example: Rotate, flip, scale, and translate the points
        if self.rotate_aug:
            # Apply rotation augmentation
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            raw_points[:, :2] = np.dot(raw_points[:, :2], j)

        if self.flip_aug:
            # Apply flipping augmentation
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                raw_points[:, 0] = -raw_points[:, 0]
            elif flip_type == 2:
                raw_points[:, 1] = -raw_points[:, 1]
            elif flip_type == 3:
                raw_points[:, :2] = -raw_points[:, :2]


        if self.scale_aug:
            # Apply scaling augmentation
            noise_scale = np.random.uniform(self.scale_params[0], self.scale_params[1])
            raw_points[:, 0] = noise_scale * raw_points[:, 0]
            raw_points[:, 1] = noise_scale * raw_points[:, 1]

        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            raw_points[:, 0:3] += noise_translate

        if self.elastic_aug:
            # Apply elastic distortion
            raw_points[:, 0:3] = elastic(raw_points[:, 0:3], self.elastic_gran[0], self.elastic_mag[0])
            raw_points[:, 0:3] = elastic(raw_points[:, 0:3], self.elastic_gran[1], self.elastic_mag[1])

        # Assign features and labels
        feats = raw_points
        xyz = raw_points[:, :3]
        labels_in = raw_labels

        # Process xyz, feats, labels_in using the data_prepare function
        # Adjust this part based on your data_prepare function
        if self.split == 'train':
            coords, xyz, feats, labels = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
        else:
            coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)

        # Return the processed data
        if self.split in ['val', 'test']:
            return coords, xyz, feats, labels, inds_reconstruct
        else:
            return coords, xyz, feats, labels
