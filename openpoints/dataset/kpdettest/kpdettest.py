import logging
import os
from typing import Dict

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from openpoints.transforms.transforms_factory import Compose

from ..build import DATASETS


@DATASETS.register_module()
class KeypointDetectionTest(Dataset):
    SUPPORTED_TRANSFORMS = [
        'PointsToTensor',
        'PointCloudCenterAndNormalize',
        'PointCloudScaling',
        'PointCloudJitter'
    ]
    TRANSFORMS_SKIP_KEYPOINTS = {
        'PointCloudJitter'
    }

    def __init__(self,
                 data_root: str = 'data/kpdettest',
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 presample_type: str = 'random',
                 presample_size: int = None,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.presample_type = presample_type
        self.presample_size = presample_size
        self.variable = variable
        self.shuffle = shuffle 
        data_root = os.path.join(data_root, split)
        self.data_root = data_root
        data_list = sorted(os.listdir(data_root))
        self.data_list = [os.path.splitext(item)[0] for item in data_list if item.endswith('.ply')]
        self.data_idx = np.arange(len(self.data_list))
        self.transforms : Dict[str, object] = {}

        if type(self.transform) is Compose:
            for transform in self.transform.transforms:
                transform_fqn = type(transform).__module__ + '.' + type(transform).__name__
                if transform_fqn not in self.transforms:
                    self.transforms[transform_fqn] = transform
        else:
            transform_fqn = type(self.transform).__module__ + '.' + type(self.transform).__name__
            self.transforms[transform_fqn] = self.transform


        assert len(self.data_idx) > 0
        logging.info(f"\nTotal {len(self.data_idx)} samples in {split} set")
 
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        
        ply_path = os.path.join(self.data_root, self.data_list[data_idx] + '.ply')
        json_path = os.path.join(self.data_root, self.data_list[data_idx] + '.json')
        
        # 1) Load PLY point cloud and JSON, parse to numpy arrays
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=8))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # 2) Load JSON
        import json
        with open(json_path) as json_file:
            json_data = json.load(json_file)
            keypoints = np.array(json_data['keypoints'])

        # 3) Transform
        data = {'filename': self.data_list[data_idx], 'points': points, 'keypoints': keypoints, 'feats': normals}

        # Check if type of self.transform is Compose (<class 'openpoints.transforms.transforms_factory.Compose'>)
        if self.transform is not None:
            data['transforms'] = []
            def _transform(transform):
                assert transform.__class__.__name__ in self.SUPPORTED_TRANSFORMS
                if transform.__class__.__name__ == 'PointsToTensor':
                    data['points'] = transform({ 'points' : data['points'] })['points']
                    data['keypoints'] = transform({ 'keypoints' : data['keypoints'] })['keypoints']
                    data['feats'] = transform({ 'feats' : data['feats'] })['feats']
                    state = {}
                else:
                    # If transform contains the methods fit and transform, use those
                    if transform is not None and hasattr(transform, 'fit') and hasattr(transform, 'transform'):
                        state = transform.fit(data['points']) # .fit here serves the purpose of initializing the transform parameters
                        data['points'] = transform.transform(data['points'], state)
                        # Do not transform keypoints if split is not train or if the transform is in the list of transforms that skip keypoints
                        if self.split == 'train' and transform.__class__.__name__ not in self.TRANSFORMS_SKIP_KEYPOINTS:
                            data['keypoints'] = transform.transform(data['keypoints'], state)
                    else:
                        state = {}
                        data['points'] = transform(data['points'])
                transform_fqn = type(transform).__module__ + '.' + type(transform).__name__
                data['transforms'].append({'transform_fqn': transform_fqn, 'state': state})
            if type(self.transform) is Compose:
                for transform in self.transform.transforms:
                    _transform(transform)
            else:
                _transform(self.transform)

        if self.presample:
            if self.presample_type == 'random_constant':
                if self.presample_size is not None:
                    random_indices = torch.randperm(data['points'].shape[0])[:self.presample_size]
                else:
                    raise ValueError('presample_size must be specified if presample_type is random')
            else:
                raise ValueError('presample_type not supported')
            data['points'] = data['points'][random_indices].contiguous()
            data['feats'] = data['feats'][random_indices].contiguous()
            
        return data

    def __len__(self):
        return len(self.data_idx)