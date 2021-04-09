import os
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms

from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from data.voxel_generator import kitti_voxel_generator
from ops.common import add_center_to_point_


def merge_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key == 'calib':
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key in ['bbox', 'rgb']:
            ret[key] = torch.stack(elems, dim=0)
        elif key in ['point_dict', 'voxel_dict', 'scene_idx']:
            ret[key] = elems
        else:
            raise NotImplementedError
    return ret


class KittiGroundTruthCar(Dataset):
    """ A car depth completion dataset extracted from the Kitti 3D object detection benchmark,
        which uses ground-truth 3d bounding boxes as mask.
    """

    def __init__(self, cfg):
        self.pkl_path = cfg['pkl_path']
        self.data_root = cfg['data_root']
        self.split = cfg['split']
        self.split_ratio = float(cfg['split_ratio'])
        self.adjusted_rgb = eval(cfg['adjusted_rgb'])  # [w, h]
        self.voxelization = cfg.getboolean('voxelization')
        assert all(size % 8 == 0 for size in self.adjusted_rgb), "only support this"

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        if self.split == 'train':
            self.data = self.data[:int(len(self.data) * self.split_ratio)]
        elif self.split == 'val':
            self.data = self.data[int(len(self.data) * self.split_ratio):]
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])

    def __getraw__(self, index):
        da = self.data[index]

        # get calib
        calib_temp = da['calib']
        calib = {
            'P2': calib_temp['P2'].astype(np.float32),
            'Trv2c': calib_temp['Tr_velo_to_cam'].astype(np.float32),
            'rect': calib_temp['R0_rect'].astype(np.float32)
        }

        # get adjusted bbox
        ori_bbox = da['rgb']['bbox']
        w_h_ratio = self.adjusted_rgb[0] / self.adjusted_rgb[1]
        bbox = np.array([ori_bbox[0].item(),
                         ori_bbox[1].item(),
                         int(ori_bbox[0] + (ori_bbox[3] - ori_bbox[1]) * w_h_ratio),
                         ori_bbox[3].item()])

        # get adjusted image array
        img = Image.open(
            os.path.join(self.data_root, 'training/image_2', str(da['rgb']['image_idx']).zfill(6) + '.png'))
        img_array = np.array(img.crop(bbox).resize(self.adjusted_rgb))
        rgb = self.transform(img_array)

        # get point dict
        point_dict = da['point']['point_dict']
        centerxyz = da['point']['box3d_lidar'][0:3]
        point_dict = add_center_to_point_(centerxyz, point_dict)

        return calib, bbox, rgb, point_dict

    def get_voxel(self, point_dict):
        voxel_dict = {}
        for key, items in point_dict.items():
            voxel_dict.setdefault(key, [])
            for item in items:
                voxel_dict[key].append(kitti_voxel_generator.generate(item, 17000))
        return voxel_dict

    def __getitem__(self, item):
        calib, bbox, rgb, point_dict = self.__getraw__(item)
        example = {
            'calib': calib,
            'bbox': torch.tensor(bbox),
            'rgb': rgb,
            'point_dict': point_dict
        }
        if self.voxelization:
            example['voxel_dict'] = self.get_voxel(point_dict)
        return example

    def __len__(self):
        return len(self.data)
