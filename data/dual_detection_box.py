import os
import fire
import torch
import random
import pickle
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from data.voxel_generator import kitti_voxel_generator
from ops.common import read_pkl, box3d_to_bbox, expand_bbox, expand_box3d, get_kitti_calib, read_bin
from second.data.kitti_common import get_label_annos, anno_to_rbboxes
from second.core.box_np_ops import box_camera_to_lidar, points_in_rbbox


class DualDetectionBox(Dataset):
    """ An object depth completion dataset extracted from the Kitti 3D object detection benchmark
    """

    def __init__(self, cfg):
        self.pkl_file_dict = eval(cfg['pkl_file_dict'])
        self.image_root = cfg['image_root']
        self.adjusted_rgb = eval(cfg['adjusted_rgb'])  # [w, h]
        self.shuffle_split = cfg.getboolean('shuffle_split')
        self.voxelization = cfg.getboolean('voxelization')
        assert all(size % 8 == 0 for size in self.adjusted_rgb), "only support this"

        self.data = []
        for pkl_file, split_ratio in self.pkl_file_dict.items():
            pkl_data = read_pkl(pkl_file)
            if self.shuffle_split:
                random.shuffle(pkl_data)
            self.data += pkl_data[:int(len(pkl_data) * split_ratio)]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        ])

    def __getraw__(self, index):
        da = self.data[index]

        # get scene index
        scene_idx = da['rgb']['img_id']

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
        img = Image.open(os.path.join(self.image_root, str(scene_idx).zfill(6) + '.png'))
        img_cropped = img.crop(bbox).resize(self.adjusted_rgb, resample=Image.BILINEAR)
        rgb = self.transform(np.array(img_cropped))

        # get point dict
        point_dict = da['point']['point_dict']

        return scene_idx, calib, bbox, rgb, point_dict

    def get_voxel(self, point_dict):
        voxel_dict = {}
        for key, items in point_dict.items():
            voxel_dict.setdefault(key, [])
            if items:
                for item in items:
                    voxel_dict[key].append(kitti_voxel_generator.generate(item, 17000))
        return voxel_dict

    def __getitem__(self, item):
        scene_idx, calib, bbox, rgb, point_dict = self.__getraw__(item)
        example = {
            'scene_idx': scene_idx,
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


def create_inference_dataset(det_anno_path,
                             calib_path,
                             velodyne_path,
                             box3d_expansion,
                             bbox_expansion,
                             output_file):
    # get annos
    annos = get_label_annos(det_anno_path)

    # create examples
    examples = []
    for anno in tqdm(annos):
        # continue if zero object detected
        if anno['image_idx'].shape[0] == 0:
            continue

        # get scene index
        scene_idx = str(anno['image_idx'][0]).zfill(6)

        # get calib
        calib = get_kitti_calib(os.path.join(calib_path, scene_idx + '.txt'), True)

        # get box3d_camera
        box3d_camera = anno_to_rbboxes(anno)
        box3d_lidar = box_camera_to_lidar(box3d_camera, calib["R0_rect"], calib["Tr_velo_to_cam"])
        box3d_lidar[:, 2] += box3d_lidar[:, 5] / 2

        # get expanded box3d
        box3d_lidar_expanded = expand_box3d(box3d_lidar, box3d_expansion)

        # get bbox
        bbox = box3d_to_bbox(box3d_lidar_expanded, calib["R0_rect"], calib["Tr_velo_to_cam"], calib['P2'])
        bbox_expanded = expand_bbox(bbox, bbox_expansion).astype(np.int)
        bbox_expanded[:, 0] = np.clip(bbox_expanded[:, 0], 0, 1242)
        bbox_expanded[:, 1] = np.clip(bbox_expanded[:, 1], 0, 375)
        bbox_expanded[:, 2] = np.clip(bbox_expanded[:, 2], 0, 1242)
        bbox_expanded[:, 3] = np.clip(bbox_expanded[:, 3], 0, 375)

        # read scene pts
        pts = read_bin(os.path.join(velodyne_path, scene_idx + '.bin'))[:, :3]

        # create example
        for idx in range(box3d_lidar_expanded.shape[0]):
            filtered_pts = points_in_rbbox(pts, box3d_lidar_expanded[idx][np.newaxis, ...])
            res = {
                'rgb': {},
                'point': {},
                'calib': calib,
            }
            res['rgb']['bbox'] = bbox_expanded[idx]
            res['rgb']['img_id'] = anno['image_idx'][0]
            res['point']['box3d_lidar'] = box3d_lidar_expanded[idx]
            res['point']['point_dict'] = {
                'source': [pts[filtered_pts.reshape(-1)]],
                'gt': [],
                3: [],
                4: [],
                5: [],
                7: [],
                9: [],
            }
            examples.append(res)

    # write to file
    with open(output_file, 'wb') as f:
        pickle.dump(examples, f)


if __name__ == '__main__':
    fire.Fire()
