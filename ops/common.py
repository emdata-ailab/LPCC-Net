import sys
import pickle
import importlib
import numpy as np

from second.core.box_np_ops import box_lidar_to_camera, center_to_corner_box3d, project_to_image


def read_txt(path):
    with open(path, 'r') as f:
        da = f.readlines()
    return da


def read_pkl(path):
    with open(path, 'rb') as f:
        da = pickle.load(f)
    return da


def read_bin(path):
    points = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    return points


def add_center_to_point_(center, point_dict):
    for key, points in point_dict.items():
        for point in points:
            point[:, 0:3] += center
    return point_dict


def get_class(class_str):
    mod_name, class_name = class_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, class_name)
    return cls


def get_input(ret_dict, in_key):
    in_key_levels = in_key.split('.')
    res = ret_dict[in_key_levels[0]]
    for key in in_key_levels[1:]:
        res = res[key]
    return res


class Logger(object):
    """ simple logger
    """

    def __init__(self, log_file='log.txt'):
        self.log = open(log_file, 'w')
        self.terminal = None
        self.bind_stdout = False

    def __del__(self):
        self.log.close()
        if self.bind_stdout:
            self.release()

    def bind(self):
        if not self.bind_stdout:
            self.terminal = sys.stdout
            sys.stdout = self
            self.bind_stdout = True

    def release(self):
        if self.bind_stdout:
            sys.stdout = self.terminal
            self.terminal = None
            self.bind_stdout = False

    def write(self, msg):
        self.log.write(msg)
        if self.terminal:
            self.terminal.write(msg)

    def flush(self):
        self.log.flush()
        if self.terminal:
            self.terminal.flush()


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_kitti_calib(calib_path, extend_matrix):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array(
        [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
        [3, 4])
    P1 = np.array(
        [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
        [3, 4])
    P2 = np.array(
        [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
        [3, 4])
    P3 = np.array(
        [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
        [3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    R0_rect = np.array([
        float(info) for info in lines[4].split(' ')[1:10]
    ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect

    Tr_velo_to_cam = np.array([
        float(info) for info in lines[5].split(' ')[1:13]
    ]).reshape([3, 4])
    Tr_imu_to_velo = np.array([
        float(info) for info in lines[6].split(' ')[1:13]
    ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    res = {
        'P0': P0,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'R0_rect': rect_4x4,
        'Tr_velo_to_cam': Tr_velo_to_cam,
        'Tr_imu_to_velo': Tr_imu_to_velo,
    }
    return res


# modified from second.core.box_np_ops.box3d_to_bbox, bugfix
def box3d_to_bbox(box3d, rect, Trv2c, P2):
    box3d_lidar = box3d.copy()
    box3d_lidar[:, 2] -= box3d_lidar[:, 5] / 2
    box3d_camera = box_lidar_to_camera(box3d_lidar, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d_camera[:, :3], box3d_camera[:, 3:6], box3d_camera[:, 6], [0.5, 1.0, 0.5], axis=1)
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def expand_box3d(box3d, expand_ratio):
    res = box3d.copy()
    res[:, 3:6] *= np.array(expand_ratio)
    return res


def expand_bbox(bbox, expand_ratio):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    c_x = bbox[:, 0] + 0.5 * w
    c_y = bbox[:, 1] + 0.5 * h
    res = np.zeros_like(bbox)
    res[:, 0] = c_x - w * expand_ratio[0] * 0.5
    res[:, 1] = c_y - h * expand_ratio[1] * 0.5
    res[:, 2] = c_x + w * expand_ratio[0] * 0.5
    res[:, 3] = c_y + h * expand_ratio[1] * 0.5
    return res


def get_box3d_distance(box3d_lidar):
    return np.sqrt(box3d_lidar[:, 0] ** 2 + box3d_lidar[:, 1] ** 2 + box3d_lidar[:, 2] ** 2)


# flatten nested dict and remove non-dict key-value pairs of the input dict
def flatten_deep_dict(deep_dict, res_dict=None, c_key=None):
    # init
    res_dict = res_dict if res_dict else {}

    # iterate dict elements
    for key, value in deep_dict.items():
        # move every dict to res_dict
        if isinstance(value, dict):
            # set hierarchical keys
            n_key = key if not c_key else c_key + '.' + key
            # make a copy
            res_dict[n_key] = value.copy()
            # remove nested dicts from copys
            for k, v in value.items():
                if isinstance(v, dict):
                    _ = res_dict[n_key].pop(k)
            # check empty dict
            if not res_dict[n_key]:
                _ = res_dict.pop(n_key)
            # recursively flatten nested dict
            _ = flatten_deep_dict(value, res_dict, c_key=key)

    # return flat dict
    return res_dict
