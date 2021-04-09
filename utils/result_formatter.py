import fire

from ops.common import read_pkl
from second.data.kitti_dataset import KittiDataset, kitti_anno_to_label_file


def SECOND_formatter(res_file,
                     data_root,
                     info_pkl_path,
                     class_names,
                     output_dir):
    """ transform SECOND detection results to kitti label format
    """
    # get SECOND dataset
    ds = KittiDataset(data_root, info_pkl_path, class_names)

    # transform detection results to kitti annos
    annos = ds.convert_detection_to_kitti_annos(read_pkl(res_file))
    for anno in annos:
        anno['dimensions'] = anno['dimensions'][:, [1, 2, 0]]

    # output annos as kitti label files
    kitti_anno_to_label_file(annos, output_dir)


# main entrance
if __name__ == '__main__':
    fire.Fire()
