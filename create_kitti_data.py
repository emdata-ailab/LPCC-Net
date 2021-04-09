import os
import fire
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from configparser import ConfigParser
from models.lpcc_net import LPCC_Net
from ops.common import get_class, get_input, read_bin
from ops.transform import indices_to_coors


# create inferenced kitti velodyne data for secondary 3d object detection
def create_inferenced_velodyne_data(dataset_cfg_path,
                                    dataset_section,
                                    model_cfg_path,
                                    model_path,
                                    pred_key,
                                    confidence,
                                    velodyne_path,
                                    output_path,
                                    batch_size=6,
                                    eval_flag=True):
    # get configurations
    dataset_cfg = ConfigParser()
    model_cfg = ConfigParser()
    dataset_cfg.read(dataset_cfg_path)
    model_cfg.read(model_cfg_path)

    # prepare dataset
    dataset = get_class(dataset_cfg[dataset_section]['class'])(dataset_cfg[dataset_section])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size,
        pin_memory=False,
        collate_fn=get_class(dataset_cfg[dataset_section]['collate_fn']),
    )

    # prepare network model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LPCC_Net(model_cfg['MODEL']).to(device)
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    if eval_flag:
        net.eval()

    # generate predictions
    print('generating predictions ...', flush=True)
    pred_dict = defaultdict(list)
    for example in tqdm(dataloader):
        # network feed-forward
        with torch.no_grad():
            res = net(example)
        ret_dict = res[-1]

        # get output confidence and indices
        out_features = torch.sigmoid(get_input(ret_dict, pred_key).features)
        out_indices = get_input(ret_dict, pred_key).indices
        filtered_indices = out_indices[out_features[:, 0] > confidence]

        # transform to points
        coors = indices_to_coors(filtered_indices, [0.1, 0.05, 0.05], [-3.0, -40.0, 0.0])[:, [3, 2, 1]]
        coors_w_r = torch.cat((coors, torch.zeros(coors.shape[0], 1, dtype=coors.dtype, device=coors.device)), dim=1)

        # deal with batches
        for batch in range(len(example['scene_idx'])):
            scene_idx = example['scene_idx'][batch]
            b_idx = filtered_indices[:, 0] == batch
            pred_dict[scene_idx].append(coors_w_r[b_idx].cpu().numpy())
    print('done.', flush=True)

    # merge with original velodyne data
    print('writing to file ...', flush=True)
    for s_idx in tqdm(pred_dict.keys()):
        scene_pts = read_bin(os.path.join(velodyne_path, str(s_idx).zfill(6) + '.bin'))
        merged_pts = np.concatenate([*pred_dict[s_idx], scene_pts], axis=0)
        with open(os.path.join(output_path, str(s_idx).zfill(6) + '.bin'), 'wb') as f:
            merged_pts.tofile(f)
    print('done.', flush=True)


# main entrance
if __name__ == '__main__':
    fire.Fire()
