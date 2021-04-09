import torch
import fire

from configparser import ConfigParser
from models.lpcc_net import LPCC_Net
from ops.common import get_class
from ops.common import get_input
from utils.ui_dialog import MyDialog


# transform indices to coordinates
def indices_2_coors(voxel_indices_cpu, small_voxel_size=[0.1, 0.05, 0.05]):
    coors_shape = (voxel_indices_cpu.shape[0], 4)
    voxel_coors_cpu = torch.zeros(coors_shape, dtype=torch.float32)
    voxel_coors_cpu[:, 0] = voxel_indices_cpu[:, 0]
    voxel_coors_cpu[:, 1] = (voxel_indices_cpu[:, 3].float() + 0.5) * small_voxel_size[2]
    voxel_coors_cpu[:, 2] = (voxel_indices_cpu[:, 2].float() + 0.5) * small_voxel_size[1] - 40
    voxel_coors_cpu[:, 3] = (voxel_indices_cpu[:, 1].float() + 0.5) * small_voxel_size[0] - 3
    return voxel_coors_cpu


# filter coordinates with confidence threshold
def choose_coors(coors, features, threshold):
    new_coors = coors[features[:, 0] > threshold]
    return new_coors


# group coordinates
def group_coors(coors, batch_size):
    coors_list = []
    for b in range(batch_size):
        coors_list.append(coors[coors[:, 0] == b, 1:])
    return coors_list


# feed data to the network and return results
def generator(dataloader,
              net,
              confidence,
              input_key,
              pred_key,
              gt_key):
    for example in dataloader:
        # network feed-forward
        with torch.no_grad():
            res = net(example)
        ret_dict = res[-1]

        # get indices and output features
        in_indices = get_input(ret_dict, input_key).indices
        out_features = torch.sigmoid(get_input(ret_dict, pred_key).features)
        out_indices = get_input(ret_dict, pred_key).indices
        if gt_key:
            gt_indices = get_input(ret_dict, gt_key) if isinstance(get_input(ret_dict, gt_key), torch.Tensor) \
                else get_input(ret_dict, gt_key).indices
        else:
            gt_indices = torch.zeros(0, 4, dtype=torch.int32, device=in_indices.device)

        # get coordinates
        in_coors = indices_2_coors(in_indices)
        out_coors = indices_2_coors(out_indices)
        out_coors = choose_coors(out_coors, out_features, confidence)
        gt_coors = indices_2_coors(gt_indices)

        # group coordinates together
        batch_size = get_input(ret_dict, input_key).batch_size
        in_coors_list = group_coors(in_coors, batch_size)
        out_coors_list = group_coors(out_coors, batch_size)
        gt_coors_list = group_coors(gt_coors, batch_size)

        # return results
        yield in_coors_list, out_coors_list, gt_coors_list


# create data generator
def data_gene(dataset_cfg_path,
              dataset_section,
              model_cfg_path,
              model_path,
              confidence,
              input_key,
              pred_key,
              gt_key,
              eval_flag):
    # get configurations
    dataset_cfg = ConfigParser()
    model_cfg = ConfigParser()
    dataset_cfg.read(dataset_cfg_path)
    model_cfg.read(model_cfg_path)

    # prepare dataset
    dataset = get_class(dataset_cfg[dataset_section]['class'])(dataset_cfg[dataset_section])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
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

    return generator(dataloader, net, confidence, input_key, pred_key, gt_key)


# visualize input, output, and gt point clouds
def visualize(dataset_cfg_path,
              dataset_section,
              model_cfg_path,
              model_path,
              confidence,
              input_key,
              pred_key,
              gt_key=None,
              eval_flag=True):
    data_generator = data_gene(dataset_cfg_path, dataset_section, model_cfg_path, model_path, confidence, input_key,
                               pred_key, gt_key, eval_flag)
    m = MyDialog(data_generator)
    m.configure_traits()


# main entrance
if __name__ == '__main__':
    fire.Fire()
