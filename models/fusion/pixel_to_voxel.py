import torch

from torch import nn
from ops.enhanced_spconv import simple_sp_tensor_expansion
from ops.feature_fusion import fuse_rgb_to_voxel
from ops.common import get_input


class Pixel2Voxel(nn.Module):
    """ fuse RGB features to voxels
    """

    def __init__(self,
                 arg_dict):
        super(Pixel2Voxel, self).__init__()

        self.rgb_in_key = arg_dict['rgb_in_key']
        self.voxel_in_key = arg_dict['voxel_in_key']
        self.out_key = arg_dict['out_key']
        self.expansion = eval(arg_dict['expansion'])

    def forward(self, example, ret_dict):
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get input sp tensor
        input_dict = get_input(ret_dict, self.voxel_in_key)
        sample_sp = input_dict['input_sp']  # spconv tensor

        # get rgb features
        rgb_features = get_input(ret_dict, self.rgb_in_key)  # torch tensor

        # get true batch calib
        calib = example['calib']

        # generate dummy inputs
        batch_size = sample_sp.batch_size
        dummy_batch_list = input_dict['dummy_batch_list']
        dummy_rgb_features_list = []
        dummy_calib_dict = {
            'P2_list': [],
            'Trv2c_list': [],
            'rect_list': []
        }
        dummy_pr_list = []
        for index in range(batch_size):
            dummy_rgb_features_list.append(rgb_features[dummy_batch_list[index]].unsqueeze(0))
            for key in ['P2', 'Trv2c', 'rect']:
                dummy_calib_dict[f'{key}_list'].append(
                    torch.tensor(calib[key][dummy_batch_list[index]], device=device).unsqueeze(0))
            dummy_pr_list.append(input_dict['rgb_coor_refine'][dummy_batch_list[index]])
        for key in ['P2', 'Trv2c', 'rect']:
            dummy_calib_dict[key] = torch.cat(dummy_calib_dict[f'{key}_list'], dim=0)
        dummy_rgb_features = torch.cat(dummy_rgb_features_list, dim=0)

        # expand active indices
        expanded = simple_sp_tensor_expansion(sample_sp, self.expansion['kernel'], self.expansion['padding'],
                                              computation_device=device)

        # merge rgb features to voxels
        merged_feature = fuse_rgb_to_voxel(expanded, [0.1, 0.05, 0.05], [-3, -40, 0], dummy_rgb_features,
                                           dummy_calib_dict, pixel_refinement=dummy_pr_list)

        return {self.out_key: merged_feature}
