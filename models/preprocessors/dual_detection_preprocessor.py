import torch
import spconv
import random

from torch import nn


class DualDetectionPreprocessor(nn.Module):
    """ pre-process examples from dataset - dual_detection_box
    """

    def __init__(self,
                 arg_dict):
        super(DualDetectionPreprocessor, self).__init__()

        self.out_key = arg_dict['out_key']
        self.spatial_shape = eval(arg_dict['spatial_shape'])
        self.sample_num = eval(arg_dict['sample_num'])

    def forward(self, example, ret_dict):
        # get batch size and device
        batch_size = example['rgb'].shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # generate sample indice list and corresponding gt indice list
        sample_indice_list = []
        gt_indice_list = []
        dummy_batch_list = []
        dummy_batch = 0
        for batch in range(batch_size):
            if not example['voxel_dict'][batch]['gt']:  # for bg (train) / inference examples
                sample_torch = torch.tensor(example['voxel_dict'][batch]['source'][0]['coordinates'],
                                            dtype=torch.int32, device=device)
                sample_indice = torch.cat(
                    (sample_torch.new_full((sample_torch.shape[0], 1), dummy_batch), sample_torch), dim=1)
                sample_indice_list.append(sample_indice)

                gt_indice = torch.zeros(0, 4, dtype=torch.int32, device=device)
                gt_indice_list.append(gt_indice)

                dummy_batch_list.append(batch)
                dummy_batch += 1
                continue

            for line in self.sample_num.keys():  # objects to be completed
                random.shuffle(example['voxel_dict'][batch][line])
                for sample_index in range(self.sample_num[line]):
                    # continue if number of samples is less than current index
                    if len(example['voxel_dict'][batch][line]) <= sample_index:
                        continue
                    # add sample to list
                    sample_torch = torch.tensor(example['voxel_dict'][batch][line][sample_index]['coordinates'],
                                                dtype=torch.int32, device=device)
                    sample_indice = torch.cat(
                        (sample_torch.new_full((sample_torch.shape[0], 1), dummy_batch), sample_torch), dim=1)
                    sample_indice_list.append(sample_indice)
                    # add gt to list (for labeling)
                    gt_torch = torch.tensor(example['voxel_dict'][batch]['gt'][0]['coordinates'], dtype=torch.int32,
                                            device=device)
                    gt_indice = torch.cat((gt_torch.new_full((gt_torch.shape[0], 1), dummy_batch), gt_torch), dim=1)
                    gt_indice_list.append(gt_indice)
                    # dummy batch iteration
                    dummy_batch_list.append(batch)
                    dummy_batch += 1

        # check dummy_batch
        assert dummy_batch > 0, "cannot deal with zero dummy_batch"

        # make sample spconv tensor and gt indices
        sample_indices = torch.cat(sample_indice_list, dim=0)
        sample_features = torch.ones(sample_indices.shape[0], 1, dtype=torch.float32, device=device)
        sample_sp = spconv.SparseConvTensor(sample_features, sample_indices, self.spatial_shape, dummy_batch)
        gt_indices = torch.cat(gt_indice_list, dim=0)

        # provide resized rgb tensor
        rgb_coor_refine_list = []
        resized_height = example['rgb'].shape[2]
        for batch in range(batch_size):
            ori_height = (example['bbox'][batch, 3] - example['bbox'][batch, 1]).item()
            # save transformation info
            x_offset = example['bbox'][batch, 0].item()
            y_offset = example['bbox'][batch, 1].item()
            resize_scale = resized_height / ori_height
            rgb_coor_refine_list.append({
                'x_offset': x_offset,
                'y_offset': y_offset,
                'resize_scale': resize_scale
            })

        # return dict containing sample spconv tensor, gt indices, and resized rgb tensor
        m_ret_dict = {
            'input_sp': sample_sp,
            'gt_indice': gt_indices,
            'resized_rgb': example['rgb'].to(device),
            'rgb_coor_refine': rgb_coor_refine_list,
            'dummy_batch_list': dummy_batch_list
        }
        return {self.out_key: m_ret_dict}
