import torch
import torch.nn.functional as F

from torch import nn
from models.fusion.rgb_extractors.seg_hrnet.config_read import parse_args
from models.fusion.rgb_extractors.seg_hrnet.seg_hrnet import get_seg_model
from ops.common import get_input

kitti_mean = [0.37913898, 0.3984994, 0.38369808]
kitti_std = [0.31077385, 0.31943962, 0.3287271]
seg_hrnet_channels = {
    'stage2': 48 + 96,
    'stage3': 48 + 96 + 192,
    'stage4': 48 + 96 + 192 + 384,
}
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class HRNetRGBExtractor(nn.Module):
    """ extract rgb features using modified hrnetv2
    """

    def __init__(self,
                 arg_dict):
        super(HRNetRGBExtractor, self).__init__()

        self.in_key = arg_dict['in_key']
        self.out_key = arg_dict['out_key']
        self.out_channel = int(arg_dict['out_channel'])
        self.use_stages = eval(arg_dict['use_stages'])

        # set 1x1 conv layers
        for stage in self.use_stages:
            in_channel = seg_hrnet_channels[stage]
            conv = nn.Conv2d(in_channel, self.out_channel, kernel_size=1, stride=1, padding=0)
            self.add_module(f'conv1x1_{stage}', conv)

        # post process
        self.post_process = nn.Sequential(
            BatchNorm2d(self.out_channel, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        # load pre-trained HRNet model
        self.model = get_seg_model(parse_args())

    def forward(self, example, ret_dict):
        # get input rgb
        rgb = get_input(ret_dict, self.in_key)

        # normalize input rgb
        rgb = rgb.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C
        rgb -= torch.tensor(kitti_mean, dtype=rgb.dtype, device=rgb.device)
        rgb /= torch.tensor(kitti_std, dtype=rgb.dtype, device=rgb.device)
        rgb = rgb.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W

        # get rgb features
        rgb_res_dict = self.model(rgb)

        # merge features
        res_feature = None
        for stage in self.use_stages:
            feature = self.fuse_stage_result(rgb_res_dict[stage])
            feature = getattr(self, f'conv1x1_{stage}')(feature)
            if res_feature is None:
                res_feature = feature
            else:
                res_feature += feature
        res_feature = self.post_process(res_feature)

        return {self.out_key: res_feature}

    def fuse_stage_result(self, features):
        h = features[0].shape[2]
        w = features[0].shape[3]

        resized_feature_list = []
        for feature in features:
            if feature.shape[2] == h:
                resized_feature_list.append(feature)
                continue
            resized = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=False)
            resized_feature_list.append(resized)

        return torch.cat(resized_feature_list, dim=1)
