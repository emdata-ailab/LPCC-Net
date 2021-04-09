import argparse
from models.fusion.rgb_extractors.seg_hrnet.hrnet_config.config import update_config, config


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./models/fusion/rgb_extractors/seg_hrnet/hrnet_config/seg_hrnet_w48_trainval_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484x2.yaml'
                        # default='./second/hrnet_config/seg_hrnet_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
                        , type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return config
