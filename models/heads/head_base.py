import spconv

from torch import nn
from torchplus.tools import change_default_args
from models.basic_modules import BasicBlock
from ops.common import get_input


class HeadBase(nn.Module):
    """ a simple head base
    """

    def __init__(self,
                 arg_dict):
        super(HeadBase, self).__init__()

        self.in_key = arg_dict['in_key']
        self.out_key = arg_dict['out_key']
        self.in_channel = int(arg_dict['in_channel'])
        self.out_channel = int(arg_dict['out_channel'])
        self.block_num = int(arg_dict['block_num'])
        assert self.block_num > 0, "minimum block number is one"
        self.se_mode = arg_dict['se_mode']
        self.use_norm = True if arg_dict['use_norm'] == 'True' else False

        if self.use_norm:
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)

        for index in range(self.block_num):
            block = BasicBlock(self.in_channel, self.in_channel, 2, f'subm_{id(self)}', self.se_mode,
                               use_norm=self.use_norm)
            self.add_module(f'block_{index}', block)

        self.post_block = spconv.SparseSequential(
            SubMConv3d(self.in_channel, self.out_channel, 1)
        )

    def forward(self, example, ret_dict):
        # get input sparse conv tensor
        res = get_input(ret_dict, self.in_key)

        # go through blocks
        for index in range(self.block_num):
            res = getattr(self, f'block_{index}')(res)

        # post block process
        res = self.post_block(res)

        return {self.out_key: res}
