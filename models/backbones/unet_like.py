import spconv

from torch import nn
from torchplus.nn import Empty
from torchplus.tools import change_default_args
from models.basic_modules import BasicBlock
from ops.enhanced_spconv import concat_sp_tensors
from ops.common import get_input


class UNet_Like(nn.Module):
    """ a UNet-like backbone network
    """

    def __init__(self,
                 arg_dict):
        super(UNet_Like, self).__init__()

        self.in_key = arg_dict['in_key']
        self.out_key = arg_dict['out_key']
        self.init_channel = int(arg_dict['init_channel'])
        self.channels = eval(arg_dict['channels'])
        assert isinstance(self.channels, list) and len(self.channels) > 1, "invalid channels"
        self.se_mode = arg_dict['se_mode']
        self.feature_concat = True if arg_dict['feature_concat'] == 'True' else False
        self.use_norm = True if arg_dict['use_norm'] == 'True' else False

        if self.use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            SpInverseConv3d = change_default_args(bias=False)(spconv.SparseInverseConv3d)
        else:
            BatchNorm1d = Empty
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            SpInverseConv3d = change_default_args(bias=True)(spconv.SparseInverseConv3d)

        self.stem = spconv.SparseSequential(
            BasicBlock(self.init_channel, self.channels[0], 2, 'subm_0', self.se_mode, use_norm=self.use_norm),
            BasicBlock(self.channels[0], self.channels[0], 2, 'subm_0', self.se_mode, use_norm=self.use_norm)
        )

        for level in range(1, len(self.channels)):
            block = spconv.SparseSequential(
                SpConv3d(self.channels[level - 1], self.channels[level], 3, 2, padding=1, indice_key=f'down_{level}'),
                BatchNorm1d(self.channels[level]),
                nn.ReLU(inplace=True),

                BasicBlock(self.channels[level], self.channels[level], 2, f'subm_{level}', self.se_mode,
                           use_norm=self.use_norm),
                BasicBlock(self.channels[level], self.channels[level], 2, f'subm_{level}', self.se_mode,
                           use_norm=self.use_norm)
            )
            self.add_module(f'encoder_{level}', block)

        for level in range(1, len(self.channels)):
            block = spconv.SparseSequential(
                SpInverseConv3d(self.channels[level], self.channels[level - 1], 3, indice_key=f'down_{level}'),
                BatchNorm1d(self.channels[level - 1]),
                nn.ReLU(inplace=True),

                BasicBlock(self.channels[level - 1], self.channels[level - 1], 2, f'subm_{level - 1}', self.se_mode,
                           use_norm=self.use_norm),
                BasicBlock(self.channels[level - 1], self.channels[level - 1], 2, f'subm_{level - 1}', self.se_mode,
                           use_norm=self.use_norm)
            )
            self.add_module(f'decoder_{level}', block)

        if self.feature_concat:
            for level in range(len(self.channels) - 1):
                block = spconv.SparseSequential(
                    SubMConv3d(self.channels[level] * 2, self.channels[level], 1),
                    BatchNorm1d(self.channels[level]),
                    nn.ReLU(inplace=True)
                )
                self.add_module(f'conv1x1_{level}', block)

    def forward(self, example, ret_dict):
        input_sp = get_input(ret_dict, self.in_key)

        # stem
        m_ret_dict = {'stem': self.stem(input_sp)}

        # encoders
        for level in range(1, len(self.channels)):
            encoder = getattr(self, f'encoder_{level}')
            m_ret_dict[f'enc_{level}'] = encoder(m_ret_dict['stem'] if level == 1 else m_ret_dict[f'enc_{level - 1}'])

        # decoders
        for level in reversed(range(1, len(self.channels))):
            decoder = getattr(self, f'decoder_{level}')
            inter_res = m_ret_dict[f'enc_{level}'] if level == len(self.channels) - 1 else concat_sp_tensors(
                m_ret_dict[f'enc_{level}'], m_ret_dict[f'dec_{level + 1}'], self.feature_concat)
            if self.feature_concat and level != len(self.channels) - 1:
                inter_res = getattr(self, f'conv1x1_{level}')(inter_res)
            m_ret_dict[f'dec_{level}'] = decoder(inter_res)

        # post result
        res = concat_sp_tensors(m_ret_dict['stem'], m_ret_dict['dec_1'], feature_concat=self.feature_concat)
        if self.feature_concat:
            res = getattr(self, 'conv1x1_0')(res)
        m_ret_dict['res'] = res

        return {self.out_key: m_ret_dict}
