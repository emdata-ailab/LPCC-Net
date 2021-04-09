import torch
import spconv

from torch import nn
from torch.nn import functional as F
from torchplus.tools import change_default_args
from torchplus.nn import Empty
from spconv.modules import SparseModule
from ops.enhanced_spconv import concat_sp_tensors


class SE_Block(SparseModule):
    """ inplace SE block for sparse convolution tensor
    """

    def __init__(self,
                 in_channels,
                 channels,
                 global_pooling='max'):
        super(SE_Block, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.global_pooling = global_pooling
        self.fc_0 = nn.Linear(self.in_channels, self.channels, bias=True)
        self.fc_1 = nn.Linear(self.channels, self.in_channels, bias=True)

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor), "input should be spconv.SparseConvTensor"
        batch_size = input.batch_size

        # containing all batch SE weights
        w_m = input.features.new_full((input.features.shape[0], input.features.shape[1]), 0)

        # for each batch
        for batch in range(batch_size):
            batch_indexes = input.indices[:, 0] == batch
            if self.global_pooling == 'max':
                w, _ = input.features[batch_indexes].max(dim=0)
            elif self.global_pooling == 'mean':  # averaged by the number of active indices
                w = input.features[batch_indexes].mean(dim=0)
            elif self.global_pooling == 'sum':  # sum
                w = input.features[batch_indexes].sum(dim=0)
            else:
                raise NotImplementedError
            w = F.relu(self.fc_0(w), inplace=True)
            w = torch.sigmoid(self.fc_1(w))
            w_m[batch_indexes] = w

        # calculate new features
        input.features = input.features * w_m

        return input


class BasicBlock(SparseModule):
    """a basic block
    """

    def __init__(self,
                 in_c,
                 out_c,
                 layer_number=2,
                 indice_key=None,
                 se_mode='none',
                 use_norm=True):
        super(BasicBlock, self).__init__()
        assert layer_number > 1, "minimum number of layers is 2"

        self.in_c = in_c
        self.out_c = out_c
        self.layer_number = layer_number
        self.indice_key = indice_key if indice_key is not None else id(self)

        self.se = None
        if se_mode != 'none':
            c = max(8, int(round(self.out_c / 16.0)))
            self.se = SE_Block(self.out_c, c, se_mode)

        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            BatchNorm1d = Empty
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)

        self.conv = spconv.SparseSequential()
        self.conv.add(
            spconv.SparseSequential(
                SubMConv3d(self.in_c, self.out_c, 3, indice_key=self.indice_key),
                BatchNorm1d(self.out_c),
                nn.ReLU(inplace=True),
            )
        )
        for l in range(self.layer_number - 2):
            self.conv.add(
                spconv.SparseSequential(
                    SubMConv3d(self.out_c, self.out_c, 3, indice_key=self.indice_key),
                    BatchNorm1d(self.out_c),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv.add(
            spconv.SparseSequential(
                SubMConv3d(self.out_c, self.out_c, 3, indice_key=self.indice_key),
                BatchNorm1d(self.out_c)
            )
        )

        self.conv1x1 = spconv.SparseSequential(
            SubMConv3d(self.in_c, self.out_c, 1),
            BatchNorm1d(self.out_c)
        )

    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor), "input must be SparseConvTensor"
        residual = input if self.in_c == self.out_c else self.conv1x1(input)
        out = self.conv(input)
        if self.se is not None:
            out = self.se(out)
        out.features += residual.features
        out.features = F.relu(out.features, inplace=True)
        return out


class MultiResolutionModule(nn.Module):
    """fully fuse multi-resolution feature maps
       use sparse inverse convolution for up-sampling
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 op_dict,  # describe how lower branches can be obtained from higher branches
                 kernel_size=3,
                 padding=1,
                 stride=2,
                 feature_concat=False,
                 use_norm=True):
        super(MultiResolutionModule, self).__init__()

        assert isinstance(in_channels, (list, tuple)), "in_channels must be python built-in list or tuple"
        assert isinstance(out_channels, (list, tuple)), "out_channels must be python built-in list or tuple"
        assert isinstance(op_dict, dict), "op_dict must be python built-in dict"
        assert len(out_channels) == len(in_channels) or len(out_channels) - len(in_channels) == 1, \
            "invalid out_channels and in_channels pair"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op_dict = op_dict

        if len(self.out_channels) > len(self.in_channels):  # add a new branch
            if not isinstance(kernel_size, (list, tuple)):
                kernel_size = [kernel_size] * 3
            if not isinstance(padding, (list, tuple)):
                padding = [padding] * 3
            if not isinstance(stride, (list, tuple)):
                stride = [stride] * 3
            op_pair = f"{len(self.in_channels) - 1}_to_{len(self.in_channels)}"
            self.op_dict[op_pair] = {"kernel_size": kernel_size, "padding": padding, "stride": stride,
                                     "indice_key": op_pair}

        self.feature_concat = feature_concat

        if use_norm:
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

        # prepare each output branch
        for output_branch in range(len(self.out_channels)):
            # skip this output branch if corresponding out_channels is set to None
            if self.out_channels[output_branch] is None:
                continue
            # prepare each input branch (w.r.t. corresponding output branch)
            for input_branch in range(len(self.in_channels)):
                # skip this input branch if corresponding in_channels is set to None
                if self.in_channels[input_branch] is None:
                    continue
                # operation list of current input and output branch pair
                operation_list = []
                # generate operation list
                if output_branch > input_branch:  # need to be down-sampled
                    for diff in range(input_branch, output_branch):
                        in_c = self.in_channels[input_branch] if diff == input_branch else self.out_channels[
                            output_branch]
                        out_c = self.out_channels[output_branch]
                        c_op_pair = f"{diff}_to_{diff + 1}"
                        k_s = self.op_dict[c_op_pair]["kernel_size"]
                        p = self.op_dict[c_op_pair]["padding"]
                        s = self.op_dict[c_op_pair]["stride"]
                        i_k = self.op_dict[c_op_pair]["indice_key"]
                        operation_list.append(spconv.SparseSequential(
                            SpConv3d(in_c, out_c, k_s, s, padding=p, indice_key=i_k),
                            BatchNorm1d(out_c),
                            nn.ReLU(inplace=True))
                        )
                elif output_branch == input_branch:
                    operation_list.append(spconv.SparseSequential(
                        SubMConv3d(self.in_channels[input_branch], self.out_channels[output_branch], 3),
                        BatchNorm1d(self.out_channels[output_branch]),
                        nn.ReLU(inplace=True))
                    )
                else:  # need to be up-sampled
                    for i_diff in range(input_branch, output_branch, -1):
                        in_c = self.in_channels[input_branch] if i_diff == input_branch else self.out_channels[
                            output_branch]
                        out_c = self.out_channels[output_branch]
                        i_c_op_pair = f"{i_diff - 1}_to_{i_diff}"
                        k_s = self.op_dict[i_c_op_pair]["kernel_size"]
                        i_k = self.op_dict[i_c_op_pair]["indice_key"]
                        operation_list.append(spconv.SparseSequential(
                            SpInverseConv3d(in_c, out_c, k_s, indice_key=i_k),
                            BatchNorm1d(out_c),
                            nn.ReLU(inplace=True))
                        )
                self.add_module(f"{input_branch}_to_{output_branch}", nn.ModuleList(operation_list))  # add modules

    def forward(self, *branches):
        assert len(branches) == len(self.in_channels), "length of branches and in_channels do not match"
        assert all(b is None or isinstance(b, spconv.SparseConvTensor) for b in
                   branches), "all branches should be SparseConvTensor or None"
        assert all((b is None and in_c is None) or b.features.shape[1] == in_c for b, in_c in
                   zip(branches, self.in_channels)), "channels of branches and in_channels do not match"

        # calculate output branches
        output = []
        # for each output branch
        for output_branch in range(len(self.out_channels)):
            if self.out_channels[output_branch] is None:
                output.append(None)
                continue
            c_b_o_list = []
            # for each input branch (w.r.t. corresponding output branch)
            for input_branch in range(len(self.in_channels)):
                if self.in_channels[input_branch] is None:
                    continue
                o = branches[input_branch]
                # for each operation
                for op in getattr(self, f"{input_branch}_to_{output_branch}"):
                    o = op(o)
                c_b_o_list.append(o)
            c_b_o = concat_sp_tensors(*c_b_o_list, feature_concat=self.feature_concat) if c_b_o_list else None
            output.append(c_b_o)

        return output


class SparseUpSampling(SparseModule):
    """a simple version of up-sampling for sparse tensors.
       deprecated, do not use unless being fully aware of the consequences
    """

    def __init__(self,
                 scale):
        super(SparseUpSampling, self).__init__()
        if not isinstance(scale, (list, tuple)):
            scale = [scale] * 3
        self.scale = scale

    def forward(self, input, resize_shape=None):
        assert isinstance(input, spconv.SparseConvTensor), "input must be SparseConvTensor"
        indices = input.indices * torch.tensor([1, *self.scale], dtype=torch.int32, device=input.indices.device)
        spatial_shape = (torch.tensor(input.spatial_shape) * torch.tensor(self.scale)).tolist()
        if resize_shape is not None:
            assert isinstance(resize_shape, (list, tuple)), "resize_shape must be python built-in list or tuple"
            assert all(abs(n - o) < s for o, n, s in zip(spatial_shape, resize_shape, self.scale)), \
                "offset should be less than scale"
            spatial_shape = resize_shape
        return spconv.SparseConvTensor(input.features, indices, spatial_shape, input.batch_size)
