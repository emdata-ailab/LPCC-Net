from torch import nn
from models.basic_modules import BasicBlock, MultiResolutionModule
from ops.common import get_input


class HRNet_Like(nn.Module):
    """ a HRNet-like backbone network
    """

    def __init__(self,
                 arg_dict):
        super(HRNet_Like, self).__init__()

        self.in_key = arg_dict['in_key']
        self.out_key = arg_dict['out_key']
        self.init_channel = int(arg_dict['init_channel'])
        self.channels = eval(arg_dict['channels'])
        assert isinstance(self.channels, list) and len(self.channels) == 4, "only support this"
        self.se_mode = arg_dict['se_mode']
        self.feature_concat = True if arg_dict['feature_concat'] == 'True' else False
        self.use_norm = True if arg_dict['use_norm'] == 'True' else False

        self.stage_1_basic = BasicBlock(self.init_channel, self.channels[0], layer_number=3, indice_key='0',
                                        se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_1_mr = MultiResolutionModule(self.channels[0:1], self.channels[0:2], {},
                                                feature_concat=self.feature_concat, use_norm=self.use_norm)

        self.stage_2_basic_0 = BasicBlock(self.channels[0], self.channels[0], layer_number=2, indice_key='0',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_2_basic_1 = BasicBlock(self.channels[1], self.channels[1], layer_number=2, indice_key='1',
                                          se_mode=self.se_mode, use_norm=self.use_norm)  # [20 * 800 * 704]
        self.stage_2_mr = MultiResolutionModule(self.channels[0:2], self.channels[0:3], self.stage_1_mr.op_dict,
                                                feature_concat=self.feature_concat, use_norm=self.use_norm)

        self.stage_3_basic_0 = BasicBlock(self.channels[0], self.channels[0], layer_number=2, indice_key='0',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_3_basic_1 = BasicBlock(self.channels[1], self.channels[1], layer_number=2, indice_key='1',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_3_basic_2 = BasicBlock(self.channels[2], self.channels[2], layer_number=2, indice_key='2',
                                          se_mode=self.se_mode, use_norm=self.use_norm)  # [10 * 400 * 352]
        self.stage_3_mr = MultiResolutionModule(self.channels[0:3], self.channels[0:4], self.stage_2_mr.op_dict,
                                                feature_concat=self.feature_concat, use_norm=self.use_norm)

        self.stage_4_basic_0 = BasicBlock(self.channels[0], self.channels[0], layer_number=3, indice_key='0',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_4_basic_1 = BasicBlock(self.channels[1], self.channels[1], layer_number=3, indice_key='1',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_4_basic_2 = BasicBlock(self.channels[2], self.channels[2], layer_number=3, indice_key='2',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_4_basic_3 = BasicBlock(self.channels[3], self.channels[3], layer_number=3, indice_key='3',
                                          se_mode=self.se_mode, use_norm=self.use_norm)  # [5 * 200 * 176]
        self.stage_4_mr = MultiResolutionModule(self.channels[0:4], [self.channels[0], None, None, None],
                                                self.stage_3_mr.op_dict, feature_concat=self.feature_concat,
                                                use_norm=self.use_norm)

        self.stage_5_basic_0 = BasicBlock(self.channels[0], self.channels[0], layer_number=2, indice_key='0',
                                          se_mode=self.se_mode, use_norm=self.use_norm)
        self.stage_5_basic_0_2 = BasicBlock(self.channels[0], self.channels[0], layer_number=2, indice_key='0',
                                            se_mode=self.se_mode, use_norm=self.use_norm)

    def forward(self, example, ret_dict):
        input_sp = get_input(ret_dict, self.in_key)

        # stage 1
        o_1_b = self.stage_1_basic(input_sp)
        o_1_0, o_1_1 = self.stage_1_mr(o_1_b)

        # stage 2
        o_2_0_b = self.stage_2_basic_0(o_1_0)
        o_2_1_b = self.stage_2_basic_1(o_1_1)
        o_2_0, o_2_1, o_2_2 = self.stage_2_mr(o_2_0_b, o_2_1_b)

        # stage 3
        o_3_0_b = self.stage_3_basic_0(o_2_0)
        o_3_1_b = self.stage_3_basic_1(o_2_1)
        o_3_2_b = self.stage_3_basic_2(o_2_2)
        o_3_0, o_3_1, o_3_2, o_3_3 = self.stage_3_mr(o_3_0_b, o_3_1_b, o_3_2_b)

        # stage 4
        o_4_0_b = self.stage_4_basic_0(o_3_0)
        o_4_1_b = self.stage_4_basic_1(o_3_1)
        o_4_2_b = self.stage_4_basic_2(o_3_2)
        o_4_3_b = self.stage_4_basic_3(o_3_3)
        o_4_0, _, _, _ = self.stage_4_mr(o_4_0_b, o_4_1_b, o_4_2_b, o_4_3_b)

        # stage 5
        o_5_0_b = self.stage_5_basic_0(o_4_0)
        o_5_0_b_2 = self.stage_5_basic_0_2(o_5_0_b)

        return {self.out_key: o_5_0_b_2}
