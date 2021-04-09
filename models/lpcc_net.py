from torch import nn
from ops.common import get_class

DEFAULT_MODULE_ARGS = {
    'use_norm': 'True',
}


def read_module_args(args):
    ret_dict = DEFAULT_MODULE_ARGS.copy()
    arg_list = args.split('|')
    for arg in arg_list:
        if not arg.strip():
            continue
        k, v = arg.split('=', 1)
        ret_dict[k.strip()] = v.strip()
    return ret_dict


class LPCC_Net(nn.Module):
    """
    Local point cloud completion network

    """

    def __init__(self,
                 model_cfg):
        super().__init__()
        self.name = 'LPCC_Net'

        # initialize modules
        self.module_list = []
        mods_str_list = model_cfg['mods'].split(';')
        mods_args_str_list = model_cfg['mods_args'].split(';')
        assert len(mods_str_list) == len(mods_args_str_list), 'mods and mods_args do not match'
        for index in range(len(mods_str_list)):
            mod_name = mods_str_list[index].strip()
            mod_name_short = mod_name.rsplit('.', 1)[1] + '_' + str(index)
            mod_args = mods_args_str_list[index].strip()
            mod = get_class(mod_name)(read_module_args(mod_args))
            self.add_module(mod_name_short, mod)
            self.module_list.append(mod_name_short)

    def network_forward(self, example):
        """
        forward function for each sub module

        :param example: input to the network
        :return: feed forward result dict
        """
        ret_dict = {}
        for mod_name in self.module_list:
            mod_ret_dict = getattr(self, mod_name)(example, ret_dict)
            ret_dict.update(mod_ret_dict)
        return ret_dict

    def forward(self, example):
        """
        forward function for the whole network

        :param example: input to the network
        :return: dict, at least including 'loss' in training mode, or 'preds' in eval mode
        """
        ff_ret_dict = self.network_forward(example)
        if self.training:
            return self.loss(example, ff_ret_dict)
        else:
            return self.predict(example, ff_ret_dict)

    def loss(self, example, ff_ret_dict):
        """
        call the loss function of each sub module, if exists

        :param example: input to the network
        :param ff_ret_dict: dict, containing the feed forward results of all sub modules
        :return: loss dict (at least includes 'loss'), and loss info dict
        """
        loss_ret = {}
        loss_info_ret = {}

        loss = 0.0
        for mod_name in self.module_list:
            mod = getattr(self, mod_name)
            if hasattr(mod, 'loss'):
                mod_loss, loss_info = mod.loss(example, ff_ret_dict)
                loss += mod_loss
                loss_ret[mod_name] = mod_loss
                if loss_info:
                    loss_info_ret.update(loss_info)
        loss_ret['loss'] = loss

        return loss_ret, loss_info_ret, ff_ret_dict

    def predict(self, example, ff_ret_dict):
        """
        call the predict function of each sub module, if exists

        :param example: input to the network
        :param ff_ret_dict: dict, containing the feed forward results of all sub modules
        :return: dict, including the predictions of each sub module, if exist
        """
        pred_ret = {}
        for mod_name in self.module_list:
            mod = getattr(self, mod_name)
            if hasattr(mod, 'predict'):
                mod_pred = mod.predict(example, ff_ret_dict)
                pred_ret.update(mod_pred)
        return pred_ret, ff_ret_dict
