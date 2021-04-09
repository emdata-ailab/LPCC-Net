import torch

from torch import nn
from torch.nn import functional as F
from .head_base import HeadBase
from ops.common import get_input


class FL_Only_Head(HeadBase):
    def __init__(self,
                 arg_dict):
        super(FL_Only_Head, self).__init__(arg_dict)

        self.gt_key = arg_dict['gt_key']

    def loss(self, example, ff_ret_dict):
        # get final output sp tensor of the network
        pred_sp = get_input(ff_ret_dict, self.out_key)

        # get predicted indices and features (sigmoid function applied)
        pred_indices = pred_sp.indices
        pred_features = torch.sigmoid(pred_sp.features)

        # get true ground truth indices
        label_indices = get_input(ff_ret_dict, self.gt_key)

        # generate dummy indices and features for loss calculation
        indices_cated = torch.cat((pred_indices, label_indices), dim=0)
        dummy_pred_indices, i_indices_index = torch.unique(indices_cated, dim=0, return_inverse=True)
        dummy_pred_features = torch.ones(dummy_pred_indices.shape[0], 1, dtype=pred_features.dtype,
                                         device=pred_features.device)
        dummy_pred_features[i_indices_index[:pred_indices.shape[0]]] = pred_features

        # generate label features (with same shape as dummy_pred_features)
        label_features = torch.zeros(dummy_pred_features.shape[0], 1, device=dummy_pred_features.device)
        concatenated = torch.cat((dummy_pred_indices, label_indices), dim=0)
        uniqued, i_index = torch.unique(concatenated, dim=0,
                                        return_inverse=True)  # unique should be the same as dummy_pred_indices
        label_features[i_index[dummy_pred_indices.shape[0]:]] = 1.0

        # calculate focal_loss
        focal_loss = FocalLoss()
        loss = focal_loss(dummy_pred_features, label_features)

        # get stat.
        gt_num = label_indices.shape[0]
        drop_gt_num = dummy_pred_indices.shape[0] - pred_indices.shape[0]
        pred_num = pred_indices.shape[0]

        # calculate recall at different confidence thresholds
        recall = {
            '0.3': float((dummy_pred_features[i_index[dummy_pred_features.shape[0]:]] > 0.3).sum().float() / gt_num),
            '0.5': float((dummy_pred_features[i_index[dummy_pred_features.shape[0]:]] > 0.5).sum().float() / gt_num),
            '0.7': float((dummy_pred_features[i_index[dummy_pred_features.shape[0]:]] > 0.7).sum().float() / gt_num),
            '0.9': float((dummy_pred_features[i_index[dummy_pred_features.shape[0]:]] > 0.9).sum().float() / gt_num),
        }

        # information to be logged and displayed
        loss_info = {
            'gt_num': gt_num,
            'drop_gt_num': drop_gt_num,
            'pred_num': pred_num,
            'recall': recall,
        }

        return loss, loss_info


# copied from internet
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
