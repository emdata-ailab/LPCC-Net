import torch
import numpy as np

from ops.transform import indices_to_coors, coors_to_pixels
from ops.common import get_input
from .head_base import HeadBase
from .fl_only import FocalLoss


def dist_chamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b: Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()
    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x)  # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)  # Diagonal elements yy
    P = (rx.transpose(2, 1) + ry - 2 * zz).sqrt()
    return P


class SFL_PPC_Head(HeadBase):
    def __init__(self,
                 arg_dict):
        super(SFL_PPC_Head, self).__init__(arg_dict)

        self.gt_key = arg_dict['gt_key']
        self.pre_key = arg_dict['pre_key']
        self.lambda_F = float(arg_dict['lambda_f'])
        self.lambda_C = float(arg_dict['lambda_c'])
        self.k = int(arg_dict['k'])
        self.m = int(arg_dict['m'])

    def loss(self, example, ff_ret_dict):
        # get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get final output sp tensor of the network
        pred_sp = get_input(ff_ret_dict, self.out_key)

        # get preprocessor output dict
        pre_dict = get_input(ff_ret_dict, self.pre_key)

        # get predicted indices and features (sigmoid function applied)
        pred_indices = pred_sp.indices
        pred_features = torch.sigmoid(pred_sp.features)

        # get true ground truth indices
        label_indices = get_input(ff_ret_dict, self.gt_key)

        # generate dummy indices and features for loss calculation
        indices_cated = torch.cat((pred_indices, label_indices), dim=0)
        dummy_pred_indices, i_indices_index = torch.unique(indices_cated, dim=0,
                                                           return_inverse=True)
        dummy_pred_features = torch.ones(dummy_pred_indices.shape[0], 1, dtype=pred_features.dtype,
                                         device=pred_features.device)
        dummy_pred_features[i_indices_index[:pred_indices.shape[0]]] = pred_features

        # generate label features (with same shape as dummy_pred_features)
        label_features = torch.zeros(dummy_pred_features.shape[0], 1, device=dummy_pred_features.device)
        concatenated = torch.cat((dummy_pred_indices, label_indices), dim=0)
        uniqued, i_index = torch.unique(concatenated, dim=0, return_inverse=True)
        label_features[i_index[dummy_pred_indices.shape[0]:]] = 1.0

        # project dummy_pred_indices to pixels
        coors = indices_to_coors(dummy_pred_indices, voxel_size=[0.1, 0.05, 0.05], offset=[-3.0, -40.0, 0.0])
        calib = example['calib']
        batch_size = pred_sp.batch_size
        dummy_batch_list = pre_dict['dummy_batch_list']
        dummy_calib_dict = {
            'P2_list': [],
            'Trv2c_list': [],
            'rect_list': []
        }
        for index in range(batch_size):
            for key in ['P2', 'Trv2c', 'rect']:
                dummy_calib_dict[f'{key}_list'].append(
                    torch.tensor(calib[key][dummy_batch_list[index]], device=device).unsqueeze(0))
        for key in ['P2', 'Trv2c', 'rect']:
            dummy_calib_dict[key] = torch.cat(dummy_calib_dict[f'{key}_list'], dim=0)

        # pixel_location is location of pixel corresponding the 3d points
        pixel_location = coors_to_pixels(coors, dummy_calib_dict, pixel_refinement=None, post_process='none')
        for b in range(pred_sp.batch_size):
            b_idx = pixel_location[:, 0] == b
            pixel_location[b_idx, 1:] *= pre_dict['rgb_coor_refine'][dummy_batch_list[b]]['resize_scale']
        pixel_location = pixel_location.round()

        mean_dis = torch.zeros(pixel_location.shape[0], 1, device=pixel_location.device)
        for batch in range(pred_sp.batch_size):
            dp_idx = dummy_pred_indices[:, 0] == batch
            shrinked_dummy_pred_indices = dummy_pred_indices[dp_idx, 1:]
            gt_idx = label_indices[:, 0] == batch
            shrinked_label_indices = label_indices[gt_idx, 1:]
            # if bg examples, continue
            if shrinked_label_indices.shape[0] == 0:
                continue

            dummy_dis_to_gt = dist_chamfer(torch.unsqueeze(shrinked_dummy_pred_indices, dim=0),
                                           torch.unsqueeze(shrinked_label_indices, dim=0))
            k = shrinked_label_indices.shape[0] if self.k > shrinked_label_indices.shape[0] else self.k
            k_nearest_dis, _ = dummy_dis_to_gt[0].topk(k, dim=1, largest=False, sorted=True)
            batch_mean_dis = torch.mean(k_nearest_dis, dim=1, keepdim=True)
            mean_dis[dp_idx] = batch_mean_dis.float()

        pixel_w_dis = torch.cat((pixel_location, mean_dis.float()), dim=1)
        pixel_w_dis_np = pixel_w_dis.cpu().numpy()
        pixel_w_dis_sorted_index = torch.tensor(
            np.lexsort((pixel_w_dis_np[:, 3], pixel_w_dis_np[:, 2], pixel_w_dis_np[:, 1], pixel_w_dis_np[:, 0]),
                       axis=0))
        pixel_w_dis_sorted = pixel_w_dis[pixel_w_dis_sorted_index]
        pixel_w_dis_sorted_unique, pixel_w_dis_sorted_unique_inv_idx, pixel_w_dis_sorted_unique_c = torch.unique(
            pixel_w_dis_sorted[:, :3], dim=0, return_inverse=True, return_counts=True)
        p_w_d_head = [torch.sum(pixel_w_dis_sorted_unique_c[:i]).tolist() for i in
                      range(len(pixel_w_dis_sorted_unique_c))]

        dummy_pred_features = dummy_pred_features[pixel_w_dis_sorted_index]
        label_features = label_features[pixel_w_dis_sorted_index]
        flag = torch.ones(label_features.shape[0] + self.m - 1, 1, dtype=torch.uint8)
        for i in range(self.m):
            flag[torch.tensor(p_w_d_head) + i] = False
        flag = flag[:label_features.shape[0]]
        flag[label_features == 1.0] = True  # make sure gt always be labeled
        label_features_l = label_features[flag]
        dummy_pred_features_l = dummy_pred_features[flag]

        # calculate focal_loss
        focal_loss = FocalLoss()
        loss_f = focal_loss(dummy_pred_features_l, label_features_l)

        # calculate stat.
        dropped_f_label = (label_features.sum() - label_features_l.sum()).cpu().item()

        set_p__p_gt = label_features == 0.0
        set_p_gt = (label_features == 1.0) & (dummy_pred_features != 1.0)
        set_gt__p_gt = dummy_pred_features == 1.0
        flag_p__p_gt = flag[set_p__p_gt].sum().item()
        flag_p_gt = flag[set_p_gt].sum().item()
        flag_gt__p_gt = flag[set_gt__p_gt].sum().item()
        total_p__p_gt = set_p__p_gt.sum().item()
        total_p_gt = set_p_gt.sum().item()
        total_gt__p_gt = set_gt__p_gt.sum().item()
        flag_test_dict = {
            'p__p_gt_total': total_p__p_gt,
            'p__p_gt_flag': flag_p__p_gt,
            'p__p_gt_ratio': 0 if total_p__p_gt == 0 else flag_p__p_gt / total_p__p_gt,
            'p_gt_total': total_p_gt,
            'p_gt_flag': flag_p_gt,
            'p_gt_ratio': 0 if total_p_gt == 0 else flag_p_gt / total_p_gt,
            'gt__p_gt_total': total_gt__p_gt,
            'gt__p_gt_flag': flag_gt__p_gt,
            'gt__p_gt_ratio': 0 if total_gt__p_gt == 0 else flag_gt__p_gt / total_gt__p_gt,
        }

        gt_num = label_indices.shape[0]
        drop_gt_num = dummy_pred_indices.shape[0] - pred_indices.shape[0]
        pred_num = pred_indices.shape[0]

        if label_indices.shape[0] > 0:
            thres_acc_dict = {}
            thres_recall_dict = {}
            thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for thre in thres:
                num_dummy_labeled_gt = (dummy_pred_features_l.detach().cpu() == 1.0).sum().float()
                # acc.
                thre_pred = (dummy_pred_features_l.detach().cpu() > thre).float()
                thre_total = dummy_pred_features_l.detach().cpu().shape[0]
                thre_correct = ((thre_pred - label_features_l.cpu()).abs() < 0.001).sum()
                thres_acc_dict[f'{thre}'] = ((thre_correct - num_dummy_labeled_gt) /
                                        (thre_total - num_dummy_labeled_gt)).item()
                # recall
                thre_gt_idx = label_features_l == 1.0
                thre_recall_total = thre_gt_idx.sum().cpu()
                thre_recall_correct = (dummy_pred_features_l[thre_gt_idx].detach().cpu() > thre).sum()
                thres_recall_dict[f'{thre}'] = ((thre_recall_correct - num_dummy_labeled_gt) /
                                           (thre_recall_total - num_dummy_labeled_gt)).item()

        # Projection Perspective Constraint Loss loss
        label_features_ppc = label_features.clone()
        BUVF = torch.cat((pixel_w_dis_sorted[:, :3], dummy_pred_features.detach()), dim=1)
        BUVF_np = BUVF.cpu().numpy()
        BUVF_sorted_idx = torch.tensor(
            np.lexsort((-BUVF_np[:, 3], BUVF_np[:, 2], BUVF_np[:, 1], BUVF_np[:, 0]), axis=0))
        BUVF_sorted = BUVF[BUVF_sorted_idx]
        label_features_ppc = label_features_ppc[BUVF_sorted_idx]
        label_features_ppc[torch.tensor(p_w_d_head), 0] = BUVF_sorted[torch.tensor(p_w_d_head), 3]
        dummy_pred_features = dummy_pred_features[BUVF_sorted_idx]
        flag = flag[BUVF_sorted_idx]

        # deal with bg examples
        for b in range(pred_sp.batch_size):
            if len(example['voxel_dict'][pre_dict['dummy_batch_list'][b]]['gt']) == 0:
                b_bg_idx = BUVF_sorted[:, 0] == b
                label_features_ppc[b_bg_idx] = 0

        # calculate loss_ppc
        loss_ppc_L1 = torch.nn.L1Loss()
        loss_ppc = loss_ppc_L1(dummy_pred_features[~flag], label_features_ppc[~flag])

        # total loss
        loss = self.lambda_F * loss_f + self.lambda_C * loss_ppc

        # information to be logged and displayed
        loss_info = {
            'gt_num': gt_num,
            'drop_gt_num': drop_gt_num,
            'pred_num': pred_num,
            'dropped_f_label': dropped_f_label,
            'flag_test': flag_test_dict,
            'loss': {
                'loss_total': loss,
                'loss_f': loss_f.item(),
                'loss_ppc': loss_ppc.item(),
            }
        }
        if label_indices.shape[0] > 0:
            loss_info['acc_f'] = thres_acc_dict
            loss_info['recall_f'] = thres_recall_dict

        return loss, loss_info
