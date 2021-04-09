import torch


def indices_to_coors(voxel_indices, voxel_size, offset):
    """
    transform voxel indices to actual coordinates
    
    Args:
        voxel_indices : indices to be transformed, in N x 4 (B, Z, Y, X) ***[torch.Tensor]***
        voxel_size : size of a single voxel, in [Z, Y, X] ***[python built-in list]***
        offset : lower boundary of the space, in [Z, Y, X] ***[python built-in list]***

    """
    assert isinstance(voxel_indices, torch.Tensor), "invalid type of voxel_indices"
    assert len(voxel_indices.shape) == 2 and voxel_indices.shape[1] == 4, "wrong shape of voxel_indices"
    assert isinstance(voxel_size, list) and len(voxel_size) == 3, "invalid voxel_size"
    assert isinstance(offset, list) and len(offset) == 3, "invalid offset"

    res = voxel_indices.float()
    res[:, 1:] += 0.5
    res[:, 1:] *= torch.tensor(voxel_size, dtype=res.dtype, device=res.device)
    res[:, 1:] += torch.tensor(offset, dtype=res.dtype, device=res.device)
    return res


def coors_to_pixels(coors, calib, pixel_refinement=None, post_process='round'):
    """
    project LiDAR coordinates to image plane
    
    Args:
        coors : coordinates to be projected, in N x 4 (B, Z, Y, X) ***[torch.Tensor]***
        calib : sensor calibration information ***[python built-in dict]***
        pixel_refinement : in case of using cropped and scaled rgb images ***None or [python built-in list]***
        post_process : use torch.round(), .int(), or do nothing to the result before return ***['round', 'int', 'none']***

    """
    assert isinstance(coors, torch.Tensor), "invalid type of coors"
    assert len(coors.shape) == 2 and coors.shape[1] == 4, "wrong shape of coors"
    assert isinstance(calib, dict), "invalid type of calib"
    assert pixel_refinement is None or isinstance(pixel_refinement, list), "invalid pixel_refinement"
    assert calib['P2'].shape[0] == coors[:, 0].max() + 1, "batch sizes of coors and calib do not match"

    # get batch size
    batch_size = calib['P2'].shape[0]

    # transform to (B, X, Y, Z)
    res = coors[:, [0, 3, 2, 1]]

    # for each batch
    for batch in range(batch_size):
        # Projection matrix from rect camera coord to image2 coord
        P = calib['P2'][batch].float()
        P = P[:3, :4]
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calib['Trv2c'][batch].float()
        V2C = V2C[:3, :4]
        # Rotation from reference camera coord to rect camera coord
        R0 = calib['rect'][batch].float()
        R0 = R0[:3, :3]

        # get pts indexes of current batch
        pts_indexes = res[:, 0] == batch
        # get (X, Y, Z) of pts
        pts_3d_velo = res[pts_indexes, 1:]

        # velo_to_ref
        pts_3d_velo = torch.cat((pts_3d_velo, torch.ones(pts_3d_velo.shape[0], 1, device=pts_3d_velo.device)),
                                dim=1)  # nx3 -> nx4
        pts_3d_ref = torch.mm(pts_3d_velo, V2C.t())

        # ref_to_rect
        pts_3d_rect = torch.mm(R0, pts_3d_ref.t()).t()

        # rect_to_image
        pts_3d_rect = torch.cat((pts_3d_rect, torch.ones(pts_3d_rect.shape[0], 1, device=pts_3d_rect.device)),
                                dim=1)  # nx3 -> nx4
        pts_2d = torch.mm(pts_3d_rect, P.t())  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        pts_2d = pts_2d[:, 0:2]

        # adjust pixels' (u, v)
        if pixel_refinement:
            pts_2d[:, 0] -= pixel_refinement[batch]['x_offset']
            pts_2d[:, 1] -= pixel_refinement[batch]['y_offset']
            pts_2d *= pixel_refinement[batch]['resize_scale']

        # save (u, v) coordinates
        res[pts_indexes, 1:3] = pts_2d

    # recover shape
    res = res[:, :3]

    # post process the coordinates
    if post_process == 'round':
        res[:, 1:] = torch.round(res[:, 1:])
    elif post_process == 'int':
        res[:, 1:] = res[:, 1:].int()
    elif post_process == 'none':
        pass
    else:
        raise NotImplementedError

    return res


def _inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = torch.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = Tr[0:3, 0:3].t()
    inv_Tr[0:3, [3]] = torch.mm(-Tr[0:3, 0:3].t(), Tr[0:3, [3]])
    return inv_Tr
