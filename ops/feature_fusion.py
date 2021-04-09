import torch
import spconv

from . import transform


def fuse_rgb_to_voxel(sp_tensor, voxel_size, offset, rgb_features, calib, downsample_factor=1, pixel_refinement=None,
                      feature_concat=True):
    """
    fetch and concat corresponding rgb features to voxels, where original voxels are kept and only features are modified

    Args:
        sp_tensor : sparse tensor ***[spconv.SparseConvTensor]***
        voxel_size : size of each voxel, in (Z, Y, X) ***[python built-in list]***
        offset : lower boundary of the 0th voxel, in (Z, Y, X) ***[python built-in list]***
        rgb_features : rgb features, in (B, C, H, W) ***[torch.Tensor]***
        calib : calibration for projection from liDAR to RGB, includes 'rect', 'Trv2c', and 'P2' ***[python built-in dict]***
        downsample_factor : downsample factor, necessary for recovering correct projection
        pixel_refinement : in case of using cropped and scaled rgb images ***None or [python built-in list]***
        feature_concat : concat voxel and rgb features if True, otherwise conduct element-wise addition
    """

    assert isinstance(sp_tensor, spconv.SparseConvTensor), "sp_tensor must be spconv.SparseConvTensor"
    assert isinstance(voxel_size, list) and len(voxel_size) == 3, "invalid voxel_size"
    assert isinstance(offset, list) and len(offset) == 3, "invalid offset"
    assert isinstance(rgb_features, torch.Tensor) and rgb_features.dim() == 4, "invalid rgb_features"
    assert feature_concat is True or sp_tensor.features.shape[1] == rgb_features.shape[
        1], "length of voxel and rgb features must match when feature_concat == False"

    # modify rgb feature orders
    rgb_features = rgb_features.permute(0, 3, 2, 1)  # (B, C, H, W) -> (B, W, H, C)

    # voxel infos
    indices = sp_tensor.indices
    features = sp_tensor.features
    batch_size = sp_tensor.batch_size
    spatial_shape = sp_tensor.spatial_shape
    indice_dict = sp_tensor.indice_dict

    # project to RGB plane
    scaled_indices = indices * torch.tensor([1] + [downsample_factor] * 3, dtype=indices.dtype,
                                            device=indices.device)  # -> original size for correct projection
    coors = transform.indices_to_coors(scaled_indices, voxel_size, offset)
    pixels = transform.coors_to_pixels(coors, calib, pixel_refinement, 'none')
    pixels = pixels / torch.tensor([1] + [downsample_factor] * 2, dtype=pixels.dtype, device=pixels.device)
    pixels = torch.round(pixels).int()

    # filter outsiders
    index_keep = (pixels[:, 1] >= 0) & (pixels[:, 1] < rgb_features.shape[1]) & \
                 (pixels[:, 2] >= 0) & (pixels[:, 2] < rgb_features.shape[2])

    # get rgb features
    voxel_rgb_features = torch.zeros(features.shape[0], rgb_features.shape[-1], dtype=features.dtype,
                                     device=features.device)
    voxel_rgb_features[index_keep] = rgb_features[tuple(pixels[index_keep].long().transpose(0, 1))]

    # generate output features
    if feature_concat:
        features = torch.cat((features, voxel_rgb_features), dim=1)
    else:
        features = features + voxel_rgb_features

    res = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
    res.indice_dict = indice_dict

    return res
