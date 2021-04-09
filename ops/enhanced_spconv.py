import numpy as np
import torch
import spconv


def concat_sp_tensors(*tensors, merge_concat=False, feature_concat=False):
    """concat several sparse tensors.
    """
    assert len(tensors) > 0, "expect at least one tensor"

    # nothing to do, if there is only one tensor
    if len(tensors) == 1:
        return tensors[0]

    # sort tensors by number of active voxels
    tensors = np.array(tensors)
    tensor_lens = np.zeros(len(tensors), dtype=np.int32)
    for i in range(len(tensors)):
        assert isinstance(tensors[i], spconv.SparseConvTensor), "input must be SparseConvTensor"
        tensor_lens[i] = len(tensors[i].indices)
    tensors = tensors[tensor_lens.argsort()]

    # concat tensors
    _concat_helper = _merge_concat_helper if merge_concat else _simple_concat_helper
    base, *ts = tensors
    for t in ts:
        base = _concat_helper(base, t, feature_concat)
    return base


def _merge_concat_helper(spt1, spt2, feature_concat=False):
    """concat two sparse tensors with different active indices, the indice_dict of output is ignored.
    """
    assert feature_concat == True or spt1.features.shape[1] == spt2.features.shape[1], \
        "length of features must match when feature_concat == False"
    assert all(s1 == s2 for s1, s2 in zip(spt1.spatial_shape, spt2.spatial_shape)), \
        "spatial shape of tensors must match"
    assert spt1.batch_size == spt2.batch_size, "batch size of tensors must match"

    # resolve indices
    indices_concat = torch.cat((spt1.indices, spt2.indices))
    indices_unique, inverse_index, counts = torch.unique(indices_concat, dim=0, return_inverse=True, return_counts=True)
    indices = indices_unique

    # resolve features
    if feature_concat:
        features = torch.zeros(len(indices_unique),
                               spt1.features.shape[1] + spt2.features.shape[1],
                               dtype=spt1.features.dtype,
                               device=spt1.features.device)
        features[inverse_index[:spt1.features.shape[0]], :spt1.features.shape[1]] = spt1.features
        features[inverse_index[spt1.features.shape[0]:], spt1.features.shape[1]:] = spt2.features
    else:
        features = torch.zeros(len(indices_unique),
                               spt1.features.shape[1],
                               dtype=spt1.features.dtype,
                               device=spt1.features.device)
        features[inverse_index[:spt1.features.shape[0]]] += spt1.features
        features[inverse_index[spt1.features.shape[0]:]] += spt2.features
        # features[counts == 2] /= 2.0 # should features be averaged???

    spatial_shape = spt1.spatial_shape
    batch_size = spt1.batch_size

    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)


def _simple_concat_helper(spt1, spt2, feature_concat=False):
    """concat two sparse tensors with exactly the same active indices, the indice_dicts are combined.
    """
    assert torch.equal(spt1.indices, spt2.indices), "indices of the input sp tensors should be exactly the same"
    assert feature_concat == True or spt1.features.shape[1] == spt2.features.shape[1], \
        "length of features must match when feature_concat == False"
    assert all(s1 == s2 for s1, s2 in zip(spt1.spatial_shape, spt2.spatial_shape)), \
        "spatial shape of tensors must match"
    assert spt1.batch_size == spt2.batch_size, "batch size of tensors must match"

    indices = spt1.indices

    if feature_concat:
        features = torch.cat((spt1.features, spt2.features), dim=1)
    else:
        features = spt1.features + spt2.features

    spatial_shape = spt1.spatial_shape
    batch_size = spt1.batch_size

    res = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
    res.indice_dict = spt1.indice_dict
    res.indice_dict.update(spt2.indice_dict)

    return res


def simple_sp_tensor_expansion(sp_tensor, kernel_size, padding, feature_fill=0, computation_device='cpu'):
    """expand active indices of a sparse convolution tensor, the features of expanded indices are filled with
    feature_fill.
    """
    assert isinstance(sp_tensor, spconv.SparseConvTensor), "sp_tensor must be spconv.SparseConvTensor"
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size] * 3
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * 3
    assert all(k == p * 2 + 1 for k, p in zip(kernel_size, padding)), "only support this"

    # get required info.
    to_be_expanded = sp_tensor.indices.to(computation_device)
    batch_size = sp_tensor.batch_size
    spatial_shape = sp_tensor.spatial_shape
    input_device = sp_tensor.indices.device

    # get expanded indices
    expanded, _, _ = spconv.ops.get_indice_pairs(to_be_expanded, batch_size, spatial_shape, ksize=kernel_size,
                                                 padding=padding)

    # concat with current active indices
    concatenated = torch.cat((to_be_expanded.to(input_device), expanded.to(input_device)), dim=0)

    # put features back
    unique_indices, i_index = torch.unique(concatenated, return_inverse=True, dim=0)
    features = torch.zeros(unique_indices.shape[0], sp_tensor.features.shape[1], dtype=sp_tensor.features.dtype,
                           device=input_device)
    features = features.fill_(feature_fill)
    features[i_index[:sp_tensor.features.shape[0]]] = sp_tensor.features

    return spconv.SparseConvTensor(features, unique_indices, spatial_shape, batch_size)
