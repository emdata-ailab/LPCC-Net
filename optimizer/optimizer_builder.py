# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions to build DetectionModel training optimizers."""

#############################################
# copied and modified (from project SECOND) #
#############################################

import torch
from torch import nn
from torchplus.train.fastai_optim import OptimWrapper
from functools import partial


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]

get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]


def build(optimizer_config, net, name=None):
    """Create optimizer based on config.

  Args:
    optimizer_config: An optimizer configparser object.
    net: Network model
    name: Assign a name to optimizer for checkpoint system

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config['type']
    optimizer_func = None

    if optimizer_type == 'rms_prop':
        decay = float(optimizer_config['decay'])
        momentum_optimizer_value = float(optimizer_config['momentum_optimizer_value'])
        epsilon = float(optimizer_config['epsilon'])
        optimizer_func = partial(
            torch.optim.RMSprop,
            alpha=decay,
            momentum=momentum_optimizer_value,
            eps=epsilon)

    if optimizer_type == 'momentum':
        momentum_optimizer_value = float(optimizer_config['momentum_optimizer_value'])
        epsilon = float(optimizer_config['epsilon'])
        optimizer_func = partial(
            torch.optim.SGD,
            momentum=momentum_optimizer_value,
            eps=epsilon)

    if optimizer_type == 'adam':
        amsgrad = optimizer_config.getboolean('amsgrad')
        if optimizer_config.getboolean('fixed_weight_decay'):
            optimizer_func = partial(
                torch.optim.Adam, betas=(0.9, 0.99), amsgrad=amsgrad)
        else:
            # regular adam
            optimizer_func = partial(
                torch.optim.Adam, amsgrad=amsgrad)

    if optimizer_func is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        wd=float(optimizer_config['weight_decay']),
        true_wd=optimizer_config.getboolean('fixed_weight_decay'),
        bn_wd=True)

    if name is None:
        optimizer.name = optimizer_type
    else:
        optimizer.name = name

    return optimizer
