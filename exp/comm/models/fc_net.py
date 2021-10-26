from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tl2.proj.fvcore import MODEL_REGISTRY, build_model
from tl2.proj.stylegan2_ada import persistence
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils
from tl2.proj.pytorch import init_func


class FCNet(nn.Module):

  def __init__(self,
               input_dim,
               hidden_dim,
               hidden_layers,
               rgb_dim=3,
               device=None,
               name_prefix='fc',
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super().__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.name_prefix = name_prefix

    _out_dim = input_dim

    network = []
    for i in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim

      network.append(nn.Linear(in_features=_in_dim, out_features=_out_dim))
      network.append(nn.LeakyReLU(0.2, inplace=True))

    if len(network) > 0:
      self.network = nn.Sequential(*network)
      self.network.apply(init_func.kaiming_leaky_init)
    else:
      self.network = nn.Identity()

    self.to_rbg = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    torch_utils.print_number_params({
      'network': self.network,
      'to_rbg': self.to_rbg,
      'fc_net': self})
    pass

  def forward(self,
              input,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param kwargs:
    :return:

    """

    x = input

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.network,
                                   inputs_args=(x, ),
                                   name_prefix=f"{self.name_prefix}.network.")
    x = self.network(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.to_rbg,
                                   inputs_args=(x, ),
                                   name_prefix='to_rgb.')
    out = self.to_rbg(x)
    return out

