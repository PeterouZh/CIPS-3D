import collections
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tl2.proj.fvcore import MODEL_REGISTRY, build_model
from tl2.proj.stylegan2_ada import persistence
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils, init_func

from exp.comm.models.film_layer import FiLMLayer
from exp.comm.models.cond_layer_norm import CLN


def frequency_init(freq):
  def init(m):
    with torch.no_grad():
      if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

  return init


class INRNetwork(nn.Module):

  def __init__(self,
               input_dim,
               hidden_dim,
               style_dim,
               hidden_layers,
               rgb_dim=3,
               device=None,
               name_prefix='inr',
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

    self.style_dim_dict = {}

    _out_dim = input_dim

    self.network = nn.ModuleList()
    for i in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim

      _film_layer = FiLMLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{i}'] = _film_layer.style_dim
    # self.network.apply(frequency_init(25))

    self.to_rbg = nn.Sequential(
      nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    self.dim_styles = sum(self.style_dim_dict.values())

    self.print_number_params()
    pass

  def forward(self,
              input,
              style_dict,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    x = input

    for index, layer in enumerate(self.network):
      style = style_dict[f'{self.name_prefix}_w{index}']

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"{self.name_prefix}.network.{index}.")
      x = layer(x, style)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.to_rbg,
                                   inputs_args=(x, ),
                                   name_prefix='to_rgb.')
    out = self.to_rbg(x)
    return out

  def get_freq_phase(self, style_dict, name):
    styles = style_dict[name]
    styles = rearrange(styles, "b (n d) -> b d n", n=2)
    frequencies, phase_shifts = styles.unbind(-1)
    frequencies = frequencies * 15 + 30
    return frequencies, phase_shifts

  def print_number_params(self):
    print()
    torch_utils.print_number_params({
                                     'network': self.network,
                                     'to_rbg': self.to_rbg,
                                     'inr_net': self})
    pass


class ToRGB(nn.Module):
  def __init__(self,
               in_dim,
               dim_rgb=3):
    super().__init__()
    self.in_dim = in_dim
    self.dim_rgb = dim_rgb

    self.linear = nn.Linear(in_dim, dim_rgb)
    pass

  def forward(self,
              input,
              skip=None):

    out = self.linear(input)

    if skip is not None:
      out = out + skip
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"dim_rgb={self.dim_rgb})"
    return repr


class INRNetwork_Skip(nn.Module):

  def __init__(self,
               input_dim,
               style_dim,
               hidden_layers,
               dim_scale=1,
               rgb_dim=3,
               device=None,
               name_prefix='inr',
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
    self.rgb_dim = rgb_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix

    self.channels = {
      0: int(512 * dim_scale),  # 4
      1: int(512 * dim_scale),  # 8
      2: int(512 * dim_scale),  # 16
      3: int(512 * dim_scale),  # 32
      4: int(512 * dim_scale),  # 64
      5: int(128 * dim_scale),  # 128
      6: int(64 * dim_scale),  # 256
      7: int(32 * dim_scale),  # 512
      8: int(16 * dim_scale),  # 1024
    }

    self.style_dim_dict = {}

    _out_dim = input_dim

    self.network = nn.ModuleList()
    self.to_rbgs = nn.ModuleList()
    for i in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = self.channels[i]

      _film_layer = FiLMLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{i}_0'] = _film_layer.style_dim

      _film_layer = FiLMLayer(in_dim=_out_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{i}_1'] = _film_layer.style_dim

      to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=3)
      self.to_rbgs.append(to_rgb)

    # self.network.apply(frequency_init(25))

    self.tanh = nn.Sequential(
      # nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    self.dim_styles = sum(self.style_dim_dict.values())

    torch_utils.print_number_params(
      {
        'network': self.network,
        'to_rbgs': self.to_rbgs,
        'inr_net': self
      })
    pass

  def forward(self,
              input,
              style_dict,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    x = input
    rgb = 0
    for index in range(self.hidden_layers):

      _layer = self.network[index * 2]
      style = style_dict[f'{self.name_prefix}_w{index}_0']

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(_layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"{self.name_prefix}.network.{index}.0.")
      x = _layer(x, style)

      _layer = self.network[index * 2 + 1]
      style = style_dict[f'{self.name_prefix}_w{index}_1']

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(_layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"{self.name_prefix}.network.{index}.1.")
      x = _layer(x, style)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.to_rbgs[index],
                                     inputs_args=(x, rgb),
                                     name_prefix=f'to_rgb.{index}')
      rgb = self.to_rbgs[index](x, skip=rgb)

    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.to_rbg,
    #                                inputs_args=(x, ),
    #                                name_prefix='to_rgb.')
    # out = self.to_rbg(x)
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.tanh,
                                   inputs_args=(rgb, ),
                                   name_prefix='tanh.')
    out = self.tanh(rgb)
    return out


class FiLMBlock(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               name_prefix,
               ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim
    self.name_prefix = name_prefix

    self.style_dim_dict = {}

    self.film1 = FiLMLayer(in_dim=in_dim,
                           out_dim=out_dim,
                           style_dim=style_dim)

    self.style_dim_dict[f'{name_prefix}_0'] = self.film1.style_dim

    self.film2 = FiLMLayer(in_dim=out_dim,
                           out_dim=out_dim,
                           style_dim=style_dim)
    self.style_dim_dict[f'{name_prefix}_1'] = self.film2.style_dim

    pass

  def forward(self,
              x,
              style_dict,
              skip=False):
    x_orig = x

    style = style_dict[f'{self.name_prefix}_0']
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.film1,
                                   inputs_args=(x, style),
                                   name_prefix=f"{self.name_prefix}.film1.")
    x = self.film1(x, style)

    style = style_dict[f'{self.name_prefix}_1']
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.film2,
                                   inputs_args=(x, style),
                                   name_prefix=f"{self.name_prefix}.film2.")
    out = self.film2(x, style)
    if skip and out.shape[-1] == x_orig.shape[-1]:
      out = out + x_orig
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim}, " \
           f"style_dim={self.style_dim})"
    return repr


class LinearBlock(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               name_prefix,
               ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.name_prefix = name_prefix

    self.net = nn.Sequential(
      nn.Linear(in_features=in_dim, out_features=out_dim),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(in_features=out_dim, out_features=out_dim),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.net.apply(init_func.kaiming_leaky_init)

    pass

  def forward(self,
              x,
              *args,
              **kwargs):

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.net,
                                   inputs_args=(x, ),
                                   name_prefix=f"{self.name_prefix}.net.")
    out = self.net(x)
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim})"
    return repr


class INRNetwork_Skip_Prog(nn.Module):

  def __init__(self,
               input_dim,
               style_dim,
               dim_scale=1,
               rgb_dim=3,
               device=None,
               name_prefix='inr',
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
    self.rgb_dim = rgb_dim
    self.name_prefix = name_prefix

    self.channels = {
      "16": int(256 * dim_scale),
      "32": int(256 * dim_scale),
      "64": int(256 * dim_scale),
      "128": int(256 * dim_scale),
      "256": int(256 * dim_scale),
      "512": int(256 * dim_scale),
      "1024": int(256 * dim_scale),
    }

    self.style_dim_dict = {}

    _out_dim = input_dim

    network = collections.OrderedDict()
    to_rbgs = collections.OrderedDict()
    for i, (name, channel) in enumerate(self.channels.items()):
      _in_dim = _out_dim
      _out_dim = channel

      if name.startswith(('none', )):
        _linear_block = LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
        network[name] = _linear_block
      else:
        _film_block = FiLMBlock(in_dim=_in_dim,
                                out_dim=_out_dim,
                                style_dim=style_dim,
                                name_prefix=f'{name_prefix}_w{name}')
        self.style_dim_dict.update(_film_block.style_dim_dict)
        network[name] = _film_block

      _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=3)
      to_rbgs[name] = _to_rgb

    self.network = nn.ModuleDict(network)
    self.to_rgbs = nn.ModuleDict(to_rbgs)
    self.to_rgbs.apply(frequency_init(100))

    self.tanh = nn.Sequential(
      # nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    self.dim_styles = sum(self.style_dim_dict.values())

    torch_utils.print_number_params(
      {
        'network': self.network,
        'to_rbgs': self.to_rgbs,
        'inr_net': self
      })
    pass

  def forward(self,
              input,
              style_dict,
              img_size,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    x = input
    img_size = str(2 ** int(np.log2(img_size)))

    rgb = 0
    for name, block in self.network.items():
      x = block(x, style_dict, skip=True)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.to_rgbs[name],
                                     inputs_args=(x, rgb),
                                     name_prefix=f'to_rgb.{name}.')
      rgb = self.to_rgbs[name](x, skip=rgb)
      if name == img_size:
        break

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.tanh,
                                   inputs_args=(rgb, ),
                                   name_prefix='tanh.')
    out = self.tanh(rgb)
    return out


class CLNBlock(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               name_prefix,
               ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim
    self.name_prefix = name_prefix

    self.style_dim_dict = {}

    self.linear1 = nn.Linear(in_dim, out_dim)
    self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    self.style_dim_dict[f'{name_prefix}_0'] = self.cln1.style_dim
    self.act1 = nn.LeakyReLU(0.2, inplace=True)

    self.linear2 = nn.Linear(out_dim, out_dim)
    self.cln2 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    self.style_dim_dict[f'{name_prefix}_1'] = self.cln2.style_dim
    self.act2 = nn.LeakyReLU(0.2, inplace=True)

    pass

  def forward(self,
              x,
              style_dict,
              skip=False):
    x_orig = x

    x = self.linear1(x)
    style = style_dict[f'{self.name_prefix}_0']
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.cln1,
                                   inputs_args=(x, style),
                                   name_prefix=f"{self.name_prefix}.cln1.")
    x = self.cln1(x, style)
    x = self.act1(x)

    x = self.linear2(x)
    style = style_dict[f'{self.name_prefix}_1']
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.cln2,
                                   inputs_args=(x, style),
                                   name_prefix=f"{self.name_prefix}.cln2.")
    x = self.cln2(x, style)
    out = self.act2(x)

    if skip and out.shape[-1] == x_orig.shape[-1]:
      out = out + x_orig
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim}, " \
           f"style_dim={self.style_dim})"
    return repr


class INRNetwork_Skip_CLN(nn.Module):

  def __init__(self,
               input_dim,
               style_dim,
               dim_scale=1,
               rgb_dim=3,
               device=None,
               name_prefix='inr',
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
    self.rgb_dim = rgb_dim
    self.name_prefix = name_prefix

    self.channels = {
      "16": int(256 * dim_scale),
      "32": int(256 * dim_scale),
      "64": int(256 * dim_scale),
      "128": int(256 * dim_scale),
      "256": int(256 * dim_scale),
      "512": int(256 * dim_scale),
      "1024": int(256 * dim_scale),
    }

    self.style_dim_dict = {}

    _out_dim = input_dim

    network = collections.OrderedDict()
    to_rbgs = collections.OrderedDict()
    for i, (name, channel) in enumerate(self.channels.items()):
      _in_dim = _out_dim
      _out_dim = channel

      if name.startswith(('none', )):
        _linear_block = LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
        network[name] = _linear_block
      else:
        _film_block = CLNBlock(in_dim=_in_dim,
                               out_dim=_out_dim,
                               style_dim=style_dim,
                               name_prefix=f'{name_prefix}_w{name}')
        self.style_dim_dict.update(_film_block.style_dim_dict)
        network[name] = _film_block

      _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=3)
      to_rbgs[name] = _to_rgb

    self.network = nn.ModuleDict(network)
    self.to_rgbs = nn.ModuleDict(to_rbgs)
    self.to_rgbs.apply(frequency_init(100))

    self.tanh = nn.Sequential(
      # nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    self.dim_styles = sum(self.style_dim_dict.values())

    torch_utils.print_number_params(
      {
        'network': self.network,
        'to_rbgs': self.to_rgbs,
        'inr_net': self
      })
    pass

  def forward(self,
              input,
              style_dict,
              img_size,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    x = input
    img_size = str(2 ** int(np.log2(img_size)))

    rgb = 0
    for name, block in self.network.items():
      skip = int(name) >= 128
      x = block(x, style_dict, skip=skip)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.to_rgbs[name],
                                     inputs_args=(x, rgb),
                                     name_prefix=f'to_rgb.{name}.')
      rgb = self.to_rgbs[name](x, skip=rgb)
      if name == img_size:
        break

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.tanh,
                                   inputs_args=(rgb, ),
                                   name_prefix='tanh.')
    out = self.tanh(rgb)
    return out


class CLNLayer(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               name_prefix,
               ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim
    self.name_prefix = name_prefix

    self.style_dim_dict = {}

    self.linear1 = nn.Linear(in_dim, out_dim)
    self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    self.style_dim_dict[f'{name_prefix}'] = self.cln1.style_dim
    self.act1 = nn.LeakyReLU(0.2, inplace=True)

    pass

  def forward(self,
              x,
              style_dict,
              skip=False):
    x_orig = x

    x = self.linear1(x)
    style = style_dict[f'{self.name_prefix}']
    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.cln1,
    #                                inputs_args=(x, style),
    #                                name_prefix=f"{self.name_prefix}.cln1.")
    x = self.cln1(x, style)
    x = self.act1(x)

    return x

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim}, " \
           f"style_dim={self.style_dim})"
    return repr


class Linear_Skip_Prog(nn.Module):

  def __init__(self,
               input_dim,
               hidden_dim,
               rgb_dim=3,
               device=None,
               name_prefix='linear',
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
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.name_prefix = name_prefix

    self.channels = {
      # "2": int(256 * dim_scale),
      # "4": int(256 * dim_scale),
      # "8": int(256 * dim_scale),
      # "16": int(256 * dim_scale),
      "32": int(hidden_dim),
      "64": int(hidden_dim),
      "128": int(hidden_dim),
      "256": int(hidden_dim),
      "512": int(hidden_dim),
      "1024": int(hidden_dim),
    }

    _in_dim = input_dim
    _out_dim = hidden_dim

    self.cln_layer = CLNLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=256,
                              name_prefix=f"{name_prefix}_w")
    self.style_dim_dict = self.cln_layer.style_dim_dict

    network = collections.OrderedDict()
    to_rbgs = collections.OrderedDict()
    for i, (name, channel) in enumerate(self.channels.items()):
      _in_dim = _out_dim
      _out_dim = channel

      # _linear_block = LinearBlock(in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
      _linear_block = nn.Sequential(
        nn.Linear(in_features=_in_dim, out_features=_out_dim),
        nn.LeakyReLU(0.2, inplace=True),
      )
      network[name] = _linear_block

      _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=self.rgb_dim)
      to_rbgs[name] = _to_rgb

    self.network = nn.ModuleDict(network)
    self.network.apply(frequency_init(25))

    self.to_rgbs = nn.ModuleDict(to_rbgs)

    self.tanh = nn.Sequential(
      # nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    torch_utils.print_number_params(
      {
        'network': self.network,
        'to_rbgs': self.to_rgbs,
        'linear_net': self
      })
    pass

  def forward(self,
              input,
              img_size,
              style_dict,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    x = input
    img_size = str(2 ** int(np.log2(img_size)))

    x = self.cln_layer(x, style_dict)

    rgb = 0
    for name, block in self.network.items():
      x = block(x)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.to_rgbs[name],
                                     inputs_args=(x, rgb),
                                     name_prefix=f'to_rgb.{name}.')
      rgb = self.to_rgbs[name](x, skip=rgb)
      if name == img_size:
        break

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.tanh,
                                   inputs_args=(rgb, ),
                                   name_prefix='tanh.')
    out = self.tanh(rgb)
    return out










