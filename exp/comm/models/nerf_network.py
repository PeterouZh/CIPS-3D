from collections import OrderedDict
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
from exp.comm.models.cond_layer_norm import CLN

from exp.comm import comm_utils
from exp.comm.models.film_layer import FiLMLayer


def first_layer_film_sine_init(m):
  with torch.no_grad():
    if isinstance(m, nn.Linear):
      num_input = m.weight.size(-1)
      m.weight.uniform_(-1 / num_input, 1 / num_input)


def frequency_init(freq):
  def init(m):
    with torch.no_grad():
      if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

  return init


class UniformBoxWarp(nn.Module):
  def __init__(self, sidelength):
    super().__init__()
    self.scale_factor = 2 / sidelength

  def forward(self, coordinates):
    return coordinates * self.scale_factor


@MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetwork(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               rgb_dim=3,
               style_dim=512,
               hidden_layers=2,
               device=None,
               name_prefix='nerf',
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
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.style_dim = style_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    # self.network = nn.ModuleList([
    #   FiLMLayer(3, hidden_dim),
    #   # FiLMLayer(dim_xyz_emb, hidden_dim),
    #   FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    #   # FiLMLayer(hidden_dim, hidden_dim),
    # ])

    self.network = nn.ModuleList()
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim
      _film_layer = FiLMLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim,
                              use_style_fc=True)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= hidden_dim
    _out_dim = hidden_dim // 2
    self.color_layer_sine = FiLMLayer(in_dim=_in_dim,
                                      out_dim=_out_dim,
                                      style_dim=style_dim,
                                      use_style_fc=True)
    # self.color_layer_sine.apply(frequency_init(25))
    self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

    self.print_number_params()
    pass

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            style_dict,
                                            **kwargs):
    """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(nn.Sequential(
        OrderedDict([
          ('gridwarper', self.gridwarper),
          # ('xyz_emb', self.xyz_emb),
        ])),
        inputs_args=(input,),
        name_prefix="xyz.")
    input = self.gridwarper(input)
    # xyz_emb = self.xyz_emb(input)
    # x = xyz_emb
    x = input

    for index, layer in enumerate(self.network):
      style = style_dict[f'{self.name_prefix}_w{index}']

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"network.{index}.")
      x = layer(x, style)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix="final_layer")
    sigma = self.final_layer(x)

    # rgb branch
    style = style_dict[f'{self.name_prefix}_rgb']
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(x, style),
                                   name_prefix=f"color_layer_sine.")
    x = self.color_layer_sine(x, style)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_linear,
                                   inputs_args=(x,),
                                   name_prefix='color_layer_linear.')
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    rbg = self.color_layer_linear(x)

    out = torch.cat([rbg, sigma], dim=-1)
    return out

  def forward(self,
              input,
              style_dict,
              ray_directions,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param style_dict:
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: (b, num_points, 4), rgb(3) + sigma(1)
    """

    out = self.forward_with_frequencies_phase_shifts(
      input=input,
      style_dict=style_dict,
      ray_directions=ray_directions,
      **kwargs)

    return out

  def print_number_params(self):
    print()
    torch_utils.print_number_params({
                                     'network': self.network,
                                     'final_layer': self.final_layer,
                                     'color_layer_sine': self.color_layer_sine,
                                     'color_layer_linear': self.color_layer_linear,
                                     'nerf_net': self})
    pass

  def get_freq_phase(self, style_dict, name):
    styles = style_dict[name]
    styles = rearrange(styles, "b (n d) -> b d n", n=2)
    frequencies, phase_shifts = styles.unbind(-1)
    frequencies = frequencies * 15 + 30
    return frequencies, phase_shifts

  def staged_forward(self,
                     transformed_points,
                     transformed_ray_directions_expanded,
                     style_dict,
                     max_points,
                     num_steps,
                     ):

    batch_size, num_points, _ = transformed_points.shape

    rgb_sigma_output = torch.zeros((batch_size, num_points, self.rgb_dim + 1),
                                   device=self.device)
    for b in range(batch_size):
      head = 0
      while head < num_points:
        tail = head + max_points
        rgb_sigma_output[b:b + 1, head:tail] = self(
          input=transformed_points[b:b + 1, head:tail],  # (b, h x w x s, 3)
          style_dict={name: style[b:b + 1] for name, style in style_dict.items()},
          ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
        head += max_points
    rgb_sigma_output = rearrange(rgb_sigma_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)
    return rgb_sigma_output


@MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetworkL(NeRFNetwork):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               rgb_dim=3,
               style_dim=512,
               hidden_layers=2,
               device=None,
               name_prefix='nerf',
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super(NeRFNetwork, self).__init__()

    self.device = device
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.style_dim = style_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    self.network = nn.ModuleList()
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim
      _film_layer = FiLMLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim,
                              use_style_fc=True)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= hidden_dim
    # _out_dim = hidden_dim // 2
    _out_dim = hidden_dim
    self.color_layer_sine = FiLMLayer(in_dim=_in_dim,
                                      out_dim=_out_dim,
                                      style_dim=style_dim,
                                      use_style_fc=True)
    # self.color_layer_sine.apply(frequency_init(25))
    self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

    self.print_number_params()
    pass


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


@MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetwork_CLN(NeRFNetwork):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               rgb_dim=3,
               style_dim=512,
               hidden_layers=2,
               device=None,
               name_prefix='nerf',
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super(NeRFNetwork, self).__init__()

    self.device = device
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.style_dim = style_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    self.network = nn.ModuleList()
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim
      if idx == 0:
        _film_layer = FiLMLayer(in_dim=_in_dim,
                                out_dim=_out_dim,
                                style_dim=style_dim,
                                use_style_fc=True)
        self.network.append(_film_layer)
        self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
      else:
        _cln_layer = CLNLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim,
                              name_prefix=f'{name_prefix}_w{idx}')
        self.network.append(_cln_layer)
        self.style_dim_dict.update(_cln_layer.style_dim_dict)
    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= hidden_dim
    # _out_dim = hidden_dim // 2
    _out_dim = hidden_dim
    # self.color_layer_sine = FiLMLayer(in_dim=_in_dim,
    #                                   out_dim=_out_dim,
    #                                   style_dim=style_dim,
    #                                   use_style_fc=True)
    # # self.color_layer_sine.apply(frequency_init(25))
    # self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
    self.color_layer_sine = CLNLayer(in_dim=_in_dim,
                                     out_dim=_out_dim,
                                     style_dim=style_dim,
                                     name_prefix=f'{name_prefix}_rgb')
    self.style_dim_dict.update(self.color_layer_sine.style_dim_dict)

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

    self.print_number_params()
    pass

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            style_dict,
                                            **kwargs):
    """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(nn.Sequential(
        OrderedDict([
          ('gridwarper', self.gridwarper),
          # ('xyz_emb', self.xyz_emb),
        ])),
        inputs_args=(input,),
        name_prefix="xyz.")
    input = self.gridwarper(input)
    # xyz_emb = self.xyz_emb(input)
    # x = xyz_emb
    x = input

    for index, layer in enumerate(self.network):

      if index == 0:
        style = style_dict[f'{self.name_prefix}_w{index}']
        if global_cfg.tl_debug:
          VerboseModel.forward_verbose(layer,
                                       inputs_args=(x, style),
                                       name_prefix=f"network.{index}.")
        x = layer(x, style)
      else:
        if global_cfg.tl_debug:
          VerboseModel.forward_verbose(layer,
                                       inputs_args=(x, style_dict),
                                       name_prefix=f"network.{index}.")
        x = layer(x, style_dict)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix="final_layer")
    sigma = self.final_layer(x)

    # rgb branch
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(x, style_dict),
                                   name_prefix=f"color_layer_sine.")
    x = self.color_layer_sine(x, style_dict)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_linear,
                                   inputs_args=(x,),
                                   name_prefix='color_layer_linear.')
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    rbg = self.color_layer_linear(x)

    out = torch.cat([rbg, sigma], dim=-1)
    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetwork_Small(NeRFNetwork):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               rgb_dim=3,
               style_dim=512,
               hidden_layers=2,
               device=None,
               name_prefix='nerf',
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super(NeRFNetwork, self).__init__()

    self.device = device
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.style_dim = style_dim
    self.hidden_layers = hidden_layers
    self.name_prefix = name_prefix

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    self.network = nn.ModuleList()
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim
      _film_layer = FiLMLayer(in_dim=_in_dim,
                              out_dim=_out_dim,
                              style_dim=style_dim,
                              use_style_fc=True)
      self.network.append(_film_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _film_layer.style_dim
    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= hidden_dim
    # _out_dim = hidden_dim // 2
    _out_dim = hidden_dim
    self.color_layer_sine = nn.Identity()
    # self.color_layer_sine = FiLMLayer(in_dim=_in_dim,
    #                                   out_dim=_out_dim,
    #                                   style_dim=style_dim,
    #                                   use_style_fc=True)
    # # self.color_layer_sine.apply(frequency_init(25))
    # self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

    self.print_number_params()
    pass

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            style_dict,
                                            **kwargs):
    """

    :param input: (b, n, 3)
    :param style_dict:
    :param ray_directions:
    :param kwargs:
    :return:
    """

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(nn.Sequential(
        OrderedDict([
          ('gridwarper', self.gridwarper),
          # ('xyz_emb', self.xyz_emb),
        ])),
        inputs_args=(input,),
        name_prefix="xyz.")
    input = self.gridwarper(input)
    # xyz_emb = self.xyz_emb(input)
    # x = xyz_emb
    x = input

    for index, layer in enumerate(self.network):
      style = style_dict[f'{self.name_prefix}_w{index}']

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"network.{index}.")
      x = layer(x, style)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix="final_layer")
    sigma = self.final_layer(x)

    # rgb branch
    # style = style_dict[f'{self.name_prefix}_rgb']
    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.color_layer_sine,
    #                                inputs_args=(x, style),
    #                                name_prefix=f"color_layer_sine.")
    # x = self.color_layer_sine(x, style)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_linear,
                                   inputs_args=(x,),
                                   name_prefix='color_layer_linear.')
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    rbg = self.color_layer_linear(x)

    out = torch.cat([rbg, sigma], dim=-1)
    return out
