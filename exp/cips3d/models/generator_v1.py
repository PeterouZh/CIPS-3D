from itertools import chain
import math
import logging
from collections import OrderedDict
import tqdm
import random
import time
# from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tl2.proj.fvcore import MODEL_REGISTRY, build_model
# from tl2.proj.stylegan2_ada import persistence
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils
from tl2.proj.pytorch import torch_utils, init_func
from tl2 import tl2_utils

from exp.pigan import pigan_utils
from exp.dev.nerf_inr.models.generator_nerf_inr import INRNetwork
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base
from exp.comm import comm_utils
from exp.comm.models import nerf_network
from exp.comm.models import inr_network
from exp.comm.models import film_layer
from exp.comm.models import mod_conv_fc
from exp.cips3d.models import multi_head_mapping

class SkipLayer(nn.Module):
  def __init__(self, ):
    super(SkipLayer, self).__init__()

  def forward(self, x0, x1):
    # out = (x0 + x1) / math.pi
    out = (x0 + x1)
    return out



class SinAct(nn.Module):
  def __init__(self, ):
    super(SinAct, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class LinearSinAct(nn.Module):
  def __init__(self,
               in_features,
               out_features):
    super(LinearSinAct, self).__init__()

    self.linear = nn.Linear(in_features=in_features, out_features=out_features)
    self.sin = SinAct()
    pass

  def forward(self, x, *args, **kwargs):
    x = self.linear(x)
    x = self.sin(x)
    return x


class FiLMLayer(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               use_style_fc=True,
               which_linear=nn.Linear,
               **kwargs):
    super(FiLMLayer, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim
    self.use_style_fc = use_style_fc

    self.linear = which_linear(in_dim, out_dim)
    # self.linear.apply(film_layer.frequency_init(25))

    # self.gain_scale = film_layer.LinearScale(scale=15, bias=30)
    self.gain_scale = nn.Identity()
    # Prepare gain and bias layers
    if use_style_fc:
      self.gain_fc = which_linear(style_dim, out_dim)
      self.bias_fc = which_linear(style_dim, out_dim)
      # self.gain_fc.weight.data.mul_(0.25)
      # self.bias_fc.weight.data.mul_(0.25)
    else:
      self.style_dim = out_dim * 2

    self.sin = SinAct()
    self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    # self.register_buffer('stored_mean', torch.zeros(output_size))
    # self.register_buffer('stored_var', torch.ones(output_size))
    pass

  def forward(self,
              x,
              style):
    """

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """

    if self.use_style_fc:
      gain = self.gain_fc(style)
      gain = self.gain_scale(gain)
      bias = self.bias_fc(style)
    else:
      style = rearrange(style, "b (n c) -> b n c", n=2)
      gain, bias = style.unbind(dim=1)
      gain = self.gain_scale(gain)

    if x.dim() == 3:
      gain = rearrange(gain, "b c -> b 1 c")
      bias = rearrange(bias, "b c -> b 1 c")
    elif x.dim() == 2:
      pass
    else:
      assert 0

    x = self.linear(x)

    x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
    # out = self.sin(gain * x + bias)
    out = self.lrelu((gain + 1.) * x + bias)

    return out

  def __repr__(self):
    s = f'{self.__class__.__name__}(' \
        f'in_dim={self.in_dim}, ' \
        f'out_dim={self.out_dim}, ' \
        f'style_dim={self.style_dim}, ' \
        f'use_style_fc={self.use_style_fc}, ' \
        f')'
    return s


# @MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetwork(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               hidden_layers=2,
               style_dim=512,
               rgb_dim=3,
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

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'in_dim': in_dim,
      'hidden_dim': hidden_dim,
      'hidden_layers': hidden_layers,
      'style_dim': style_dim,
      'rgb_dim': rgb_dim,
    })

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

    self.module_name_list = []

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
    self.module_name_list.append('network')
    _out_dim = in_dim
    for idx in range(hidden_layers):
      _in_dim = _out_dim
      _out_dim = hidden_dim

      _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)

      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))
    self.module_name_list.append('final_layer')

    _in_dim= hidden_dim
    _out_dim = hidden_dim // 2
    self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    # self.color_layer_sine = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
    self.module_name_list.append('color_layer_sine')

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.LeakyReLU(0.2, inplace=True),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)
    self.module_name_list.append('color_layer_linear')

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = nerf_network.UniformBoxWarp(0.24)

    logger = logging.getLogger('tl')
    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['nerf'] = self
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)
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
class NeRFNetwork_sigma(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               hidden_layers=2,
               style_dim=512,
               rgb_dim=3,
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

    self.repr = f"in_dim={in_dim}, " \
                f"hidden_dim={hidden_dim}, " \
                f"hidden_layers={hidden_layers}, " \
                f"style_dim={style_dim}, " \
                f"rgb_dim={rgb_dim}, " \

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
      if True:
        _layer = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
      else:
        _layer = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= 3
    _out_dim = hidden_dim // 2
    # self.color_layer_sine = film_layer.FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    # self.color_layer_sine = FiLMLayer(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, use_style_fc=True)
    # self.style_dim_dict[f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
    self.color_layer_sine = LinearSinAct(in_features=_in_dim, out_features=_out_dim)

    self.color_layer_linear = nn.Sequential(
      nn.Linear(_out_dim, rgb_dim),
      # nn.Sigmoid()
    )
    # self.color_layer_linear.apply(frequency_init(25))
    self.color_layer_linear.apply(init_func.kaiming_leaky_init)

    self.dim_styles = sum(self.style_dim_dict.values())

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = nerf_network.UniformBoxWarp(0.24)

    torch_utils.print_number_params({
      'network': self.network,
      'final_layer': self.final_layer,
      'color_layer_sine': self.color_layer_sine,
      'color_layer_linear': self.color_layer_linear,
      'nerf_net': self})
    logging.getLogger('tl').info(self)
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
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(input, ),
                                   name_prefix=f"color_layer_sine.")
    x = self.color_layer_sine(input, )

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


class INRNetwork_Skip(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

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

    self.repr = f"input_dim={input_dim}, " \
                f"style_dim={style_dim}, " \
                f"hidden_layers={hidden_layers}, " \
                f"dim_scale={dim_scale}, "

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

      _layer = film_layer.FiLMLayer(in_dim=_in_dim,
                                    out_dim=_out_dim,
                                    style_dim=style_dim)
      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{i}_0'] = _layer.style_dim

      _layer = film_layer.FiLMLayer(in_dim=_out_dim,
                                    out_dim=_out_dim,
                                    style_dim=style_dim)
      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{i}_1'] = _layer.style_dim

      to_rgb = inr_network.ToRGB(in_dim=_out_dim, dim_rgb=3)
      self.to_rbgs.append(to_rgb)

    self.tanh = nn.Sequential(
      # nn.Linear(hidden_dim, rgb_dim),
      nn.Tanh()
    )
    # self.to_rbg.apply(frequency_init(25))

    torch_utils.print_number_params(
      {
        'network': self.network,
        'to_rbgs': self.to_rbgs,
        'inr_net': self
      })
    logging.getLogger('tl').info(self)
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



class ModSinLayer(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_dim,
               use_style_fc=False,
               style_dim=None,
               which_linear=nn.Linear,
               spectral_norm=False,
               eps=1e-5,
               freq=1,
               phase=0,
               **kwargs):
    super(ModSinLayer, self).__init__()

    self.repr = f"in_dim={in_dim}, use_style_fc={use_style_fc}, style_dim={style_dim}, " \
                f"freq={freq}, phase={phase}"

    self.in_dim = in_dim
    self.use_style_fc = use_style_fc
    self.style_dim = style_dim
    self.freq = freq
    self.phase = phase

    self.spectral_norm = spectral_norm
    # Prepare gain and bias layers

    if use_style_fc:
      self.gain_fc = which_linear(style_dim, in_dim)
      self.bias_fc = which_linear(style_dim, in_dim)
      if spectral_norm:
        self.gain_fc = nn.utils.spectral_norm(self.gain_fc)
        self.bias_fc = nn.utils.spectral_norm(self.bias_fc)
    else:
      self.style_dim = in_dim * 2

    self.eps = eps

    self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    # self.register_buffer('stored_mean', torch.zeros(output_size))
    # self.register_buffer('stored_var', torch.ones(output_size))
    pass

  def forward(self,
              x,
              style):
    """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
    assert style.shape[-1] == self.style_dim

    if self.use_style_fc:
      gain = self.gain_fc(style) + 1.
      bias = self.bias_fc(style)
    else:
      style = rearrange(style, "b (n c) -> b n c", n=2)
      gain, bias = style.unbind(dim=1)
      gain = gain + 1.

    if x.dim() == 3:
      gain = rearrange(gain, "b c -> b 1 c")
      bias = rearrange(bias, "b c -> b 1 c")
    elif x.dim() == 2:
      pass
    else:
      assert 0

    # x = torch.sin(self.freq * x + self.phase)
    # out = x * gain + bias

    x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
    x = x * gain + bias

    out = self.lrelu(x)
    return out


class ModSinLayer_NoBias(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_dim,
               use_style_fc=False,
               style_dim=None,
               which_linear=nn.Linear,
               spectral_norm=False,
               eps=1e-5,
               freq=1,
               phase=0,
               **kwargs):
    super(ModSinLayer_NoBias, self).__init__()

    self.repr = f"in_dim={in_dim}, use_style_fc={use_style_fc}, style_dim={style_dim}, " \
                f"freq={freq}, phase={phase}"

    self.in_dim = in_dim
    self.use_style_fc = use_style_fc
    self.style_dim = style_dim
    self.freq = freq
    self.phase = phase

    self.spectral_norm = spectral_norm
    # Prepare gain and bias layers

    if use_style_fc:
      self.gain_fc = which_linear(style_dim, in_dim)
      # self.bias_fc = which_linear(style_dim, in_dim)
      if spectral_norm:
        self.gain_fc = nn.utils.spectral_norm(self.gain_fc)
        # self.bias_fc = nn.utils.spectral_norm(self.bias_fc)
    else:
      self.style_dim = in_dim * 2

    self.eps = eps
    pass

  def forward(self,
              x,
              style):
    """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
    assert style.shape[-1] == self.style_dim

    if self.use_style_fc:
      gain = self.gain_fc(style) + 1.
    else:
      style = rearrange(style, "b (n c) -> b n c", n=2)
      gain, bias = style.unbind(dim=1)
      gain = gain + 1.

    if x.dim() == 3:
      gain = rearrange(gain, "b c -> b 1 c")
    elif x.dim() == 2:
      pass
    else:
      assert 0

    x = torch.sin(self.freq * x + self.phase)
    # out = x * gain + bias
    out = x * gain
    return out


class SinBlock(nn.Module):
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

    # self.mod1 = mod_conv_fc.Modulated_FC_Conv(in_channel=in_dim,
    #                                           out_channel=out_dim,
    #                                           style_dim=style_dim,
    #                                           use_style_fc=True,
    #                                           scale=1.,
    #                                           # scale=None,
    #                                           )
    self.mod1 = mod_conv_fc.SinStyleMod(in_channel=in_dim,
                                        out_channel=out_dim,
                                        style_dim=style_dim,
                                        use_style_fc=True,
                                        )
    self.style_dim_dict[f'{name_prefix}_0'] = self.mod1.style_dim
    self.act1 = nn.LeakyReLU(0.2, inplace=True)

    # self.mod2 = mod_conv_fc.Modulated_FC_Conv(in_channel=out_dim,
    #                                           out_channel=out_dim,
    #                                           style_dim=style_dim,
    #                                           use_style_fc=True,
    #                                           scale=1.,
    #                                           # scale=None,
    #                                           )
    self.mod2 = mod_conv_fc.SinStyleMod(in_channel=out_dim,
                                        out_channel=out_dim,
                                        style_dim=style_dim,
                                        use_style_fc=True,
                                        )
    self.style_dim_dict[f'{name_prefix}_1'] = self.mod2.style_dim
    self.act2 = nn.LeakyReLU(0.2, inplace=True)

    # self.linear1 = nn.Linear(in_dim, out_dim)
    # self.mod1 = ModSinLayer(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    # self.style_dim_dict[f'{name_prefix}_0'] = self.mod1.style_dim

    # self.linear2 = nn.Linear(out_dim, out_dim)
    # self.mod2 = ModSinLayer(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    # self.style_dim_dict[f'{name_prefix}_1'] = self.mod2.style_dim

    self.skip = SkipLayer()
    pass

  def forward(self,
              x,
              style_dict,
              skip=False):
    x_orig = x

    style = style_dict[f'{self.name_prefix}_0']
    x = self.mod1(x, style)
    x = self.act1(x)

    style = style_dict[f'{self.name_prefix}_1']
    x = self.mod2(x, style)
    out = self.act2(x)

    # x = self.linear1(x)
    # style = style_dict[f'{self.name_prefix}_0']
    # x = self.mod1(x, style)

    # x = self.linear2(x)
    # style = style_dict[f'{self.name_prefix}_1']
    # out = self.mod2(x, style)

    if skip and out.shape[-1] == x_orig.shape[-1]:
      # out = (out + x_orig) / 1.41421
      out = self.skip(out, x_orig)
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim}, " \
           f"style_dim={self.style_dim})"
    return repr


class ToRGB(nn.Module):
  def __init__(self,
               in_dim,
               dim_rgb=3,
               use_equal_fc=False):
    super().__init__()
    self.in_dim = in_dim
    self.dim_rgb = dim_rgb

    if use_equal_fc:
      self.linear = mod_conv_fc.EqualLinear(in_dim, dim_rgb, scale=1.)
    else:
      self.linear = nn.Linear(in_dim, dim_rgb)
    pass

  def forward(self,
              input,
              skip=None):

    out = self.linear(input)

    if skip is not None:
      out = out + skip
    return out


class CIPSNet(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               input_dim,
               style_dim,
               hidden_dim=256,
               pre_rgb_dim=32,
               device=None,
               name_prefix='inr',
               **kwargs):
    """

    :param input_dim:
    :param style_dim:
    :param hidden_dim:
    :param pre_rgb_dim:
    :param device:
    :param name_prefix:
    :param kwargs:
    """
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'input_dim': input_dim,
      'style_dim': style_dim,
      'hidden_dim': hidden_dim,
      'pre_rgb_dim': pre_rgb_dim,
    })

    self.device = device
    self.pre_rgb_dim = pre_rgb_dim
    self.name_prefix = name_prefix

    self.channels = {
      "4": hidden_dim,
      "8": hidden_dim,
      "16": hidden_dim,
      "32": hidden_dim,
      "64": hidden_dim,
      "128": hidden_dim,
      "256": hidden_dim,
      "512": hidden_dim,
      "1024": hidden_dim,
    }

    self.module_name_list = []

    self.style_dim_dict = {}

    _out_dim = input_dim

    network = OrderedDict()
    to_rbgs = OrderedDict()
    for i, (name, channel) in enumerate(self.channels.items()):
      _in_dim = _out_dim
      _out_dim = channel

      if name.startswith(('none', )):
        _linear_block = inr_network.LinearBlock(
          in_dim=_in_dim, out_dim=_out_dim, name_prefix=f'{name_prefix}_{name}')
        network[name] = _linear_block
      else:
        _film_block = SinBlock(in_dim=_in_dim,
                               out_dim=_out_dim,
                               style_dim=style_dim,
                               name_prefix=f'{name_prefix}_w{name}')
        self.style_dim_dict.update(_film_block.style_dim_dict)
        network[name] = _film_block

      _to_rgb = ToRGB(in_dim=_out_dim, dim_rgb=pre_rgb_dim, use_equal_fc=False)
      to_rbgs[name] = _to_rgb

    self.network = nn.ModuleDict(network)
    self.to_rgbs = nn.ModuleDict(to_rbgs)
    self.to_rgbs.apply(inr_network.frequency_init(100))
    self.module_name_list.append('network')
    self.module_name_list.append('to_rgbs')

    out_layers = []
    if pre_rgb_dim > 3:
      out_layers.append(nn.Linear(pre_rgb_dim, 3))
    out_layers.append(nn.Tanh())
    self.tanh = nn.Sequential(*out_layers)
    # self.tanh.apply(init_func.kaiming_leaky_init)
    self.tanh.apply(inr_network.frequency_init(100))
    self.module_name_list.append('tanh')

    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['cips'] = self
    logger = logging.getLogger('tl')
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)
    pass

  def forward(self,
              input,
              style_dict,
              img_size=1024,
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
    for idx, (name, block) in enumerate(self.network.items()):
      # skip = int(name) >= 32
      if idx >= 4:
        skip = True
      else:
        skip = False
      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(block,
                                     inputs_args=(x, style_dict, skip),
                                     submodels=['mod1', 'mod2'],
                                     name_prefix=f'block.{name}.')
      x = block(x, style_dict, skip=skip)

      if idx >= 3:
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



@MODEL_REGISTRY.register(name_prefix=__name__)
class GeneratorNerfINR(GeneratorNerfINR_base):

  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               z_dim,
               nerf_cfg,
               mapping_nerf_cfg,
               inr_cfg,
               mapping_inr_cfg,
               device='cuda',
               **kwargs):
    super(GeneratorNerfINR_base, self).__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'z_dim': z_dim,
      'nerf_cfg': nerf_cfg,
      'mapping_nerf_cfg': mapping_nerf_cfg,
      'inr_cfg': inr_cfg,
      'mapping_inr_cfg': mapping_inr_cfg,
    })

    self.epoch = 0
    self.step = 0
    self.z_dim = z_dim
    self.device = device

    self.module_name_list = []
    # nerf_net
    self.siren = NeRFNetwork(**nerf_cfg)
    self.module_name_list.append('siren')

    shape_style_dict = self.siren.style_dim_dict
    color_style_dict = {
      'nerf_rgb': shape_style_dict.pop('nerf_rgb')
    }
    self.mapping_network_nerf = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_nerf_cfg, 'head_dim_dict': shape_style_dict})
    self.module_name_list.append('mapping_network_nerf')

    # inr_net
    self.inr_net = CIPSNet(**{**inr_cfg, "input_dim": self.siren.rgb_dim})
    self.module_name_list.append('inr_net')

    color_style_dict.update(self.inr_net.style_dim_dict)
    self.mapping_network_inr = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_inr_cfg, 'head_dim_dict': color_style_dict})
    self.module_name_list.append('mapping_network_inr')

    self.nerf_rgb_mapping = nn.Linear(in_features=mapping_inr_cfg['hidden_dim'],
                                      out_features=color_style_dict['nerf_rgb'])
    self.module_name_list.append('nerf_rgb_mapping')

    self.aux_to_rbg = nn.Sequential(
      nn.Linear(self.siren.rgb_dim, 3),
      nn.Tanh()
    )
    self.aux_to_rbg.apply(nerf_network.frequency_init(25))
    self.module_name_list.append('aux_to_rbg')

    self.filters = nn.Identity()

    logger = logging.getLogger('tl')
    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['G'] = self
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)

    pass

  # def get_optimizer(self,
  #                   lr,
  #                   equal_lr,
  #                   betas=(0, 0.999),
  #                   weight_decay=0.):
  #   optimizer = torch.optim.Adam(
  #     [
  #       {
  #         'params': chain(
  #           self.siren.parameters(),
  #           self.mapping_network_nerf.parameters(),
  #           self.mapping_network_inr.parameters(),
  #           self.aux_to_rbg.parameters(),
  #           self.filters.parameters(),
  #         ),
  #         'initial_lr': lr,
  #         'lr': lr},
  #       {
  #         'params': chain(
  #           self.inr_net.parameters(),
  #         ),
  #         'initial_lr': equal_lr,
  #         'lr': equal_lr}
  #     ],
  #     betas=betas,
  #     weight_decay=weight_decay)
  #
  #   num_params = 0
  #   for group in optimizer.param_groups:
  #     num_params += len(group['params'])
  #   assert num_params == len(list(self.parameters()))
  #   return optimizer

  def forward(self,
              zs,
              img_size,
              fov,
              ray_start,
              ray_end,
              num_steps,
              h_stddev,
              v_stddev,
              hierarchical_sample,
              h_mean=math.pi*0.5,
              v_mean=math.pi*0.5,
              psi=1,
              sample_dist=None,
              lock_view_dependence=False,
              clamp_mode='relu',
              nerf_noise=0.,
              white_back=False,
              last_back=False,
              return_aux_img=False,
              grad_points=None,
              forward_points=None,
              **kwargs):
    """
    Generates images from a noise vector, rendering parameters, and camera distribution.
    Uses the hierarchical sampling scheme described in NeRF.

    :param z: (b, z_dim)
    :param img_size:
    :param fov: face: 12
    :param ray_start: face: 0.88
    :param ray_end: face: 1.12
    :param num_steps: face: 12
    :param h_stddev: face: 0.3
    :param v_stddev: face: 0.155
    :param h_mean: face: pi/2
    :param v_mean: face: pi/2
    :param hierarchical_sample: face: true
    :param psi: [0, 1]
    :param sample_dist: mode for sample_camera_positions, face: 'gaussian'
    :param lock_view_dependence: face: false
    :param clamp_mode: face: 'relu'
    :param nerf_noise:
    :param last_back: face: false
    :param white_back: face: false
    :param kwargs:
    :return:
    - pixels: (b, 3, h, w)
    - pitch_yaw: (b, 2)
    """

    # mapping network
    style_dict = self.mapping_network(**zs)

    if psi < 1:
      avg_styles = self.generate_avg_frequencies(device=self.device)
      style_dict = self.get_truncated_freq_phase(
        raw_style_dict=style_dict, avg_style_dict=avg_styles, raw_lambda=psi)

    if grad_points is not None and grad_points < img_size ** 2:
      imgs, pitch_yaw = self.part_grad_forward(
        style_dict=style_dict,
        img_size=img_size,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end,
        num_steps=num_steps,
        h_stddev=h_stddev,
        v_stddev=v_stddev,
        h_mean=h_mean,
        v_mean=v_mean,
        hierarchical_sample=hierarchical_sample,
        sample_dist=sample_dist,
        lock_view_dependence=lock_view_dependence,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
        grad_points=grad_points,
      )
      return imgs, pitch_yaw
    else:
      imgs, pitch_yaw = self.whole_grad_forward(
        style_dict=style_dict,
        img_size=img_size,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end,
        num_steps=num_steps,
        h_stddev=h_stddev,
        v_stddev=v_stddev,
        h_mean=h_mean,
        v_mean=v_mean,
        hierarchical_sample=hierarchical_sample,
        sample_dist=sample_dist,
        lock_view_dependence=lock_view_dependence,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
        forward_points=forward_points,
      )
      return imgs, pitch_yaw

  def get_batch_style_dict(self, b, style_dict):
    ret_style_dict = {}
    for name, style in style_dict.items():
      ret_style_dict[name] = style[[b]]
    return ret_style_dict

  def whole_grad_forward(self,
                         style_dict,
                         img_size,
                         fov,
                         ray_start,
                         ray_end,
                         num_steps,
                         h_stddev,
                         v_stddev,
                         h_mean,
                         v_mean,
                         hierarchical_sample,
                         sample_dist=None,
                         lock_view_dependence=False,
                         clamp_mode='relu',
                         nerf_noise=0.,
                         white_back=False,
                         last_back=False,
                         return_aux_img=True,
                         forward_points=None,
                         camera_pos=None,
                         camera_lookup=None,
                         ):
    device = self.device
    # batch_size = z.shape[0]
    batch_size = list(style_dict.values())[0].shape[0]


    if forward_points is not None:
      # stage forward
      with torch.no_grad():
        num_points = img_size ** 2
        inr_img_output = torch.zeros((batch_size, num_points, 3), device=device)
        if return_aux_img:
          aux_img_output = torch.zeros((batch_size, num_points, 3), device=device)
        pitch_list = []
        yaw_list = []
        for b in range(batch_size):
          transformed_points, \
          transformed_ray_directions_expanded, \
          transformed_ray_origins, \
          transformed_ray_directions, \
          z_vals, \
          pitch, \
          yaw = comm_utils.get_world_points_and_direction(
            batch_size=1,
            num_steps=num_steps,
            img_size=img_size,
            fov=fov,
            ray_start=ray_start,
            ray_end=ray_end,
            h_stddev=h_stddev,
            v_stddev=v_stddev,
            h_mean=h_mean,
            v_mean=v_mean,
            sample_dist=sample_dist,
            lock_view_dependence=lock_view_dependence,
            device=device,
            camera_pos=camera_pos,
            camera_lookup=camera_lookup,
          )
          pitch_list.append(pitch)
          yaw_list.append(yaw)

          transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
          transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
                                                          "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
          head = 0
          while head < num_points:
            tail = head + forward_points
            cur_style_dict = self.get_batch_style_dict(b=b, style_dict=style_dict)
            cur_inr_img, cur_aux_img = self.points_forward(
              style_dict=cur_style_dict,
              transformed_points=transformed_points[:, head:tail],
              transformed_ray_directions_expanded=transformed_ray_directions_expanded[:, head:tail],
              num_steps=num_steps,
              hierarchical_sample=hierarchical_sample,
              z_vals=z_vals[:, head:tail],
              clamp_mode=clamp_mode,
              nerf_noise=nerf_noise,
              transformed_ray_origins=transformed_ray_origins[:, head:tail],
              transformed_ray_directions=transformed_ray_directions[:, head:tail],
              white_back=white_back,
              last_back=last_back,
              return_aux_img=return_aux_img,
            )
            inr_img_output[b:b + 1, head:tail] = cur_inr_img
            if return_aux_img:
              aux_img_output[b:b + 1, head:tail] = cur_aux_img
            head += forward_points
        inr_img = inr_img_output
        if return_aux_img:
          aux_img = aux_img_output
        pitch = torch.cat(pitch_list, dim=0)
        yaw = torch.cat(yaw_list, dim=0)
    else:
      transformed_points, \
      transformed_ray_directions_expanded, \
      transformed_ray_origins, \
      transformed_ray_directions, \
      z_vals, \
      pitch, \
      yaw = comm_utils.get_world_points_and_direction(
        batch_size=batch_size,
        num_steps=num_steps,
        img_size=img_size,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end,
        h_stddev=h_stddev,
        v_stddev=v_stddev,
        h_mean=h_mean,
        v_mean=v_mean,
        sample_dist=sample_dist,
        lock_view_dependence=lock_view_dependence,
        device=device,
        camera_pos=camera_pos,
        camera_lookup=camera_lookup,
      )

      transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
      transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
                                                      "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
      inr_img, aux_img = self.points_forward(
        style_dict=style_dict,
        transformed_points=transformed_points,
        transformed_ray_directions_expanded=transformed_ray_directions_expanded,
        num_steps=num_steps,
        hierarchical_sample=hierarchical_sample,
        z_vals=z_vals,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
      )

    inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.filters,
                                   inputs_args=(inr_img, ),
                                   name_prefix="filters.")
    inr_img = self.filters(inr_img)
    pitch_yaw = torch.cat([pitch, yaw], -1)

    if return_aux_img:
      aux_img = rearrange(aux_img, "b (h w) c -> b c h w", h=img_size)

      imgs = torch.cat([inr_img, aux_img])
      pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
    else:
      imgs = inr_img

    return imgs, pitch_yaw

  def part_grad_forward(self,
                        style_dict,
                        img_size,
                        fov,
                        ray_start,
                        ray_end,
                        num_steps,
                        h_stddev,
                        v_stddev,
                        h_mean,
                        v_mean,
                        hierarchical_sample,
                        sample_dist=None,
                        lock_view_dependence=False,
                        clamp_mode='relu',
                        nerf_noise=0.,
                        white_back=False,
                        last_back=False,
                        return_aux_img=True,
                        grad_points=None,
                        camera_pos=None,
                        camera_lookup=None,
                        ):
    device = self.device
    batch_size = list(style_dict.values())[0].shape[0]
    transformed_points, \
    transformed_ray_directions_expanded, \
    transformed_ray_origins, \
    transformed_ray_directions, \
    z_vals, \
    pitch, \
    yaw = comm_utils.get_world_points_and_direction(
      batch_size=batch_size,
      num_steps=num_steps,
      img_size=img_size,
      fov=fov,
      ray_start=ray_start,
      ray_end=ray_end,
      h_stddev=h_stddev,
      v_stddev=v_stddev,
      h_mean=h_mean,
      v_mean=v_mean,
      sample_dist=sample_dist,
      lock_view_dependence=lock_view_dependence,
      device=device,
      camera_pos=camera_pos,
      camera_lookup=camera_lookup,
    )

    transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
    transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
                                                    "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)

    num_points = transformed_points.shape[1]
    assert num_points > grad_points
    rand_idx = torch.randperm(num_points, device=device)
    idx_grad = rand_idx[:grad_points]
    idx_no_grad = rand_idx[grad_points:]

    inr_img_grad, aux_img_grad = self.points_forward(
      style_dict=style_dict,
      transformed_points=transformed_points,
      transformed_ray_directions_expanded=transformed_ray_directions_expanded,
      num_steps=num_steps,
      hierarchical_sample=hierarchical_sample,
      z_vals=z_vals,
      clamp_mode=clamp_mode,
      nerf_noise=nerf_noise,
      transformed_ray_origins=transformed_ray_origins,
      transformed_ray_directions=transformed_ray_directions,
      white_back=white_back,
      last_back=last_back,
      return_aux_img=return_aux_img,
      idx_grad=idx_grad,
    )

    with torch.no_grad():
      inr_img_no_grad, aux_img_no_grad = self.points_forward(
        style_dict=style_dict,
        transformed_points=transformed_points,
        transformed_ray_directions_expanded=transformed_ray_directions_expanded,
        num_steps=num_steps,
        hierarchical_sample=hierarchical_sample,
        z_vals=z_vals,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
        idx_grad=idx_no_grad,
      )

    inr_img = comm_utils.scatter_points(idx_grad=idx_grad,
                                        points_grad=inr_img_grad,
                                        idx_no_grad=idx_no_grad,
                                        points_no_grad=inr_img_no_grad,
                                        num_points=num_points)

    inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.filters,
                                   inputs_args=(inr_img,),
                                   name_prefix="filters.")
    inr_img = self.filters(inr_img)
    pitch_yaw = torch.cat([pitch, yaw], -1)

    if return_aux_img:
      aux_img = comm_utils.scatter_points(idx_grad=idx_grad,
                                          points_grad=aux_img_grad,
                                          idx_no_grad=idx_no_grad,
                                          points_no_grad=aux_img_no_grad,
                                          num_points=num_points)
      aux_img = rearrange(aux_img, "b (h w) c -> b c h w", h=img_size)

      imgs = torch.cat([inr_img, aux_img])
      pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
    else:
      imgs = inr_img

    return imgs, pitch_yaw

  def points_forward(self,
                     style_dict,
                     transformed_points,
                     transformed_ray_directions_expanded,
                     num_steps,
                     hierarchical_sample,
                     z_vals,
                     clamp_mode,
                     nerf_noise,
                     transformed_ray_origins,
                     transformed_ray_directions,
                     white_back,
                     last_back,
                     return_aux_img,
                     idx_grad=None,
                     ):
    """

    :param style_dict:
    :param transformed_points: (b, n, s, 3)
    :param transformed_ray_directions_expanded: (b, n, s, 3)
    :param num_steps: sampled points along a ray
    :param hierarchical_sample:
    :param z_vals: (b, n, s, 1)
    :param clamp_mode: 'relu'
    :param nerf_noise:
    :param transformed_ray_origins: (b, n, 3)
    :param transformed_ray_directions: (b, n, 3)
    :param white_back:
    :param last_back:
    :return:
    """
    device = transformed_points.device
    if idx_grad is not None:
      transformed_points = comm_utils.gather_points(points=transformed_points, idx_grad=idx_grad)
      transformed_ray_directions_expanded = comm_utils.gather_points(
        points=transformed_ray_directions_expanded, idx_grad=idx_grad)
      z_vals = comm_utils.gather_points(points=z_vals, idx_grad=idx_grad)
      transformed_ray_origins = comm_utils.gather_points(points=transformed_ray_origins, idx_grad=idx_grad)
      transformed_ray_directions = comm_utils.gather_points(points=transformed_ray_directions, idx_grad=idx_grad)

    transformed_points = rearrange(transformed_points, "b n s c -> b (n s) c")
    transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded, "b n s c -> b (n s) c")

    # Model prediction on course points
    coarse_output = self.siren(
      input=transformed_points,  # (b, n x s, 3)
      style_dict=style_dict,
      ray_directions=transformed_ray_directions_expanded,
    )
    coarse_output = rearrange(coarse_output, "b (n s) rgb_sigma -> b n s rgb_sigma", s=num_steps)

    # Re-sample fine points alont camera rays, as described in NeRF
    if hierarchical_sample:
      fine_points, fine_z_vals = self.get_fine_points_and_direction(
        coarse_output=coarse_output,
        z_vals=z_vals,
        dim_rgb=self.siren.rgb_dim,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        num_steps=num_steps,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions
      )

      # Model prediction on re-sampled find points
      fine_output = self.siren(
        input=fine_points,  # (b, n x s, 3)
        style_dict=style_dict,
        ray_directions=transformed_ray_directions_expanded,  # (b, n x s, 3)
      )
      fine_output = rearrange(fine_output, "b (n s) rgb_sigma -> b n s rgb_sigma", s=num_steps)

      # Combine course and fine points
      all_outputs = torch.cat([fine_output, coarse_output], dim=-2)  # (b, n, s, dim_rgb_sigma)
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)  # (b, n, s, 1)
      _, indices = torch.sort(all_z_vals, dim=-2)  # (b, n, s, 1)
      all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, n, s, 1)
      # (b, n, s, dim_rgb_sigma)
      all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    # Create images with NeRF
    pixels_fea, depth, weights = pigan_utils.fancy_integration(
      rgb_sigma=all_outputs,
      z_vals=all_z_vals,
      device=device,
      dim_rgb=self.siren.rgb_dim,
      white_back=white_back,
      last_back=last_back,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    inr_img = self.inr_net(pixels_fea, style_dict)

    if return_aux_img:
      # aux rgb_branch
      aux_img = self.aux_to_rbg(pixels_fea)
    else:
      aux_img = None

    return inr_img, aux_img

  def z_sampler(self,
                shape,
                device,
                dist='gaussian'):
    if dist == 'gaussian':
      z = torch.randn(shape, device=device)
    elif dist == 'uniform':
      z = torch.rand(shape, device=device) * 2 - 1
    return z

  def get_zs(self, b, batch_split=1):
    z_nerf = self.z_sampler(shape=(b, self.mapping_network_nerf.z_dim), device=self.device)
    z_inr = self.z_sampler(shape=(b, self.mapping_network_inr.z_dim), device=self.device)

    if batch_split > 1:
      zs_list = []
      z_nerf_list = z_nerf.split(b // batch_split)
      z_inr_list = z_inr.split(b // batch_split)
      for z_nerf_, z_inr_ in zip(z_nerf_list, z_inr_list):
        zs_ = {
          'z_nerf': z_nerf_,
          'z_inr': z_inr_,
        }
        zs_list.append(zs_)
      return zs_list
    else:
      zs = {
        'z_nerf': z_nerf,
        'z_inr': z_inr,
      }
      return zs

  def mapping_network(self,
                      z_nerf,
                      z_inr):
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_network_nerf,
                                   inputs_args=(z_nerf,),
                                   submodels=['base_net'],
                                   name_prefix='mapping_nerf.')
      VerboseModel.forward_verbose(self.mapping_network_inr,
                                   inputs_args=(z_inr,),
                                   submodels=['base_net', ],
                                   input_padding=50,
                                   name_prefix='mapping_inr.')

    style_dict = {}
    style_dict.update(self.mapping_network_nerf(z_nerf))
    style_dict.update(self.mapping_network_inr(z_inr))

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.nerf_rgb_mapping,
                                   inputs_args=(style_dict['nerf_rgb'],),
                                   input_padding=50,
                                   name_prefix='nerf_rgb_mapping.')
    style_dict['nerf_rgb'] = self.nerf_rgb_mapping(style_dict['nerf_rgb'])

    return style_dict

  def generate_avg_frequencies(self, num_samples=10000, device='cuda'):
    """Calculates average frequencies and phase shifts"""

    # z = torch.randn((num_samples, self.z_dim), device=device)
    zs = self.get_zs(num_samples)
    with torch.no_grad():
      style_dict = self.mapping_network(**zs)

    avg_styles = {}
    for name, style in style_dict.items():
      avg_styles[name] = style.mean(0, keepdim=True)

    self.avg_styles = avg_styles
    return avg_styles

  def staged_forward(self, *args, **kwargs):
    raise NotImplementedError

  def set_device(self, device):
    # self.device = device
    # self.siren.device = device
    # self.generate_avg_frequencies()
    pass

  def forward_camera_pos_and_lookup(self,
              zs,
              img_size,
              fov,
              ray_start,
              ray_end,
              num_steps,
              h_stddev,
              v_stddev,
              h_mean,
              v_mean,
              hierarchical_sample,
              camera_pos,
              camera_lookup,
              psi=1,
              sample_dist=None,
              lock_view_dependence=False,
              clamp_mode='relu',
              nerf_noise=0.,
              white_back=False,
              last_back=False,
              return_aux_img=False,
              grad_points=None,
              forward_points=None,
              **kwargs):
    """
    Generates images from a noise vector, rendering parameters, and camera distribution.
    Uses the hierarchical sampling scheme described in NeRF.

    :param z: (b, z_dim)
    :param img_size:
    :param fov: face: 12
    :param ray_start: face: 0.88
    :param ray_end: face: 1.12
    :param num_steps: face: 12
    :param h_stddev: face: 0.3
    :param v_stddev: face: 0.155
    :param h_mean: face: pi/2
    :param v_mean: face: pi/2
    :param hierarchical_sample: face: true
    :param camera_pos: (b, 3)
    :param camera_lookup: (b, 3)
    :param psi: [0, 1]
    :param sample_dist: mode for sample_camera_positions, face: 'gaussian'
    :param lock_view_dependence: face: false
    :param clamp_mode: face: 'relu'
    :param nerf_noise:
    :param last_back: face: false
    :param white_back: face: false
    :param kwargs:
    :return:
    - pixels: (b, 3, h, w)
    - pitch_yaw: (b, 2)
    """

    # mapping network
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_network_nerf,
                                   inputs_args=(zs['z_nerf'],),
                                   submodels=['base_net'],
                                   name_prefix='mapping_nerf.')
      VerboseModel.forward_verbose(self.mapping_network_inr,
                                   inputs_args=(zs['z_inr'],),
                                   submodels=['base_net', ],
                                   input_padding=50,
                                   name_prefix='mapping_inr.')
    style_dict = self.mapping_network(**zs)

    if psi < 1:
      avg_styles = self.generate_avg_frequencies(device=self.device)
      style_dict = self.get_truncated_freq_phase(
        raw_style_dict=style_dict, avg_style_dict=avg_styles, raw_lambda=psi)

    if grad_points is not None and grad_points < img_size ** 2:
      imgs, pitch_yaw = self.part_grad_forward(
        style_dict=style_dict,
        img_size=img_size,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end,
        num_steps=num_steps,
        h_stddev=h_stddev,
        v_stddev=v_stddev,
        h_mean=h_mean,
        v_mean=v_mean,
        hierarchical_sample=hierarchical_sample,
        sample_dist=sample_dist,
        lock_view_dependence=lock_view_dependence,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
        grad_points=grad_points,
        camera_pos=camera_pos,
        camera_lookup=camera_lookup,
      )
      return imgs, pitch_yaw
    else:
      imgs, pitch_yaw = self.whole_grad_forward(
        style_dict=style_dict,
        img_size=img_size,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end,
        num_steps=num_steps,
        h_stddev=h_stddev,
        v_stddev=v_stddev,
        h_mean=h_mean,
        v_mean=v_mean,
        hierarchical_sample=hierarchical_sample,
        sample_dist=sample_dist,
        lock_view_dependence=lock_view_dependence,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        white_back=white_back,
        last_back=last_back,
        return_aux_img=return_aux_img,
        forward_points=forward_points,
        camera_pos=camera_pos,
        camera_lookup=camera_lookup,
      )
      return imgs, pitch_yaw


@MODEL_REGISTRY.register(name_prefix=__name__)
class GeneratorNerfINR_freeze_NeRF(GeneratorNerfINR):

  def load_nerf_ema(self, G_ema):
    ret = self.siren.load_state_dict(G_ema.siren.state_dict())
    ret = self.mapping_network_nerf.load_state_dict(G_ema.mapping_network_nerf.state_dict())
    ret = self.aux_to_rbg.load_state_dict(G_ema.aux_to_rbg.state_dict())

    ret = self.mapping_network_inr.load_state_dict(G_ema.mapping_network_inr.state_dict())
    ret = self.nerf_rgb_mapping.load_state_dict(G_ema.nerf_rgb_mapping.state_dict())
    pass

  def mapping_network(self,
                      z_nerf,
                      z_inr):
    style_dict = {}
    with torch.no_grad():
      style_dict.update(self.mapping_network_nerf(z_nerf))
      style_dict.update(self.mapping_network_inr(z_inr))
      style_dict['nerf_rgb'] = self.nerf_rgb_mapping(style_dict['nerf_rgb'])

    return style_dict

  def points_forward(self,
                     style_dict,
                     transformed_points,
                     transformed_ray_directions_expanded,
                     num_steps,
                     hierarchical_sample,
                     z_vals,
                     clamp_mode,
                     nerf_noise,
                     transformed_ray_origins,
                     transformed_ray_directions,
                     white_back,
                     last_back,
                     return_aux_img,
                     idx_grad=None,
                     ):
    """

    :param style_dict:
    :param transformed_points: (b, n, s, 3)
    :param transformed_ray_directions_expanded: (b, n, s, 3)
    :param num_steps: sampled points along a ray
    :param hierarchical_sample:
    :param z_vals: (b, n, s, 1)
    :param clamp_mode: 'relu'
    :param nerf_noise:
    :param transformed_ray_origins: (b, n, 3)
    :param transformed_ray_directions: (b, n, 3)
    :param white_back:
    :param last_back:
    :return:
    """
    device = transformed_points.device
    if idx_grad is not None:
      transformed_points = comm_utils.gather_points(points=transformed_points, idx_grad=idx_grad)
      transformed_ray_directions_expanded = comm_utils.gather_points(
        points=transformed_ray_directions_expanded, idx_grad=idx_grad)
      z_vals = comm_utils.gather_points(points=z_vals, idx_grad=idx_grad)
      transformed_ray_origins = comm_utils.gather_points(points=transformed_ray_origins, idx_grad=idx_grad)
      transformed_ray_directions = comm_utils.gather_points(points=transformed_ray_directions, idx_grad=idx_grad)

    transformed_points = rearrange(transformed_points, "b n s c -> b (n s) c")
    transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded, "b n s c -> b (n s) c")

    # Model prediction on course points
    with torch.no_grad():
      coarse_output = self.siren(
        input=transformed_points,  # (b, n x s, 3)
        style_dict=style_dict,
        ray_directions=transformed_ray_directions_expanded,
      )
    coarse_output = rearrange(coarse_output, "b (n s) rgb_sigma -> b n s rgb_sigma", s=num_steps)

    # Re-sample fine points alont camera rays, as described in NeRF
    if hierarchical_sample:
      fine_points, fine_z_vals = self.get_fine_points_and_direction(
        coarse_output=coarse_output,
        z_vals=z_vals,
        dim_rgb=self.siren.rgb_dim,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        num_steps=num_steps,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions
      )

      # Model prediction on re-sampled find points
      with torch.no_grad():
        fine_output = self.siren(
          input=fine_points,  # (b, n x s, 3)
          style_dict=style_dict,
          ray_directions=transformed_ray_directions_expanded,  # (b, n x s, 3)
        )
      fine_output = rearrange(fine_output, "b (n s) rgb_sigma -> b n s rgb_sigma", s=num_steps)

      # Combine course and fine points
      all_outputs = torch.cat([fine_output, coarse_output], dim=-2)  # (b, n, s, dim_rgb_sigma)
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)  # (b, n, s, 1)
      _, indices = torch.sort(all_z_vals, dim=-2)  # (b, n, s, 1)
      all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, n, s, 1)
      # (b, n, s, dim_rgb_sigma)
      all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    # Create images with NeRF
    pixels_fea, depth, weights = pigan_utils.fancy_integration(
      rgb_sigma=all_outputs,
      z_vals=all_z_vals,
      device=device,
      dim_rgb=self.siren.rgb_dim,
      white_back=white_back,
      last_back=last_back,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    inr_img = self.inr_net(pixels_fea, style_dict)

    if return_aux_img:
      # aux rgb_branch
      with torch.no_grad():
        aux_img = self.aux_to_rbg(pixels_fea)
    else:
      aux_img = None

    return inr_img, aux_img





