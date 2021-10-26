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
from tl2.proj.stylegan2_ada import persistence
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils

from exp.pigan import pigan_utils
from exp.pigan.pigan_utils import FiLMLayer
# from exp.pigan.models.volumetric_rendering import *
from exp.pigan.models.siren import \
  (CustomMappingNetwork, frequency_init, first_layer_film_sine_init, UniformBoxWarp)
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base


class NeRFNetwork(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               z_dim=100,
               hidden_dim=256,
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
    self.device = device
    self.z_dim = z_dim
    self.hidden_dim = hidden_dim
    self.rgb_dim = rgb_dim
    self.name_prefix = name_prefix

    # self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    # dim_xyz_emb = self.xyz_emb.get_out_dim()
    # self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    # dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    self.network = nn.ModuleList([
      FiLMLayer(3, hidden_dim),
      # FiLMLayer(dim_xyz_emb, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
      # FiLMLayer(hidden_dim, hidden_dim),
    ])
    self.network.apply(frequency_init(25))
    self.network[0].apply(first_layer_film_sine_init)
    self.style_dim_dict[f'{name_prefix}_network'] = len(self.network) * hidden_dim * 2

    self.final_layer = nn.Linear(hidden_dim, 1)
    self.final_layer.apply(frequency_init(25))

    self.color_layer_sine = FiLMLayer(hidden_dim, hidden_dim)
    self.color_layer_sine.apply(frequency_init(25))
    self.style_dim_dict[f'{name_prefix}_color_layer_sine'] = hidden_dim * 2

    self.color_layer_linear = nn.Sequential(
      nn.Linear(hidden_dim, rgb_dim),
      # nn.Sigmoid()
    )
    self.color_layer_linear.apply(frequency_init(25))

    self.dim_styles = sum(self.style_dim_dict.values())
    # self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

    self.print_number_params()
    pass

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

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            style_dict,
                                            ray_directions,
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

    frequencies, phase_shifts = self.get_freq_phase(style_dict=style_dict, name=f"{self.name_prefix}_network")
    for index, layer in enumerate(self.network):
      start = index * self.hidden_dim
      end = (index + 1) * self.hidden_dim

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,
                                                  frequencies[..., start:end],
                                                  phase_shifts[..., start:end]),
                                     name_prefix=f"network.{index}.")
      x = layer(x,
                frequencies[..., start:end],
                phase_shifts[..., start:end])

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix="final_layer")
    sigma = self.final_layer(x)

    # rgb branch
    frequencies, phase_shifts = self.get_freq_phase(
      style_dict=style_dict, name=f"{self.name_prefix}_color_layer_sine")
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(x,
                                                frequencies,
                                                phase_shifts),
                                   name_prefix=f"color_layer_sine.")
    rbg_sine = self.color_layer_sine(
      x,
      frequencies,
      phase_shifts)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_linear,
                                   inputs_args=(rbg_sine,),
                                   name_prefix='color_layer_linear.')
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    rbg = self.color_layer_linear(rbg_sine)

    out = torch.cat([rbg, sigma], dim=-1)
    return out

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


class INRNetwork(nn.Module):

  def __init__(self,
               input_dim,
               hidden_dim,
               hidden_layers,
               fc_layers,
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

    self.network = nn.ModuleList()
    for i in range(hidden_layers):
      if i == 0:
        in_dim = input_dim
      else:
        in_dim = hidden_dim
      out_dim = hidden_dim
      self.network.append(FiLMLayer(in_dim, out_dim))
    if len(self.network) > 0:
      self.network.apply(frequency_init(25))
      # self.network[0].apply(first_layer_film_sine_init)
      self.style_dim_dict[f'{name_prefix}_network'] = len(self.network) * hidden_dim * 2
    else:
      out_dim = input_dim

    self.fc_net = nn.ModuleList()
    for i in range(fc_layers):
      if i == 0:
        in_dim = out_dim
      else:
        in_dim = hidden_dim
      out_dim = hidden_dim
      self.fc_net.append(nn.Linear(in_dim, out_dim))
      self.fc_net.append(nn.LeakyReLU(0.2, inplace=True))

    if len(self.fc_net) > 0:
      self.fc_net.apply(frequency_init(25))

    self.to_rbg = nn.Sequential(
      nn.Linear(out_dim, rgb_dim),
      nn.Tanh()
    )
    self.to_rbg.apply(frequency_init(25))

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

    if len(self.network) > 0:
      frequencies, phase_shifts = self.get_freq_phase(
        style_dict=style_dict, name=f"{self.name_prefix}_network")
      for index, layer in enumerate(self.network):
        start = index * self.hidden_dim
        end = (index + 1) * self.hidden_dim

        if global_cfg.tl_debug:
          VerboseModel.forward_verbose(layer,
                                       inputs_args=(x,
                                                    frequencies[..., start:end],
                                                    phase_shifts[..., start:end]),
                                       name_prefix=f"{self.name_prefix}.network.{index}.")
        x = layer(x,
                  frequencies[..., start:end],
                  phase_shifts[..., start:end])

    if len(self.fc_net) > 0:
      for index, layer in enumerate(self.fc_net):
        if global_cfg.tl_debug:
          VerboseModel.forward_verbose(layer,
                                       inputs_args=(x, ),
                                       name_prefix=f"{self.name_prefix}.fc_net.{index}.")
        x = layer(x)

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
    torch_utils.print_number_params(
      {
        'network': self.network,
        'fc_net': self.fc_net,
        'to_rbg': self.to_rbg,
        'inr_net': self
      })
    pass


@MODEL_REGISTRY.register(name_prefix=__name__)
# @persistence.persistent_class
class GeneratorNerfINR(GeneratorNerfINR_base):
  def __init__(self,
               z_dim,
               nerf_cfg,
               inr_cfg,
               mapping_cfg,
               **kwargs):
    super(GeneratorNerfINR_base, self).__init__()

    self.z_dim = z_dim
    self.siren = NeRFNetwork(z_dim=self.z_dim,
                             device=None,
                             **nerf_cfg)
    # self.siren = build_model(cfg=siren_cfg, output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)

    self.inr_net = INRNetwork(**{**inr_cfg,
                                 "input_dim": self.siren.rgb_dim})

    self.style_dim_dict = {}
    self.style_dim_dict.update(self.siren.style_dim_dict)
    self.style_dim_dict.update(self.inr_net.style_dim_dict)

    self.mapping_network = pigan_utils.MultiHeadMappingNetwork(
      **{**mapping_cfg,
         'z_dim': z_dim,
         'head_dim_dict': self.style_dim_dict,
         })

    self.print_number_params()

    self.epoch = 0
    self.step = 0
    pass
