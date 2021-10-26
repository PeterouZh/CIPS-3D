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
# from exp.dev.nerf_inr.models.generator_nerf_inr import INRNetwork
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base
from exp.dev.nerf_inr.models.cond_layer_norm import CLN


class INRNetwork_CLN(nn.Module):

  def __init__(self,
               input_dim,
               hidden_dim,
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

    self.linear_layers = nn.ModuleList()
    self.cln_layers = nn.ModuleList()
    self.act_layers = nn.ModuleList()
    for i in range(hidden_layers):
      if i == 0:
        in_dim = input_dim
      else:
        in_dim = hidden_dim
      out_dim = hidden_dim

      linear_layer = nn.Linear(in_dim, out_dim)
      self.linear_layers.append(linear_layer)

      cln_layer = CLN(in_dim=out_dim)
      self.cln_layers.append(cln_layer)
      self.style_dim_dict[f'{name_prefix}_cln_{i}'] = cln_layer.style_dim

      act_layer = nn.LeakyReLU(0.2)
      self.act_layers.append(act_layer)

    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)


    self.to_rbg = nn.Sequential(
      nn.Linear(hidden_dim, rgb_dim),
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

    for index, (linear_layer, cln_layer, act_layer) in enumerate(zip(self.linear_layers,
                                                                     self.cln_layers,
                                                                     self.act_layers)):
      x = linear_layer(x)

      style = style_dict[f'{self.name_prefix}_cln_{index}']
      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(cln_layer,
                                     inputs_args=(x, style),
                                     name_prefix=f"{self.name_prefix}.cln_layers.{index}.")
      x = cln_layer(x, style)

      x = act_layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.to_rbg,
                                   inputs_args=(x, ),
                                   name_prefix='to_rgb.')
    out = self.to_rbg(x)
    return out

  def print_number_params(self):
    print()
    torch_utils.print_number_params({
      'linear_layers': self.linear_layers,
      'cln_layers': self.cln_layers,
      'to_rbg': self.to_rbg,
      'inr_net': self})
    pass




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

    self.inr_net = INRNetwork_CLN(**{**inr_cfg,
                                     "input_dim": self.siren.rgb_dim})

    self.style_dim_dict = {}
    self.style_dim_dict.update(self.siren.style_dim_dict)
    self.style_dim_dict.update(self.inr_net.style_dim_dict)

    self.mapping_network = pigan_utils.MultiHeadMappingNetwork(
      **{**mapping_cfg,
         'z_dim': z_dim,
         'head_dim_dict': self.style_dim_dict,
         })

    self.aux_to_rbg = nn.Sequential(
      nn.Linear(self.siren.rgb_dim, 3),
      nn.Tanh()
    )
    self.aux_to_rbg.apply(frequency_init(25))

    self.print_number_params()

    self.epoch = 0
    self.step = 0
    pass

  def forward(self,
              z,
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
              x_scale=1 / 0.2,
              y_scale=1 / 0.17,
              z_scale=1 / 0.2,
              sample_dist=None,
              lock_view_dependence=False,
              clamp_mode='relu',
              nerf_noise=0.,
              white_back=False,
              last_back=False,
              return_aux_img=True,
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
    :param x_scale:
    :param y_scale:
    :param z_scale:
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

    batch_size = z.shape[0]

    # mapping network
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_network,
                                   inputs_args=(z,),
                                   name_prefix="mapping_network.",
                                   register_itself=True,
                                   register_children=True,
                                   input_padding=60,
                                   verbose=True)
    style_dict = self.mapping_network(z)

    transformed_points, \
    transformed_ray_directions_expanded, \
    transformed_ray_origins, \
    transformed_ray_directions, \
    z_vals, \
    pitch, \
    yaw = self.get_world_points_and_direction(
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
      lock_view_dependence=lock_view_dependence
    )

    # Model prediction on course points
    coarse_output = self.siren(
      input=transformed_points,  # (b, h x w x s, 3)
      style_dict=style_dict,
      ray_directions=transformed_ray_directions_expanded,
      x_scale=x_scale,
      y_scale=y_scale,
      z_scale=z_scale,
    )
    # coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
    coarse_output = rearrange(coarse_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)

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
        input=fine_points,  # (b, h x w x s, 3)
        style_dict=style_dict,
        ray_directions=transformed_ray_directions_expanded,  # (b, h x w x s, 3)
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
      )
      # fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, 4)
      fine_output = rearrange(fine_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)

      # Combine course and fine points
      all_outputs = torch.cat([fine_output, coarse_output], dim=-2)  # (b, h x w, s, dim_rgb_sigma)
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)  # (b, h x w, s, 1)
      _, indices = torch.sort(all_z_vals, dim=-2)  # (b, h x w, s, 1)
      all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, h x w, s, 1)
      # (b, h x w, s, dim_rgb_sigma)
      all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    # Create images with NeRF
    pixels_fea, depth, weights = pigan_utils.fancy_integration(
      rgb_sigma=all_outputs,
      z_vals=all_z_vals,
      device=self.device,
      dim_rgb=self.siren.rgb_dim,
      white_back=white_back,
      last_back=last_back,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    inr_img = self.inr_net(pixels_fea, style_dict)
    inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
    # pixels = pixels.contiguous() * 2 - 1
    pitch_yaw = torch.cat([pitch, yaw], -1)

    if return_aux_img:
      # aux rgb_branch
      aux_img = self.aux_to_rbg(pixels_fea)
      aux_img = rearrange(aux_img, "b (h w) c -> b c h w", h=img_size)

      imgs = torch.cat([inr_img, aux_img])
      pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
    else:
      imgs = inr_img

    return imgs, pitch_yaw

  def print_number_params(self):
    print()
    torch_utils.print_number_params(
      models_dict={
        'siren': self.siren,
        'inr_net': self.inr_net,
        'mapping_network': self.mapping_network,
        'aux_to_rbg': self.aux_to_rbg,
        'G': self,
      })

    pass












