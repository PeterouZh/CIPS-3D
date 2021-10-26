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

    self.xyz_emb = pigan_utils.PosEmbedding(max_logscale=9, N_freqs=10)
    dim_xyz_emb = self.xyz_emb.get_out_dim()
    self.dir_emb = pigan_utils.PosEmbedding(max_logscale=3, N_freqs=4)
    dim_dir_emb = self.dir_emb.get_out_dim()

    self.style_dim_dict = {}

    self.network = nn.ModuleList([
      # FiLMLayer(3, hidden_dim),
      FiLMLayer(dim_xyz_emb, hidden_dim),
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

    self.color_layer_sine = FiLMLayer(hidden_dim + dim_dir_emb, hidden_dim)
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
      VerboseModel.forward_verbose(nn.Sequential(OrderedDict([('gridwarper', self.gridwarper),
                                                              ('xyz_emb', self.xyz_emb)])),
                                   inputs_args=(input,),
                                   name_prefix="xyz.")
    input = self.gridwarper(input)
    xyz_emb = self.xyz_emb(input)

    x = xyz_emb
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
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.dir_emb,
                                   inputs_args=(ray_directions, ),
                                   name_prefix='dir_emb')
    dir_emb = self.dir_emb(ray_directions)

    frequencies, phase_shifts = self.get_freq_phase(
      style_dict=style_dict, name=f"{self.name_prefix}_color_layer_sine")
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(torch.cat([dir_emb, x], dim=-1),
                                                frequencies,
                                                phase_shifts),
                                   name_prefix=f"color_layer_sine.")
    rbg_sine = self.color_layer_sine(
      torch.cat([dir_emb, x], dim=-1),
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
    self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)
    self.style_dim_dict[f'{name_prefix}_network'] = len(self.network) * hidden_dim * 2

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



@MODEL_REGISTRY.register(name_prefix=__name__)
# @persistence.persistent_class
class GeneratorNerfINR(nn.Module):
  def __init__(self,
               z_dim,
               nerf_cfg,
               inr_cfg,
               mapping_cfg,
               **kwargs):
    super().__init__()
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

  def print_number_params(self):
    print()
    torch_utils.print_number_params(
      models_dict={
        'siren': self.siren,
        'inr_net': self.inr_net,
        'mapping_network': self.mapping_network,
        'G': self,
      })

    pass

  def set_device(self, device):
    self.device = device
    self.siren.device = device

    self.generate_avg_frequencies()
    pass

  def get_xyz_range(self,
                    num_samples,
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
                    sample_dist=None,
                    **kwargs):
    """
        """

    batch_size = z.shape[0]

    xyz_minmax_mean = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]

    for i in tqdm.tqdm(range(num_samples)):
      # Generate initial camera rays and sample points.
      with torch.no_grad():
        points_cam, z_vals, rays_d_cam = pigan_utils.get_initial_rays_trig(
          batch_size,
          num_steps,
          resolution=(img_size, img_size),
          device=self.device,
          fov=fov,
          ray_start=ray_start,
          ray_end=ray_end)  # batch_size, pixels, num_steps, 1
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
          pigan_utils.transform_sampled_points(points_cam,
                                               z_vals,
                                               rays_d_cam,
                                               h_stddev=h_stddev,
                                               v_stddev=v_stddev,
                                               h_mean=h_mean,
                                               v_mean=v_mean,
                                               device=self.device,
                                               mode=sample_dist)
        xyz_list = transformed_points.unbind(3)
        for minmax_mean, axis_v in zip(xyz_minmax_mean, xyz_list):
          min_v, max_v, mean_v = minmax_mean
          minmax_mean[0] = min(min_v, axis_v.min().item())
          minmax_mean[1] = max(max_v, axis_v.max().item())
          minmax_mean[2] = minmax_mean[2] + axis_v.mean().item()
    for minmax_mean in xyz_minmax_mean:
      minmax_mean[2] = minmax_mean[2] / num_samples

    for minmax_mean, axis_name in zip(xyz_minmax_mean, 'xyz'):
      minmax_mean_str = f"{axis_name}: ({minmax_mean[0]:.2f}, {minmax_mean[1]:.2f}, {minmax_mean[2]:.2f})"
      print(minmax_mean_str)
    pass

  @torch.no_grad()
  def get_world_points_and_direction(self,
                                     batch_size,
                                     num_steps,
                                     img_size,
                                     fov,
                                     ray_start,
                                     ray_end,
                                     h_stddev,
                                     v_stddev,
                                     h_mean,
                                     v_mean,
                                     sample_dist,
                                     lock_view_dependence,
                                     ):
    """
    Generate sample points and camera rays in the world coordinate system.

    :param batch_size:
    :param num_steps: number of samples for each ray
    :param img_size:
    :param fov:
    :param ray_start:
    :param ray_end:
    :param h_stddev:
    :param v_stddev:
    :param h_mean:
    :param v_mean:
    :param sample_dist: mode for sample_camera_positions
    :param lock_view_dependence:
    :return:
    - transformed_points: (b, h x w x num_steps, 3), has been perturbed
    - transformed_ray_directions_expanded: (b, h x w x num_steps, 3)
    - transformed_ray_origins: (b, h x w, 3)
    - transformed_ray_directions: (b, h x w, 3)
    - z_vals: (b, h x w, num_steps, 1), has been perturbed
    - pitch: (b, 1)
    - yaw: (b, 1)
    """

    # Generate initial camera rays and sample points.
    # batch_size, pixels, num_steps, 1
    points_cam, z_vals, rays_d_cam = pigan_utils.get_initial_rays_trig(
      bs=batch_size,
      num_steps=num_steps,
      resolution=(img_size, img_size),
      device=self.device,
      fov=fov,
      ray_start=ray_start,
      ray_end=ray_end)

    transformed_points, \
    z_vals, \
    transformed_ray_directions, \
    transformed_ray_origins, \
    pitch, yaw = pigan_utils.transform_sampled_points(points_cam,
                                                      z_vals,
                                                      rays_d_cam,
                                                      h_stddev=h_stddev,
                                                      v_stddev=v_stddev,
                                                      h_mean=h_mean,
                                                      v_mean=v_mean,
                                                      device=self.device,
                                                      mode=sample_dist)

    # transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    # transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
    # transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
    #   batch_size, img_size * img_size * num_steps, 3)
    transformed_ray_directions_expanded = repeat(
      transformed_ray_directions, "b hw xyz -> b (hw s) xyz", s=num_steps)
    if lock_view_dependence:
      transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
      transformed_ray_directions_expanded[..., -1] = -1

    # transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)
    transformed_points = rearrange(transformed_points, "b hw s xyz -> b (hw s) xyz")

    ret = (transformed_points, transformed_ray_directions_expanded,
           transformed_ray_origins, transformed_ray_directions, z_vals,
           pitch, yaw)
    return ret


  @torch.no_grad()
  def get_fine_points_and_direction(self,
                                    coarse_output,
                                    z_vals,
                                    dim_rgb,
                                    clamp_mode,
                                    nerf_noise,
                                    num_steps,
                                    transformed_ray_origins,
                                    transformed_ray_directions,
                                    ):
    """

    :param coarse_output: (b, h x w, num_samples, rgb_sigma)
    :param z_vals: (b, h x w, num_samples, 1)
    :param clamp_mode:
    :param nerf_noise:
    :param num_steps:
    :param transformed_ray_origins: (b, h x w, 3)
    :param transformed_ray_directions: (b, h x w, 3)
    :return:
    - fine_points: (b, h x w x num_steps, 3)
    - fine_z_vals: (b, h x w, num_steps, 1)
    """

    batch_size = coarse_output.shape[0]

    _, _, weights = pigan_utils.fancy_integration(
      rgb_sigma=coarse_output,
      z_vals=z_vals,
      device=self.device,
      dim_rgb=dim_rgb,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    # weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
    weights = rearrange(weights, "b hw s 1 -> (b hw) s") + 1e-5

    #### Start new importance sampling
    # z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
    z_vals = rearrange(z_vals, "b hw s 1 -> (b hw) s")
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    # z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    # z_vals = rearrange(z_vals, "(b hw) s -> b hw s 1", b=batch_size)
    fine_z_vals = pigan_utils.sample_pdf(bins=z_vals_mid,
                                         weights=weights[:, 1:-1],
                                         N_importance=num_steps,
                                         det=False).detach()
    # fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
    fine_z_vals = rearrange(fine_z_vals, "(b hw) s -> b hw s 1", b=batch_size)

    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                  transformed_ray_directions.unsqueeze(2).contiguous() * \
                  fine_z_vals.expand(-1, -1, -1, 3).contiguous()
    # fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
    fine_points = rearrange(fine_points, "b hw s c -> b (hw s) c")

    # if lock_view_dependence:
    #   transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    #   transformed_ray_directions_expanded[..., -1] = -1
    #### end new importance sampling
    return fine_points, fine_z_vals

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
                                   inputs_args=(z, ),
                                   name_prefix="mapping_network.",
                                   register_itself=True,
                                   register_children=True,
                                   input_padding=60,
                                   verbose=True)
    style_dict = self.mapping_network(z)

    transformed_points, \
    transformed_ray_directions_expanded, \
    transformed_ray_origins,\
    transformed_ray_directions,\
    z_vals,\
    pitch,\
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
      input=transformed_points, # (b, h x w x s, 3)
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
        input=fine_points, # (b, h x w x s, 3)
        style_dict=style_dict,
        ray_directions=transformed_ray_directions_expanded, # (b, h x w x s, 3)
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
      )
      # fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, 4)
      fine_output = rearrange(fine_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)

      # Combine course and fine points
      all_outputs = torch.cat([fine_output, coarse_output], dim=-2) # (b, h x w, s, dim_rgb_sigma)
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2) # (b, h x w, s, 1)
      _, indices = torch.sort(all_z_vals, dim=-2) # (b, h x w, s, 1)
      all_z_vals = torch.gather(all_z_vals, -2, indices) # (b, h x w, s, 1)
      # (b, h x w, s, dim_rgb_sigma)
      all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    # Create images with NeRF
    pixels, depth, weights = pigan_utils.fancy_integration(
      rgb_sigma=all_outputs,
      z_vals=all_z_vals,
      device=self.device,
      dim_rgb=self.siren.rgb_dim,
      white_back=white_back,
      last_back=last_back,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    pixels = self.inr_net(pixels, style_dict)

    pixels = rearrange(pixels, "b (h w) c -> b c h w", h=img_size)
    # pixels = pixels.contiguous() * 2 - 1

    pitch_yaw = torch.cat([pitch, yaw], -1)
    return pixels, pitch_yaw

  def generate_avg_frequencies(self, num_samples=10000, device='cuda'):
    """Calculates average frequencies and phase shifts"""

    z = torch.randn((num_samples, self.z_dim), device=device)
    with torch.no_grad():
      style_dict = self.mapping_network(z)

    avg_styles= {}
    for name, style in style_dict.items():
      avg_styles[name] = style.mean(0, keepdim=True)

    self.avg_styles = avg_styles
    # self.avg_frequencies = frequencies.mean(0, keepdim=True)
    # self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
    # return self.avg_frequencies, self.avg_phase_shifts
    return avg_styles

  def get_truncated_freq_phase(self,
                               raw_style_dict,
                               avg_style_dict,
                               raw_lambda):
    # truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
    # truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

    truncated_style_dict = {}
    for name, avg_style in avg_style_dict.items():
      raw_style = raw_style_dict[name]
      truncated_style = avg_style + raw_lambda * (raw_style - avg_style)
      truncated_style_dict[name] = truncated_style
    return truncated_style_dict

  def staged_forward(self,
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
                     clamp_mode,
                     nerf_noise,
                     psi=1,
                     lock_view_dependence=False,
                     max_batch_size=50000,
                     depth_map=False,
                     near_clip=0,
                     far_clip=2,
                     sample_dist=None,
                     hierarchical_sample=False,
                     white_back=False,
                     last_back=False,
                     fill_mode=None,
                     **kwargs):
    """
    Similar to forward but used for inference.
    Calls the model sequencially using max_batch_size to limit memory usage.

    :param z: (b, dim_z)
    :param img_size:
    :param fov:
    :param ray_start:
    :param ray_end:
    :param num_steps:
    :param h_stddev:
    :param v_stddev:
    :param h_mean:
    :param v_mean:
    :param clamp_mode:
    :param nerf_noise:
    :param psi: [0, 1], 0: use the avg_style
    :param lock_view_dependence:
    :param max_batch_size:
    :param depth_map:
    :param near_clip:
    :param far_clip:
    :param sample_dist:
    :param hierarchical_sample:
    :param white_back:
    :param last_back:
    :param fill_mode:
    :param kwargs:
    :return:
    - pixels:
    - depth_map:
    """

    batch_size = z.shape[0]

    self.generate_avg_frequencies()

    with torch.no_grad():

      raw_style_dict = self.mapping_network(z)

      truncated_style_dict = self.get_truncated_freq_phase(
        raw_style_dict=raw_style_dict, avg_style_dict=self.avg_styles, raw_lambda=psi)

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

      # Sequentially evaluate siren with max_batch_size to avoid OOM
      coarse_output = self.siren.staged_forward(
        transformed_points=transformed_points,
        transformed_ray_directions_expanded=transformed_ray_directions_expanded,
        style_dict=truncated_style_dict,
        max_points=max_batch_size,
        num_steps=num_steps)

      # debug_output = self.siren(
      #   input=transformed_points,  # (b, h x w x s, 3)
      #   style_dict=truncated_style_dict,
      #   ray_directions=transformed_ray_directions_expanded,
      # )
      # debug_output = rearrange(debug_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)

      if hierarchical_sample:
        with torch.no_grad():
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
          fine_output = self.siren.staged_forward(
            transformed_points=fine_points,  # (b, h x w x s, 3)
            transformed_ray_directions_expanded=transformed_ray_directions_expanded,  # (b, h x w x s, 3)
            style_dict=truncated_style_dict,
            max_points=max_batch_size,
            num_steps=num_steps)
          # fine_output = rearrange(fine_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)

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
      pixels, depth, weights = pigan_utils.fancy_integration(
        rgb_sigma=all_outputs,
        z_vals=all_z_vals,
        device=self.device,
        dim_rgb=self.siren.rgb_dim,
        white_back=white_back,
        last_back=last_back,
        clamp_mode=clamp_mode,
        fill_mode=fill_mode,
        noise_std=nerf_noise)

      pixels = self.inr_net(pixels, truncated_style_dict)

      pixels = rearrange(pixels, "b (h w) c -> b c h w", h=img_size)
      pixels = pixels.contiguous().cpu().to(dtype=torch.float32)
      depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

    return pixels, depth_map

  # Used for rendering interpolations
  def staged_forward_with_frequencies(self,
                                      truncated_frequencies,
                                      truncated_phase_shifts,
                                      img_size,
                                      fov,
                                      ray_start,
                                      ray_end,
                                      num_steps,
                                      h_stddev,
                                      v_stddev,
                                      h_mean,
                                      v_mean,
                                      psi=0.7,
                                      lock_view_dependence=False,
                                      max_batch_size=50000,
                                      depth_map=False,
                                      near_clip=0,
                                      far_clip=2,
                                      sample_dist=None,
                                      hierarchical_sample=False,
                                      **kwargs):
    batch_size = truncated_frequencies.shape[0]

    with torch.no_grad():
      points_cam, z_vals, rays_d_cam = pigan_utils.get_initial_rays_trig(
        batch_size,
        num_steps,
        resolution=(img_size, img_size),
        device=self.device,
        fov=fov,
        ray_start=ray_start,
        ray_end=ray_end)  # batch_size, pixels, num_steps, 1
      transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
        pigan_utils.transform_sampled_points(points_cam,
                                             z_vals,
                                             rays_d_cam,
                                             h_stddev=h_stddev,
                                             v_stddev=v_stddev,
                                             h_mean=h_mean,
                                             v_mean=v_mean,
                                             device=self.device,
                                             mode=sample_dist)

      transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
      transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
      transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
        batch_size, img_size * img_size * num_steps, 3)
      transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)

      if lock_view_dependence:
        transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
        transformed_ray_directions_expanded[..., -1] = -1

      # BATCHED SAMPLE
      coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
      for b in range(batch_size):
        head = 0
        while head < transformed_points.shape[1]:
          tail = head + max_batch_size
          coarse_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
            transformed_points[b:b + 1, head:tail],
            truncated_frequencies[b:b + 1],
            truncated_phase_shifts[b:b + 1],
            ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
          head += max_batch_size

      coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
      # END BATCHED SAMPLE

      if hierarchical_sample:
        with torch.no_grad():
          transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
          _, _, weights = pigan_utils.fancy_integration(
            coarse_output,
            z_vals,
            device=self.device,
            clamp_mode=kwargs['clamp_mode'],
            noise_std=kwargs['nerf_noise'])

          weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
          z_vals = z_vals.reshape(batch_size * img_size * img_size,
                                  num_steps)  # We squash the dimensions here. This means we importance sample for every batch for every ray
          z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
          z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
          fine_z_vals = pigan_utils.sample_pdf(
            z_vals_mid,
            weights[:, 1:-1],
            num_steps, det=False).detach().to(self.device)  # batch_size, num_pixels**2, num_steps
          fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

          fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                        transformed_ray_directions.unsqueeze(2).contiguous() * \
                        fine_z_vals.expand(-1, -1, -1, 3).contiguous()  # dimensions here not matching
          fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
          #### end new importance sampling

        if lock_view_dependence:
          transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
          transformed_ray_directions_expanded[..., -1] = -1
        # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
        # BATCHED SAMPLE
        fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
        for b in range(batch_size):
          head = 0
          while head < fine_points.shape[1]:
            tail = head + max_batch_size
            fine_output[b:b + 1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
              fine_points[b:b + 1, head:tail],
              truncated_frequencies[b:b + 1],
              truncated_phase_shifts[b:b + 1],
              ray_directions=transformed_ray_directions_expanded[b:b + 1, head:tail])
            head += max_batch_size

        fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
        # END BATCHED SAMPLE

        all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
        all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
        _, indices = torch.sort(all_z_vals, dim=-2)
        all_z_vals = torch.gather(all_z_vals, -2, indices)
        all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
      else:
        all_outputs = coarse_output
        all_z_vals = z_vals

      pixels, depth, weights = pigan_utils.fancy_integration(
        all_outputs,
        all_z_vals,
        device=self.device,
        white_back=kwargs.get('white_back', False),
        clamp_mode=kwargs['clamp_mode'],
        last_back=kwargs.get('last_back', False),
        fill_mode=kwargs.get('fill_mode', None),
        noise_std=kwargs['nerf_noise'])
      depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

      pixels = pixels.reshape((batch_size, img_size, img_size, 3))
      pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

    return pixels, depth_map

  def forward_with_frequencies(self,
                               frequencies,
                               phase_shifts,
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
                               **kwargs):
    batch_size = frequencies.shape[0]

    points_cam, z_vals, rays_d_cam = pigan_utils.get_initial_rays_trig(
      batch_size,
      num_steps,
      resolution=(img_size, img_size),
      device=self.device,
      fov=fov,
      ray_start=ray_start,
      ray_end=ray_end)  # batch_size, pixels, num_steps, 1
    transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
      pigan_utils.transform_sampled_points(points_cam,
                                           z_vals,
                                           rays_d_cam,
                                           h_stddev=h_stddev,
                                           v_stddev=v_stddev,
                                           h_mean=h_mean,
                                           v_mean=v_mean,
                                           device=self.device,
                                           mode=sample_dist)

    transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
    transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
      batch_size, img_size * img_size * num_steps, 3)
    transformed_points = transformed_points.reshape(batch_size, img_size * img_size * num_steps, 3)

    if lock_view_dependence:
      transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
      transformed_ray_directions_expanded[..., -1] = -1

    coarse_output = self.siren.forward_with_frequencies_phase_shifts(
      transformed_points,
      frequencies,
      phase_shifts,
      ray_directions=transformed_ray_directions_expanded) \
      .reshape(batch_size, img_size * img_size, num_steps, 4)

    if hierarchical_sample:
      with torch.no_grad():
        transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
        _, _, weights = pigan_utils.fancy_integration(
          coarse_output,
          z_vals,
          device=self.device,
          clamp_mode=kwargs['clamp_mode'],
          noise_std=kwargs['nerf_noise'])

        weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
        #### Start new importance sampling
        # RuntimeError: Sizes of tensors must match except in dimension 1.
        # Got 3072 and 6144 (The offending index is 0)
        z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
        # We squash the dimensions here. This means we importance sample for every batch for every ray
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
        fine_z_vals = pigan_utils.sample_pdf(z_vals_mid,
                                             weights[:, 1:-1],
                                             num_steps,
                                             det=False).detach()  # batch_size, num_pixels**2, num_steps
        fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

        fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                      transformed_ray_directions.unsqueeze(2).contiguous() * \
                      fine_z_vals.expand(-1, -1, -1, 3).contiguous()  # dimensions here not matching
        fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
        #### end new importance sampling

        if lock_view_dependence:
          transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
          transformed_ray_directions_expanded[..., -1] = -1

      fine_output = self.siren.forward_with_frequencies_phase_shifts(
        fine_points,
        frequencies,
        phase_shifts,
        ray_directions=transformed_ray_directions_expanded) \
        .reshape(batch_size, img_size * img_size, -1, 4)

      all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
      _, indices = torch.sort(all_z_vals, dim=-2)
      all_z_vals = torch.gather(all_z_vals, -2, indices)
      # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
      all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    pixels, depth, weights = pigan_utils.fancy_integration(
      all_outputs,
      all_z_vals,
      device=self.device,
      white_back=kwargs.get('white_back', False),
      last_back=kwargs.get('last_back', False),
      clamp_mode=kwargs['clamp_mode'],
      noise_std=kwargs['nerf_noise'])

    pixels = pixels.reshape((batch_size, img_size, img_size, 3))
    pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

    return pixels, torch.cat([pitch, yaw], -1)
