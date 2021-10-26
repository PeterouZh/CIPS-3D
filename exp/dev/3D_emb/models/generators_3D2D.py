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


class Embedding3D(nn.Module):
  def __init__(self,
               channel=128,
               size=64):
    super().__init__()

    # (b, c, d, h, w)
    self.emb = nn.Parameter(torch.randn(1, channel, size, size, size))

    pass

  def forward(self,
              points,
              x_scale,
              y_scale,
              z_scale,
              **kwargs
              ):
    """
        points: (b, hxwxd, 3) xyz
        point_shape: [b, h, w, d, 3]
        """
    # b, h, w, d, _ = point_shape
    b = points.shape[0]
    # (b, c, d, h, w)
    emb = self.emb.expand(b, -1, -1, -1, -1)

    xyz_scale = torch.tensor([x_scale, y_scale, z_scale], device=points.device)
    points = points * xyz_scale
    # points = rearrange(points, "b (h w d) xyz -> b d h w xyz", h=h, w=w, d=d)
    points = rearrange(points, "b n xyz -> b n 1 1 xyz")

    out = F.grid_sample(
      emb,
      points,
      padding_mode='border',
      mode='bilinear',
      align_corners=False
    )
    # out = rearrange(out, "b c d h w -> b (h w d) c")
    out = rearrange(out, "b c n 1 1 -> b n c")
    return out


class SinActivation(nn.Module):
  def __init__(self, ):
    super(SinActivation, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class CoordFC(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim):
    super().__init__()

    self.layer = nn.Linear(input_dim, hidden_dim)
    nn.init.uniform_(self.layer.weight, -np.sqrt(9 / input_dim), np.sqrt(9 / input_dim))

    self.act = SinActivation()
    pass

  def forward(self,
              x):
    x = self.layer(x)
    out = self.act(x)
    return out


# @MODEL_REGISTRY.register(name_prefix=__name__)
# @persistence.persistent_class
class SPATIALSIRENBASELINE(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __init__(self,
               input_dim=3,
               z_dim=100,
               hidden_dim=256,
               emb_dim=128,
               rgb_dim=3,
               device=None,
               **kwargs):
    """

    :param input_dim:
    :param z_dim:
    :param hidden_dim:
    :param emb_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super().__init__()
    self.device = device
    self.input_dim = input_dim
    self.z_dim = z_dim
    self.hidden_dim = hidden_dim

    self.emb3d = Embedding3D(channel=emb_dim, size=64)
    self.coord_fc = CoordFC(input_dim=3, hidden_dim=emb_dim)

    self.network = nn.ModuleList([
      # FiLMLayer(3, hidden_dim),
      FiLMLayer(emb_dim * 2, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
      FiLMLayer(hidden_dim, hidden_dim),
    ])
    self.network.apply(frequency_init(25))
    self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    self.final_layer.apply(frequency_init(25))

    self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
    self.color_layer_sine.apply(frequency_init(25))

    self.color_layer_linear = nn.Sequential(
      nn.Linear(hidden_dim, rgb_dim),
      nn.Sigmoid())
    self.color_layer_linear.apply(frequency_init(25))

    self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1) * hidden_dim * 2)

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)
    pass

  def forward(self,
              input,
              z,
              ray_directions,
              **kwargs):
    """

    :param input: points xyz, (b, num_points, 3)
    :param z: noise, (b, dim_z)
    :param ray_directions: (b, num_points, 3)
    :param kwargs:
    :return:
    - out: rgb(3) + sigma(1) (b, num_points, 4)
    """

    # mapping network
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_network,
                                   submodels=['network', ],
                                   inputs_args=(z,),
                                   name_prefix="mapping_network.",
                                   register_itself=True,
                                   verbose=True)
    frequencies, phase_shifts = self.mapping_network(z)

    out = self.forward_with_frequencies_phase_shifts(
      input,
      frequencies,
      phase_shifts,
      ray_directions,
      **kwargs)

    if global_cfg.tl_debug:
      self.print_number_params()
    return out

  def print_number_params(self):
    print()
    torch_utils.print_number_params({'mapping_network': self.mapping_network,
                                     'emb3d': self.emb3d,
                                     'coord_fc': self.coord_fc,
                                     'network': self.network,
                                     'final_layer': self.final_layer,
                                     'color_layer_sine': self.color_layer_sine,
                                     'color_layer_linear': self.color_layer_linear,
                                     'generator': self})
    pass

  def forward_with_frequencies_phase_shifts(self,
                                            input,
                                            frequencies,
                                            phase_shifts,
                                            ray_directions,
                                            x_scale=1 / 0.2,
                                            y_scale=1 / 0.17,
                                            z_scale=1 / 0.2,
                                            **kwargs):
    frequencies = frequencies * 15 + 30

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.emb3d,
                                   inputs_args=(input,
                                                x_scale,
                                                y_scale,
                                                z_scale),
                                   name_prefix="emb3d")
    emb = self.emb3d(input,
                     x_scale,
                     y_scale,
                     z_scale)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.gridwarper,
                                   inputs_args=(input,),
                                   name_prefix="gridwarper")
    input = self.gridwarper(input)
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.coord_fc,
                                   inputs_args=(input,),
                                   name_prefix="coord_fc.")
    input = self.coord_fc(input)

    # x = input
    x = torch.cat([input, emb], dim=2)

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

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_sine,
                                   inputs_args=(torch.cat([ray_directions, x], dim=-1),
                                                frequencies[..., -self.hidden_dim:],
                                                phase_shifts[..., -self.hidden_dim:]),
                                   name_prefix=f"color_layer_sine.")
    rbg = self.color_layer_sine(
      torch.cat([ray_directions, x], dim=-1),
      frequencies[..., -self.hidden_dim:],
      phase_shifts[..., -self.hidden_dim:])

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.color_layer_linear,
                                   inputs_args=(rbg,),
                                   name_prefix="color_layer_linear.")
    # rbg = torch.sigmoid(self.color_layer_linear(rbg))
    rbg = self.color_layer_linear(rbg)

    out = torch.cat([rbg, sigma], dim=-1)
    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class ImplicitGenerator3d(nn.Module):
  def __init__(self,
               z_dim,
               siren_cfg,
               **kwargs):
    super().__init__()
    self.z_dim = z_dim
    self.siren = SPATIALSIRENBASELINE(output_dim=4,
                                      z_dim=self.z_dim,
                                      input_dim=3,
                                      device=None,
                                      **siren_cfg)
    # self.siren = build_model(cfg=siren_cfg, output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
    self.print_number_params()

    self.epoch = 0
    self.step = 0
    pass

  def print_number_params(self):
    self.siren.print_number_params()
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
      z=z,
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
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        num_steps=num_steps,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions
      )

      # Model prediction on re-sampled find points
      fine_output = self.siren(
        fine_points, # (b, h x w x s, 3)
        z,
        ray_directions=transformed_ray_directions_expanded, # (b, h x w x s, 3)
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
      ).reshape(batch_size, img_size * img_size, -1, 4)

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
      white_back=white_back,
      last_back=last_back,
      clamp_mode=clamp_mode,
      noise_std=nerf_noise)

    # pixels = pixels.reshape((batch_size, img_size, img_size, 3))
    # pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
    pixels = rearrange(pixels, "b (h w) c -> b c h w", h=img_size)
    pixels = pixels.contiguous() * 2 - 1

    pitch_yaw = torch.cat([pitch, yaw], -1)
    return pixels, pitch_yaw

  def generate_avg_frequencies(self):
    """Calculates average frequencies and phase shifts"""

    z = torch.randn((10000, self.z_dim), device=self.siren.device)
    with torch.no_grad():
      frequencies, phase_shifts = self.siren.mapping_network(z)
    self.avg_frequencies = frequencies.mean(0, keepdim=True)
    self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
    return self.avg_frequencies, self.avg_phase_shifts

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
                     psi=1,
                     lock_view_dependence=False,
                     max_batch_size=50000,
                     depth_map=False,
                     near_clip=0,
                     far_clip=2,
                     sample_dist=None,
                     hierarchical_sample=False,
                     **kwargs):
    """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

    batch_size = z.shape[0]

    self.generate_avg_frequencies()

    with torch.no_grad():

      raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

      truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
      truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

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

      # Sequentially evaluate siren with max_batch_size to avoid OOM
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
          z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
          z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
          z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
          fine_z_vals = pigan_utils.sample_pdf(z_vals_mid,
                                               weights[:, 1:-1],
                                               num_steps,
                                               det=False).detach().to(self.device)
          fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

          fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                        transformed_ray_directions.unsqueeze(2).contiguous() * \
                        fine_z_vals.expand(-1, -1, -1, 3).contiguous()
          fine_points = fine_points.reshape(batch_size, img_size * img_size * num_steps, 3)
          #### end new importance sampling

        if lock_view_dependence:
          transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
          transformed_ray_directions_expanded[..., -1] = -1

        # Sequentially evaluate siren with max_batch_size to avoid OOM
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

