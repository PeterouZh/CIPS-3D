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
from exp.dev.nerf_inr.models.generator_nerf_inr import INRNetwork
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base
from exp.comm import comm_utils
from exp.comm.models import nerf_network
from exp.comm.models import multi_head_mapping
from exp.comm.models import inr_network
from exp.comm.models import fc_net


@MODEL_REGISTRY.register(name_prefix=__name__)
# @persistence.persistent_class
class GeneratorNerfINR(GeneratorNerfINR_base):
  def __init__(self,
               z_dim,
               nerf_cfg,
               inr_cfg,
               mapping_nerf_cfg,
               mapping_inr_cfg,
               fc_cfg,
               grad_lambda,
               device='cuda',
               **kwargs):
    super(GeneratorNerfINR_base, self).__init__()

    self.z_dim = z_dim
    self.grad_lambda = grad_lambda
    self.device = device

    # self.siren = NeRFNetwork(**nerf_cfg)
    # self.siren = nerf_network.NeRFNetwork(**nerf_cfg)
    self.siren = nerf_network.NeRFNetworkL(**nerf_cfg)
    # self.siren = build_model(cfg=siren_cfg, output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)

    # self.inr_net = INRNetwork(**{**inr_cfg,
    #                              "input_dim": self.siren.rgb_dim})
    # self.inr_net = inr_network.INRNetwork(**{**inr_cfg,
    #                                          "input_dim": self.siren.rgb_dim})
    # self.inr_net = inr_network.INRNetwork_Skip(**{**inr_cfg,
    #                                               "input_dim": self.siren.rgb_dim})
    # self.inr_net = inr_network.INRNetwork_Skip_Prog(**{**inr_cfg,
    #                                                    "input_dim": self.siren.rgb_dim})
    self.inr_net = inr_network.INRNetwork_Skip_CLN(**{**inr_cfg,
                                                      "input_dim": self.siren.rgb_dim})

    # self.style_dim_dict = {}
    # self.style_dim_dict.update(self.siren.style_dim_dict)
    # self.style_dim_dict.update(self.inr_net.style_dim_dict)

    # self.mapping_network = pigan_utils.MultiHeadMappingNetwork(
    #   **{**mapping_cfg,
    #      'z_dim': z_dim,
    #      'head_dim_dict': self.style_dim_dict,
    #      })
    # self.mapping_network = multi_head_mapping.MultiHeadMappingNetwork(
    #   **{**mapping_cfg,
    #      # 'z_dim': z_dim,
    #      'head_dim_dict': self.style_dim_dict,
    #      })

    self.mapping_network_nerf = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_nerf_cfg,
         'head_dim_dict': self.siren.style_dim_dict,
         })
    self.mapping_network_inr = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_inr_cfg,
         'head_dim_dict': self.inr_net.style_dim_dict,
         })

    # self.aux_to_rbg = nn.Sequential(
    #   nn.Linear(self.siren.rgb_dim, 3),
    #   nn.Tanh()
    # )
    # self.aux_to_rbg.apply(frequency_init(25))
    # self.aux_to_rbg = fc_net.FCNet(input_dim=self.siren.rgb_dim, **fc_cfg)
    # self.aux_to_rbg.apply(frequency_init(25))
    self.aux_to_rbg = inr_network.Linear_Skip_Prog(input_dim=self.siren.rgb_dim, **fc_cfg)

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
    style_dict = self.mapping_network(z)

    if psi < 1:
      avg_styles = self.generate_avg_frequencies(device=z.device)
      style_dict = self.get_truncated_freq_phase(
        raw_style_dict=style_dict, avg_style_dict=avg_styles, raw_lambda=psi)

    if grad_points is not None and grad_points < img_size ** 2:
      imgs, pitch_yaw = self.part_grad_forward(
        style_dict=style_dict,
        z=z,
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
        z=z,
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
    #
    # batch_size = z.shape[0]
    # transformed_points, \
    # transformed_ray_directions_expanded, \
    # transformed_ray_origins, \
    # transformed_ray_directions, \
    # z_vals, \
    # pitch, \
    # yaw = comm_utils.get_world_points_and_direction(
    #   batch_size=batch_size,
    #   num_steps=num_steps,
    #   img_size=img_size,
    #   fov=fov,
    #   ray_start=ray_start,
    #   ray_end=ray_end,
    #   h_stddev=h_stddev,
    #   v_stddev=v_stddev,
    #   h_mean=h_mean,
    #   v_mean=v_mean,
    #   sample_dist=sample_dist,
    #   lock_view_dependence=lock_view_dependence,
    #   device=self.device,
    # )
    #
    # # Model prediction on course points
    # coarse_output = self.siren(
    #   input=transformed_points,  # (b, h x w x s, 3)
    #   style_dict=style_dict,
    #   ray_directions=transformed_ray_directions_expanded,
    #   x_scale=x_scale,
    #   y_scale=y_scale,
    #   z_scale=z_scale,
    # )
    # # coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
    # coarse_output = rearrange(coarse_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)
    #
    # # Re-sample fine points alont camera rays, as described in NeRF
    # if hierarchical_sample:
    #   fine_points, fine_z_vals = self.get_fine_points_and_direction(
    #     coarse_output=coarse_output,
    #     z_vals=z_vals,
    #     dim_rgb=self.siren.rgb_dim,
    #     clamp_mode=clamp_mode,
    #     nerf_noise=nerf_noise,
    #     num_steps=num_steps,
    #     transformed_ray_origins=transformed_ray_origins,
    #     transformed_ray_directions=transformed_ray_directions
    #   )
    #
    #   # Model prediction on re-sampled find points
    #   fine_output = self.siren(
    #     input=fine_points,  # (b, h x w x s, 3)
    #     style_dict=style_dict,
    #     ray_directions=transformed_ray_directions_expanded,  # (b, h x w x s, 3)
    #     x_scale=x_scale,
    #     y_scale=y_scale,
    #     z_scale=z_scale,
    #   )
    #   # fine_output = fine_output.reshape(batch_size, img_size * img_size, -1, 4)
    #   fine_output = rearrange(fine_output, "b (hw s) rgb_sigma -> b hw s rgb_sigma", s=num_steps)
    #
    #   # Combine course and fine points
    #   all_outputs = torch.cat([fine_output, coarse_output], dim=-2)  # (b, h x w, s, dim_rgb_sigma)
    #   all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)  # (b, h x w, s, 1)
    #   _, indices = torch.sort(all_z_vals, dim=-2)  # (b, h x w, s, 1)
    #   all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, h x w, s, 1)
    #   # (b, h x w, s, dim_rgb_sigma)
    #   all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, all_outputs.shape[-1]))
    # else:
    #   all_outputs = coarse_output
    #   all_z_vals = z_vals
    #
    # # Create images with NeRF
    # pixels_fea, depth, weights = pigan_utils.fancy_integration(
    #   rgb_sigma=all_outputs,
    #   z_vals=all_z_vals,
    #   device=self.device,
    #   dim_rgb=self.siren.rgb_dim,
    #   white_back=white_back,
    #   last_back=last_back,
    #   clamp_mode=clamp_mode,
    #   noise_std=nerf_noise)
    #
    # inr_img = self.inr_net(pixels_fea, style_dict)
    # inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
    # # pixels = pixels.contiguous() * 2 - 1
    # pitch_yaw = torch.cat([pitch, yaw], -1)
    #
    # if return_aux_img:
    #   # aux rgb_branch
    #   aux_img = self.aux_to_rbg(pixels_fea)
    #   aux_img = rearrange(aux_img, "b (h w) c -> b c h w", h=img_size)
    #
    #   imgs = torch.cat([inr_img, aux_img])
    #   pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
    # else:
    #   imgs = inr_img
    #
    # return imgs, pitch_yaw

  def get_batch_style_dict(self, b, style_dict):
    ret_style_dict = {}
    for name, style in style_dict.items():
      ret_style_dict[name] = style[[b]]
    return ret_style_dict

  def whole_grad_forward(self,
                         style_dict,
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
                         sample_dist=None,
                         lock_view_dependence=False,
                         clamp_mode='relu',
                         nerf_noise=0.,
                         white_back=False,
                         last_back=False,
                         return_aux_img=True,
                         forward_points=None,
                         ):
    device = z.device
    batch_size = z.shape[0]


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
              img_size=img_size,
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
        img_size=img_size,
      )

    inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
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
                        sample_dist=None,
                        lock_view_dependence=False,
                        clamp_mode='relu',
                        nerf_noise=0.,
                        white_back=False,
                        last_back=False,
                        return_aux_img=True,
                        grad_points=None,
                        ):
    device = z.device
    batch_size = z.shape[0]
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
      img_size=img_size,
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
        img_size=img_size,
      )

    inr_img = comm_utils.scatter_points(idx_grad=idx_grad,
                                        points_grad=inr_img_grad,
                                        idx_no_grad=idx_no_grad,
                                        points_no_grad=inr_img_no_grad,
                                        num_points=num_points)

    inr_img = rearrange(inr_img, "b (h w) c -> b c h w", h=img_size)
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
                     img_size,
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

    inr_fea = self.grad_lambda * pixels_fea + (1 - self.grad_lambda) * pixels_fea.detach()
    inr_img = self.inr_net(inr_fea, style_dict, img_size=1024)

    if return_aux_img:
      # aux rgb_branch
      aux_fea = (1 - self.grad_lambda) * pixels_fea + self.grad_lambda * pixels_fea.detach()
      aux_img = self.aux_to_rbg(aux_fea, img_size=256)
    else:
      aux_img = None

    return inr_img, aux_img

  def mapping_network(self, z):
    style_dict = {}
    style_dict.update(self.mapping_network_nerf(z))
    style_dict.update(self.mapping_network_inr(z))
    return style_dict

  def print_number_params(self):
    print()
    torch_utils.print_number_params(
      models_dict={
        'siren': self.siren,
        'inr_net': self.inr_net,
        'mapping_network_nerf': self.mapping_network_nerf,
        'mapping_network_inr': self.mapping_network_inr,
        'aux_to_rbg': self.aux_to_rbg,
        'G': self,
      })

    pass

  def staged_forward(self, *args, **kwargs):
    raise NotImplementedError











