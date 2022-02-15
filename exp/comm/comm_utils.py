import json
from einops import rearrange, repeat
import random
import numpy as np
import math
import logging
from PIL import Image
from easydict import EasyDict
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms.functional as trans_f

from exp.pigan import pigan_utils


@torch.no_grad()
def to_pil(frame):
  frame = (frame.squeeze() + 1) * 0.5
  frame_pil = trans_f.to_pil_image(frame)
  return frame_pil


@torch.no_grad()
def inr_layer_swapping(swapped_net, target_net, gamma_target, swapped_layers, verbose=True):
  """
  Swapping inr_net params.

  :param swapped_net:
  :param target_net:
  :param gamma:
  :param swapped_layers:
  :return:
  """
  start_names = []
  for name in swapped_layers:
    start_names.append(f"network.{name}")
    start_names.append(f"to_rgbs.{name}")
  start_names = tuple(start_names)
  if verbose:
    st.write(f"Layer swapping: {start_names}")

  target_params = target_net.state_dict()
  for param_name, param in swapped_net.named_parameters():
    if param_name.startswith(start_names):
      target_param = target_params[param_name]
      param.data.copy_(param.data * (1 - gamma_target) + target_param.data * gamma_target)
  pass

def get_metadata_from_json(metafile,
                           return_raw=False,
                           image_size=256,
                           num_steps=12,
                           psi=0.9,
                           fov=12,
                           v_stddev=0,
                           h_stddev=0,
                           last_back=False,
                           nerf_noise=0
                           ):
  with open(metafile, 'r') as f:
    curriculum = json.load(f)
    curriculum = EasyDict(curriculum)

  if return_raw:
    return curriculum

  curriculum['img_size'] = image_size
  curriculum['num_steps'] = num_steps
  curriculum['psi'] = psi
  curriculum['fov'] = fov
  curriculum['v_stddev'] = v_stddev
  curriculum['h_stddev'] = h_stddev
  curriculum['last_back'] = last_back
  curriculum['nerf_noise'] = nerf_noise
  return curriculum


def get_yaw_pitch_by_xyz(x, y, z):
  yaw = math.atan2(z, x)
  pitch = math.atan2(math.sqrt(x ** 2 + z ** 2), y)
  return yaw, pitch

def get_circle_camera_pos_and_lookup(r=1,
                                     alpha=3.141592 / 6,
                                     num_samples=36,
                                     periods=2):
  num_samples = num_samples * periods
  xyz = np.zeros((num_samples, 3), dtype=np.float32)

  xyz[:, 2] = r * math.cos(alpha)
  z_sin = r * math.sin(alpha)

  for idx, t in enumerate(np.linspace(1, 0, num_samples)):
    beta = t * 2 * math.pi * periods
    xyz[idx, 0] = z_sin * math.cos(beta)
    xyz[idx, 1] = z_sin * math.sin(beta)
  lookup = - xyz

  yaws = np.zeros(num_samples)
  pitchs = np.zeros(num_samples)
  for idx, (x, y, z) in enumerate(xyz):
    yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
    yaws[idx] = yaw
    pitchs[idx] = pitch

  return xyz, lookup, yaws, pitchs

def _get_translate_distance(num_samples, translate_dist):
  num_samples_every = num_samples // 4
  dist_list = []

  dist_list.append(np.linspace(0, translate_dist, num_samples_every))
  dist_list.append(np.linspace(translate_dist, 0, num_samples_every))
  dist_list.append(np.linspace(0, -translate_dist, num_samples_every))
  dist_list.append(np.linspace(-translate_dist, 0, num_samples_every))
  dist_list = np.concatenate(dist_list, axis=0)
  return dist_list

def get_translate_circle_camera_pos_and_lookup(r=1,
                                               num_samples_translate=36,
                                               translate_dist=0.5,
                                               alpha=3.141592 / 6,
                                               num_samples=36,
                                               periods=2):
  trans_dist = _get_translate_distance(num_samples=num_samples_translate, translate_dist=translate_dist)
  num_samples_translate = len(trans_dist)

  translateX_xyz = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateX_lookup = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateX_lookup[:, 2] = -1
  for idx, t in enumerate(trans_dist):
    translateX_xyz[idx, 0] = t
    translateX_xyz[idx, 2] = r * math.cos(alpha)

  translateY_xyz = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateY_xyz[:, 1] = translateX_xyz[:, 0]
  translateY_xyz[:, 2] = translateX_xyz[:, 2]
  translateY_lookup = translateX_lookup

  num_samples = num_samples * periods
  xyz = np.zeros((num_samples, 3), dtype=np.float32)

  xyz[:, 2] = r * math.cos(alpha)
  z_sin = r * math.sin(alpha)

  for idx, t in enumerate(np.linspace(1, 0, num_samples)):
    beta = t * 2 * math.pi * periods
    xyz[idx, 0] = z_sin * math.cos(beta)
    xyz[idx, 1] = z_sin * math.sin(beta)
  lookup = - xyz

  xyz = np.concatenate((translateX_xyz, translateY_xyz, xyz), axis=0)
  lookup = np.concatenate((translateX_lookup, translateY_lookup, lookup), axis=0)

  num_samples = len(xyz)
  yaws = np.zeros(num_samples)
  pitchs = np.zeros(num_samples)
  for idx, (x, y, z) in enumerate(xyz):
    yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
    yaws[idx] = yaw
    pitchs[idx] = pitch

  return xyz, lookup, yaws, pitchs, num_samples_translate


def get_translate_rotate(r=1,
                         num_samples_translate=36,
                         translate_dist=0.5,
                         alpha=3.141592 / 6,
                         num_samples=36,
                         periods=1):
  trans_dist = _get_translate_distance(num_samples=num_samples_translate, translate_dist=translate_dist)
  num_samples_translate = len(trans_dist)

  translateX_lookup = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateX_lookup[:, 2] = -1
  translateX_up = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateX_up[:, 1] = 1
  translateX_xyz = np.zeros((num_samples_translate, 3), dtype=np.float32)
  for idx, t in enumerate(trans_dist):
    translateX_xyz[idx, 0] = t
    translateX_xyz[idx, 2] = r * math.cos(alpha)

  translateY_xyz = np.zeros((num_samples_translate, 3), dtype=np.float32)
  translateY_xyz[:, 1] = translateX_xyz[:, 0]
  translateY_xyz[:, 2] = translateX_xyz[:, 2]
  translateY_lookup = translateX_lookup
  translateY_up = translateX_up

  num_samples = num_samples * periods
  xyz = np.zeros((num_samples, 3), dtype=np.float32)
  xyz[:, 2] = r * math.cos(alpha)
  lookup = - xyz
  up = np.zeros((num_samples, 3), dtype=np.float32)

  for idx, beta in enumerate(np.linspace(0, 2*math.pi*periods, num_samples)):
    up[idx, 0] = - math.sin(beta)
    up[idx, 1] = math.cos(beta)

  xyz = np.concatenate((translateX_xyz, translateY_xyz, xyz), axis=0)
  lookup = np.concatenate((translateX_lookup, translateY_lookup, lookup), axis=0)
  up = np.concatenate((translateX_up, translateY_up, up), axis=0)

  num_samples = len(xyz)
  yaws = np.zeros(num_samples)
  pitchs = np.zeros(num_samples)
  for idx, (x, y, z) in enumerate(xyz):
    yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
    yaws[idx] = yaw
    pitchs[idx] = pitch

  return xyz, lookup, up, yaws, pitchs, num_samples_translate


def get_yaw_camera_pos_and_lookup(r=1,
                                  num_samples=36,
                                  ):
  xyz = np.zeros((num_samples, 3), dtype=np.float32)

  for idx, theta in enumerate(np.linspace(1, math.pi-1, num_samples)):
    xyz[idx, 0] = r * math.cos(theta)
    xyz[idx, 1] = 0
    xyz[idx, 2] = r * math.sin(theta)
  lookup = - xyz

  yaws = np.zeros(num_samples)
  pitchs = np.zeros(num_samples)
  for idx, (x, y, z) in enumerate(xyz):
    yaw, pitch = get_yaw_pitch_by_xyz(x, y, z)
    yaws[idx] = yaw
    pitchs[idx] = pitch

  return xyz, lookup, yaws, pitchs


def scatter_points(idx_grad,
                   points_grad,
                   idx_no_grad,
                   points_no_grad,
                   num_points):

  points_all = torch.zeros(points_grad.shape[0],
                           num_points,
                           points_grad.shape[-1],
                           device=points_grad.device,
                           dtype=points_grad.dtype)

  idx_grad = rearrange(idx_grad, "n -> 1 n 1")
  idx_grad_out = idx_grad.expand(points_grad.shape[0], -1, points_grad.shape[-1])
  points_all.scatter_(dim=1, index=idx_grad_out, src=points_grad)

  idx_no_grad = rearrange(idx_no_grad, "n -> 1 n 1")
  idx_no_grad_out = idx_no_grad.expand(points_no_grad.shape[0], -1, points_no_grad.shape[-1])
  points_all.scatter_(dim=1, index=idx_no_grad_out, src=points_no_grad)
  return points_all


def gather_points(points,
                  idx_grad):
  """

  :param points: (b, n, c) or (b, n, s, c)
  :param idx_grad:
  :return:
  """
  if points.dim() == 4:
    idx_grad = rearrange(idx_grad, "n -> 1 n 1 1")
    idx_grad = idx_grad.expand(
      points.shape[0], -1, points.shape[-2], points.shape[-1])
    sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
  elif points.dim() == 3:
    idx_grad = rearrange(idx_grad, "n -> 1 n 1")
    idx_grad = idx_grad.expand(
      points.shape[0], -1, points.shape[-1])
    sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
  else:
    assert 0
  return sampled_points


def batch_scatter_points(idx_grad,
                         points_grad,
                         idx_no_grad,
                         points_no_grad,
                         num_points):
  """

  :param idx_grad: (b, Ngrad)
  :param points_grad: (b, N) or (b, N, c)
  :param idx_no_grad:
  :param points_no_grad:
  :param num_points:
  :return:
  """
  if points_grad.dim() == 2:
    points_grad = points_grad.unsqueeze(-1)
    points_no_grad = points_no_grad.unsqueeze(-1)

  points_all = torch.zeros(points_grad.shape[0],
                           num_points,
                           points_grad.shape[-1],
                           device=points_grad.device,
                           dtype=points_grad.dtype)

  idx_grad = rearrange(idx_grad, "b n -> b n 1")
  idx_grad_out = idx_grad.expand(points_grad.shape[0], -1, points_grad.shape[-1])
  points_all.scatter_(dim=1, index=idx_grad_out, src=points_grad)

  idx_no_grad = rearrange(idx_no_grad, "b n -> b n 1")
  idx_no_grad_out = idx_no_grad.expand(points_no_grad.shape[0], -1, points_no_grad.shape[-1])
  points_all.scatter_(dim=1, index=idx_no_grad_out, src=points_no_grad)

  points_all = points_all.squeeze(-1)
  return points_all

def batch_gather_points(points,
                        idx_grad):
  """

  :param points: (b, n, c) or (b, n, s, c)
  :param idx_grad: (b, Ngrad)
  :return:
  """
  if points.dim() == 4:
    idx_grad = rearrange(idx_grad, "b n -> b n 1 1")
    idx_grad = idx_grad.expand(points.shape[0], -1, points.shape[-2], points.shape[-1])
    sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
  elif points.dim() == 3:
    idx_grad = rearrange(idx_grad, "b n -> b n 1")
    idx_grad = idx_grad.expand(points.shape[0], -1, points.shape[-1])
    sampled_points = torch.gather(points, dim=1, index=idx_grad, sparse_grad=True)
  else:
    assert 0
  return sampled_points


def make_noise(batch,
               latent_dim,
               n_noise,
               device):
  if n_noise == 1:
    return torch.randn(batch, latent_dim, device=device)

  noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

  return noises


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
  """
  Normalize vector lengths.

  :param vectors:
  :return:
  """

  out = vectors / (torch.norm(vectors, dim=-1, keepdim=True))
  return out


def get_initial_rays_trig(bs,
                          num_steps,
                          fov,
                          resolution,
                          ray_start,
                          ray_end,
                          device, ):
  """
  Returns sample points, z_vals, and ray directions in camera space.

  :param bs:
  :param num_steps: number of samples along a ray
  :param fov:
  :param resolution:
  :param ray_start:
  :param ray_end:
  :param device:
  :return:
  points: (b, HxW, n_samples, 3)
  z_vals: (b, HxW, n_samples, 1)
  rays_d_cam: (b, HxW, 3)

  """

  W, H = resolution
  # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
  # Y is flipped to follow image memory layouts.
  x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                        torch.linspace(1, -1, H, device=device))
  x = x.T.flatten() # (HxW, ) [[-1, ..., 1], ...]
  y = y.T.flatten() # (HxW, ) [[1, ..., -1]^T, ...]
  z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360) / 2) # (HxW, )

  rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1)) # (HxW, 3)

  z_vals = torch.linspace(ray_start,
                          ray_end,
                          num_steps,
                          device=device) \
    .reshape(1, num_steps, 1) \
    .repeat(W * H, 1, 1) # (HxW, n, 1)
  points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals # (HxW, n_samples, 3)

  points = torch.stack(bs * [points]) # (b, HxW, n_samples, 3)
  z_vals = torch.stack(bs * [z_vals]) # (b, HxW, n_samples, 1)
  rays_d_cam = torch.stack(bs * [rays_d_cam]).to(device) # (b, HxW, 3)

  return points, z_vals, rays_d_cam



def perturb_points(points,
                   z_vals,
                   ray_directions,
                   device):
  """
  Perturb z_vals and then points

  :param points: (n, num_rays, n_samples, 3)
  :param z_vals: (n, num_rays, n_samples, 1)
  :param ray_directions: (n, num_rays, 3)
  :param device:
  :return:
  points: (n, num_rays, n_samples, 3)
  z_vals: (n, num_rays, n_samples, 1)
  """
  distance_between_points = z_vals[:, :, 1:2, :] - z_vals[:, :, 0:1, :] # (n, num_rays, 1, 1)
  offset = (torch.rand(z_vals.shape, device=device) - 0.5) \
           * distance_between_points # [-0.5, 0.5] * d, (n, num_rays, n_samples, 1)
  z_vals = z_vals + offset

  points = points + \
           offset * ray_directions.unsqueeze(2) # (n, num_rays, n_samples, 3)
  return points, z_vals


def truncated_normal_(tensor, mean=0, std=1):
  size = tensor.shape
  tmp = tensor.new_empty(size + (4,)).normal_()
  valid = (tmp < 2) & (tmp > -2)
  ind = valid.max(-1, keepdim=True)[1]
  tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
  tensor.data.mul_(std).add_(mean)
  return tensor


def sample_camera_positions(device,
                            bs=1,
                            r=1,
                            horizontal_stddev=1,
                            vertical_stddev=1,
                            horizontal_mean=math.pi * 0.5,
                            vertical_mean=math.pi * 0.5,
                            mode='normal'):
  """
  Samples bs random locations along a sphere of radius r. Uses the specified distribution.

  :param device:
  :param bs:
  :param r:
  :param horizontal_stddev: yaw std
  :param vertical_stddev: pitch std
  :param horizontal_mean:
  :param vertical_mean:
  :param mode:
  :return:
  output_points: (bs, 3), camera positions
  phi: (bs, 1), pitch in radians [0, pi]
  theta: (bs, 1), yaw in radians [-pi, pi]
  """

  if mode == 'uniform':
    theta = (torch.rand((bs, 1), device=device) - 0.5) \
            * 2 * horizontal_stddev \
            + horizontal_mean
    phi = (torch.rand((bs, 1), device=device) - 0.5) \
          * 2 * vertical_stddev \
          + vertical_mean

  elif mode == 'normal' or mode == 'gaussian':
    theta = torch.randn((bs, 1), device=device) \
            * horizontal_stddev \
            + horizontal_mean
    phi = torch.randn((bs, 1), device=device) \
          * vertical_stddev \
          + vertical_mean

  elif mode == 'hybrid':
    if random.random() < 0.5:
      theta = (torch.rand((bs, 1), device=device) - 0.5) \
              * 2 * horizontal_stddev * 2 \
              + horizontal_mean
      phi = (torch.rand((bs, 1), device=device) - 0.5) \
            * 2 * vertical_stddev * 2 \
            + vertical_mean
    else:
      theta = torch.randn((bs, 1), device=device) * horizontal_stddev + horizontal_mean
      phi = torch.randn((bs, 1), device=device) * vertical_stddev + vertical_mean

  elif mode == 'truncated_gaussian':
    theta = truncated_normal_(torch.zeros((bs, 1), device=device)) \
            * horizontal_stddev \
            + horizontal_mean
    phi = truncated_normal_(torch.zeros((bs, 1), device=device)) \
          * vertical_stddev \
          + vertical_mean

  elif mode == 'spherical_uniform':
    theta = (torch.rand((bs, 1), device=device) - .5) \
            * 2 * horizontal_stddev \
            + horizontal_mean
    v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
    v = ((torch.rand((bs, 1), device=device) - .5) * 2 * v_stddev + v_mean)
    v = torch.clamp(v, 1e-5, 1 - 1e-5)
    phi = torch.arccos(1 - 2 * v)

  elif mode == 'mean':
    # Just use the mean.
    theta = torch.ones((bs, 1), device=device, dtype=torch.float) * horizontal_mean
    phi = torch.ones((bs, 1), device=device, dtype=torch.float) * vertical_mean
  else:
    assert 0

  phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

  output_points = torch.zeros((bs, 3), device=device) # (bs, 3)
  output_points[:, 0:1] = r * torch.sin(phi) * torch.cos(theta) # x
  output_points[:, 2:3] = r * torch.sin(phi) * torch.sin(theta) # z
  output_points[:, 1:2] = r * torch.cos(phi) # y

  return output_points, phi, theta


def create_cam2world_matrix(forward_vector,
                            origin,
                            device=None,
                            up_vector=None):
  """
  Takes in the direction the camera is pointing
  and the camera origin and returns a cam2world matrix.

  :param forward_vector: (bs, 3), looking at direction
  :param origin: (bs, 3)
  :param device:
  :return:
  cam2world: (bs, 4, 4)
  """
  """"""

  forward_vector = normalize_vecs(forward_vector)
  if up_vector is None:
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

  left_vector = normalize_vecs(
    torch.cross(up_vector,
                forward_vector,
                dim=-1))

  up_vector = normalize_vecs(
    torch.cross(forward_vector,
                left_vector,
                dim=-1))

  rotation_matrix = torch.eye(4, device=device) \
    .unsqueeze(0) \
    .repeat(forward_vector.shape[0], 1, 1)
  rotation_matrix[:, :3, :3] = torch.stack(
    (-left_vector, up_vector, -forward_vector), axis=-1)

  translation_matrix = torch.eye(4, device=device) \
    .unsqueeze(0) \
    .repeat(forward_vector.shape[0], 1, 1)
  translation_matrix[:, :3, 3] = origin

  cam2world = translation_matrix @ rotation_matrix

  return cam2world


def transform_sampled_points(points,
                             z_vals,
                             ray_directions,
                             device,
                             h_stddev=1,
                             v_stddev=1,
                             h_mean=math.pi * 0.5,
                             v_mean=math.pi * 0.5,
                             mode='normal',
                             camera_pos=None,
                             camera_lookup=None,
                             up_vector=None,
                             ):
  """
  Perturb z_vals and points;
  Samples a camera position and maps points in camera space to world space.

  :param points: (bs, num_rays, n_samples, 3)
  :param z_vals: (bs, num_rays, n_samples, 1)
  :param ray_directions: (bs, num_rays, 3)
  :param device:
  :param h_stddev:
  :param v_stddev:
  :param h_mean:
  :param v_mean:
  :param mode: mode for sample_camera_positions
  :return:
  - transformed_points: (bs, num_rays, n_samples, 3)
  - z_vals: (bs, num_rays, n_samples, 1)
  - transformed_ray_directions: (bs, num_rays, 3)
  - transformed_ray_origins: (bs, num_rays, 3)
  - pitch: (bs, 1)
  - yaw: (bs, 1)
  """

  bs, num_rays, num_steps, channels = points.shape

  points, z_vals = perturb_points(points,
                                  z_vals,
                                  ray_directions,
                                  device)

  if camera_pos is None or camera_lookup is None:
    # (b, 3) (b, 1) (b, 1)
    camera_origin, pitch, yaw = sample_camera_positions(
      bs=bs,
      r=1,
      horizontal_stddev=h_stddev,
      vertical_stddev=v_stddev,
      horizontal_mean=h_mean,
      vertical_mean=v_mean,
      device=device,
      mode=mode)
    forward_vector = normalize_vecs(-camera_origin) # (b, 3)
  else:
    camera_origin = camera_pos
    pitch = yaw = torch.zeros(bs, 1, device=device)
    forward_vector = normalize_vecs(camera_lookup)  # (b, 3)


  cam2world_matrix = create_cam2world_matrix(forward_vector,
                                             camera_origin,
                                             device=device,
                                             up_vector=up_vector)

  points_homogeneous = torch.ones(
    (points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1),
    device=device)
  points_homogeneous[:, :, :, :3] = points

  # (bs, 4, 4) @ (bs, 4, num_rays x n_samples) -> (bs, 4, num_rays x n_samples) -> (bs, num_rays, n_samples, 4)
  transformed_points = torch.bmm(
    cam2world_matrix,
    points_homogeneous.reshape(bs, -1, 4).permute(0, 2, 1)) \
    .permute(0, 2, 1) \
    .reshape(bs, num_rays, num_steps, 4)
  transformed_points = transformed_points[..., :3] # (bs, num_rays, n_samples, 3)

  # (bs, 3, 3) @ (bs, 3, num_rays) -> (bs, 3, num_rays) -> (bs, num_rays, 3)
  transformed_ray_directions = torch.bmm(
    cam2world_matrix[..., :3, :3],
    ray_directions.reshape(bs, -1, 3).permute(0, 2, 1)) \
    .permute(0, 2, 1) \
    .reshape(bs, num_rays, 3)

  homogeneous_origins = torch.zeros((bs, 4, num_rays), device=device)
  homogeneous_origins[:, 3, :] = 1
  # (bs, 4, 4) @ (bs, 4, num_rays) -> (bs, 4, num_rays) -> (bs, num_rays, 4)
  transformed_ray_origins = torch.bmm(
    cam2world_matrix,
    homogeneous_origins) \
    .permute(0, 2, 1) \
    .reshape(bs, num_rays, 4)
  transformed_ray_origins = transformed_ray_origins[..., :3] # (bs, num_rays, 3)

  return transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


def get_world_points_and_direction(batch_size,
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
                                   device='cpu',
                                   camera_pos=None,
                                   camera_lookup=None,
                                   up_vector=None,
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
  points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
    bs=batch_size,
    num_steps=num_steps,
    resolution=(img_size, img_size),
    device=device,
    fov=fov,
    ray_start=ray_start,
    ray_end=ray_end)

  transformed_points, \
  z_vals, \
  transformed_ray_directions, \
  transformed_ray_origins, \
  pitch, yaw = transform_sampled_points(points_cam,
                                        z_vals,
                                        rays_d_cam,
                                        h_stddev=h_stddev,
                                        v_stddev=v_stddev,
                                        h_mean=h_mean,
                                        v_mean=v_mean,
                                        device=device,
                                        mode=sample_dist,
                                        camera_pos=camera_pos,
                                        camera_lookup=camera_lookup,
                                        up_vector=up_vector)

  transformed_ray_directions_expanded = repeat(
    transformed_ray_directions, "b hw xyz -> b (hw s) xyz", s=num_steps)
  if lock_view_dependence:
    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
    transformed_ray_directions_expanded[..., -1] = -1

  transformed_points = rearrange(transformed_points, "b hw s xyz -> b (hw s) xyz")

  ret = (transformed_points, transformed_ray_directions_expanded,
         transformed_ray_origins, transformed_ray_directions, z_vals,
         pitch, yaw)
  return ret