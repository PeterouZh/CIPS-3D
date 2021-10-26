import random
import numpy as np
import math
import logging
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import torch_utils


def _kaiming_leaky_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MultiHeadMappingNetwork(nn.Module):
  def __init__(self,
               z_dim,
               hidden_dim,
               base_layers,
               head_layers,
               head_dim_dict,
               ):
    super().__init__()
    self.head_dim_dict = head_dim_dict

    # base net
    base_net = []
    for i in range(base_layers):
      if i == 0:
        in_dim = z_dim
      else:
        in_dim = hidden_dim
      out_dim = hidden_dim

      base_net.append(nn.Linear(in_dim, out_dim))
      if i != base_layers - 1:
        base_net.append(nn.LeakyReLU(0.2, inplace=True))
    self.base_net = nn.Sequential(*base_net)
    self.base_net.apply(_kaiming_leaky_init)

    # head net
    # self.head_nets = nn.ModuleDict()
    for name, head_dim in head_dim_dict.items():
      head_net = []
      in_dim = hidden_dim
      for i in range(head_layers):
        if i == head_layers - 1:
          out_dim = head_dim
        else:
          out_dim = hidden_dim
        head_net.append(nn.Linear(in_dim, out_dim))
        if i != head_layers - 1:
          head_net.append(nn.LeakyReLU(0.2, inplace=True))
      self.add_module(name, nn.Sequential(*head_net))
      with torch.no_grad():
        getattr(self, name)[-1].weight *= 0.25

    self.print_number_params()
    pass

  def forward(self, z):
    # frequencies_offsets = self.network(z)
    # frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
    # phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

    base_fea = self.base_net(z)

    out_dict = {}
    for name, head_dim in self.head_dim_dict.items():
      head_net = getattr(self, name)
      out = head_net(base_fea)
      out_dict[name] = out

    return out_dict

  def print_number_params(self):
    models_dict = {
      'base_net': self.base_net
    }
    for name, _ in self.head_dim_dict.items():
      models_dict[name] = getattr(self, name)
    models_dict['mapping_network'] = self
    print()
    torch_utils.print_number_params(models_dict)
    pass


class FiLMLayer(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim):
    super().__init__()
    self.layer = nn.Linear(input_dim, hidden_dim)
    pass

  def forward(self,
              x,
              freq,
              phase_shift):
    """

    :param x: (b, num_points, d)
    :param freq: (b, d)
    :param phase_shift: (b, d)
    :return:
    """

    x = self.layer(x)
    freq = freq.unsqueeze(1).expand_as(x)
    phase_shift = phase_shift.unsqueeze(1).expand_as(x)
    out = torch.sin(freq * x + phase_shift)
    return out


class PosEmbedding(nn.Module):
  def __init__(self,
               max_logscale,
               N_freqs,
               logscale=True,
               multi_pi=False,):
    """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
    """
    super().__init__()

    self.N_freqs = N_freqs
    self.funcs = [torch.sin, torch.cos]

    if logscale:
      self.freqs = 2 ** torch.linspace(0, max_logscale, N_freqs)
    else:
      self.freqs = torch.linspace(1, 2 ** max_logscale, N_freqs)
    if multi_pi:
      self.freqs = self.freqs * math.pi
    pass

  def get_out_dim(self):
    outdim = 3 + 3 * 2 * self.N_freqs
    return outdim

  def forward(self, x):
    """
    Inputs:
        x: (B, 3)

    Outputs:
        out: (B, 6*N_freqs+3)
    """
    out = [x]
    for freq in self.freqs:
      for func in self.funcs:
        out += [func(freq * x)]

    return torch.cat(out, -1)


def sample_pdf(bins,
               weights,
               N_importance,
               det=False,
               eps=1e-5):
  """
  Sample @N_importance samples from @bins with distribution defined by @weights.
  Inputs:
      bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
      weights: (N_rays, N_samples_)
      N_importance: the number of samples to draw from the distribution
      det: deterministic or not
      eps: a small number to prevent division by zero
  Outputs:
      samples: (N_rays, N_importance), the sampled samples
  Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
  """
  N_rays, N_samples_ = weights.shape
  weights = weights + eps  # prevent division by zero (don't do inplace op!)
  pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
  cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
  cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
  # padded to 0~1 inclusive

  if det:
    u = torch.linspace(0, 1, N_importance, device=bins.device)
    u = u.expand(N_rays, N_importance)
  else:
    u = torch.rand(N_rays, N_importance, device=bins.device)
  u = u.contiguous()

  inds = torch.searchsorted(cdf, u)
  below = torch.clamp_min(inds - 1, 0)
  above = torch.clamp_max(inds, N_samples_)

  inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
  cdf_g = torch.gather(cdf, 1, inds_sampled)
  cdf_g = cdf_g.view(N_rays, N_importance, 2)
  bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

  denom = cdf_g[..., 1] - cdf_g[..., 0]
  denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
  # anyway, therefore any value for it is fine (set to 1 here)

  samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
  return samples


def fancy_integration(rgb_sigma,
                      z_vals,
                      device,
                      dim_rgb=3,
                      noise_std=0.5,
                      last_back=False,
                      white_back=False,
                      clamp_mode=None,
                      fill_mode=None):
  """
  Performs NeRF volumetric rendering.

  :param rgb_sigma: (b, h x w, num_samples, dim_rgb + dim_sigma)
  :param z_vals: (b, h x w, num_samples, 1)
  :param device:
  :param dim_rgb: rgb feature dim
  :param noise_std:
  :param last_back:
  :param white_back:
  :param clamp_mode:
  :param fill_mode:
  :return:
  - rgb_final: (b, h x w, dim_rgb)
  - depth_final: (b, h x w, 1)
  - weights: (b, h x w, num_samples, 1)
  """

  rgbs = rgb_sigma[..., :dim_rgb] # (b, h x w, num_samples, 3)
  sigmas = rgb_sigma[..., dim_rgb:] # (b, h x w, num_samples, 1)

  deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # (b, h x w, num_samples - 1, 1)
  delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1]) # (b, h x w, 1, 1)
  deltas = torch.cat([deltas, delta_inf], -2) # (b, h x w, num_samples, 1)

  noise = torch.randn(sigmas.shape, device=device) * noise_std # (b, h x w, num_samples, 1)

  if clamp_mode == 'softplus':
    alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
  elif clamp_mode == 'relu':
    alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise))) # (b, h x w, num_samples, 1)
  else:
    assert 0, "Need to choose clamp mode"

  alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2) # (b, h x w, num_samples + 1, 1)
  weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1] # (b, h x w, num_samples, 1)
  weights_sum = weights.sum(2)

  if last_back:
    weights[:, :, -1] += (1 - weights_sum)

  rgb_final = torch.sum(weights * rgbs, -2) # (b, h x w, num_samples, 3)
  depth_final = torch.sum(weights * z_vals, -2) # (b, h x w, num_samples, 1)

  if white_back:
    rgb_final = rgb_final + 1 - weights_sum

  if fill_mode == 'debug':
    rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
  elif fill_mode == 'weight':
    rgb_final = weights_sum.expand_as(rgb_final)

  return rgb_final, depth_final, weights


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
                            device=None):
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
  up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device) \
    .expand_as(forward_vector)

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
                             mode='normal'):
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

  camera_origin, pitch, yaw = sample_camera_positions(
    bs=bs,
    r=1,
    horizontal_stddev=h_stddev,
    vertical_stddev=v_stddev,
    horizontal_mean=h_mean,
    vertical_mean=v_mean,
    device=device,
    mode=mode)
  forward_vector = normalize_vecs(-camera_origin)

  cam2world_matrix = create_cam2world_matrix(forward_vector,
                                             camera_origin,
                                             device=device)

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


def z_sampler(shape, device, dist):
  if dist == 'gaussian':
    z = torch.randn(shape, device=device)
  elif dist == 'uniform':
    z = torch.rand(shape, device=device) * 2 - 1
  return z


@torch.no_grad()
def tensor_to_PIL(img_tensor):
  img = img_tensor.squeeze() * 0.5 + 0.5
  img_np = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
  return Image.fromarray(img_np)


def load_generator_ema(model_pkl, device=torch.device('cuda')):
  logger = logging.getLogger('tl')

  logger.info(f"Loading model: {model_pkl}")
  generator = torch.load(model_pkl, map_location=torch.device(device))
  ema_file = model_pkl.split('generator')[0] + 'ema.pth'
  logger.info(f"Load model ema: {ema_file}")
  ema = torch.load(ema_file, map_location=device)
  ema.copy_to(generator.parameters())
  generator.set_device(device)
  generator.eval()
  return generator







