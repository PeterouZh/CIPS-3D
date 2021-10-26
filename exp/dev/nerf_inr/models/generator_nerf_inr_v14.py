from itertools import chain
import logging
from einops import rearrange, repeat
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tl2.proj.fvcore import MODEL_REGISTRY, build_model
from tl2.proj.pytorch import torch_utils
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.proj.pytorch import init_func

from exp.pigan import pigan_utils
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base
from exp.comm import comm_utils
from exp.comm.models import nerf_network
from exp.comm.models import multi_head_mapping
from exp.comm.models import inr_network
from exp.comm.models import film_layer
from exp.comm.models.cond_layer_norm import CLN
from exp.comm.models import mod_conv_fc


class UniformBoxWarp(nn.Module):
  def __init__(self, sidelength):
    super().__init__()
    self.scale_factor = 2 / sidelength

  def forward(self, coordinates):
    return coordinates * self.scale_factor


class CLNLayer(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               ):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim

    self.repr = f"in_dim={in_dim}, out_dim={out_dim}, style_dim={style_dim}"

    self.linear1 = nn.Linear(in_dim, out_dim)
    self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
    self.style_dim = self.cln1.style_dim
    self.act1 = nn.LeakyReLU(0.2, inplace=True)

    pass

  def forward(self,
              x,
              style):

    x = self.linear1(x)
    x = self.cln1(x, style)
    x = self.act1(x)
    return x


@MODEL_REGISTRY.register(name_prefix=__name__)
class NeRFNetwork_Small(nn.Module):
  """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_dim=3,
               hidden_dim=256,
               rgb_dim=3,
               style_dim=512,
               hidden_layers=2,
               name_prefix='nerf',
               **kwargs):
    """

    :param z_dim:
    :param hidden_dim:
    :param rgb_dim:
    :param device:
    :param kwargs:
    """
    super(NeRFNetwork_Small, self).__init__()

    self.repr = f"in_dim={in_dim}, hidden_dim={hidden_dim}, rgb_dim={rgb_dim}, " \
                f"style_dim={style_dim}, hidden_layers={hidden_layers}"

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

      _layer = film_layer.FiLMLayer(in_dim=_in_dim,
                                    out_dim=_out_dim,
                                    style_dim=style_dim,
                                    use_style_fc=True)
      self.network.append(_layer)
      self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim

    # self.network.apply(frequency_init(25))
    # self.network[0].apply(first_layer_film_sine_init)

    self.final_layer = nn.Linear(hidden_dim, 1)
    # self.final_layer.apply(frequency_init(25))

    _in_dim= hidden_dim
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

    # Don't worry about this, it was added to ensure compatibility with another model.
    # Shouldn't affect performance.
    self.gridwarper = UniformBoxWarp(0.24)

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


class AddLayer(nn.Module):
  def __init__(self, ):
    super().__init__()
    pass

  def forward(self, x0, x1):
    return (x0 + x1)


class ModLinearBlock(nn.Module):
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

    self.linear1 = mod_conv_fc.Modulated_FC_Conv(in_channel=in_dim,
                                                 out_channel=out_dim,
                                                 style_dim=style_dim,
                                                 use_style_fc=True,
                                                 scale=None,
                                                 eps=1e-4)
    # torch.nn.init.kaiming_normal_(self.linear1.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    # torch.nn.init.kaiming_normal_(self.linear1.modulation.weight)
    self.style_dim_dict[f'{name_prefix}_0'] = self.linear1.style_dim
    self.act1 = nn.LeakyReLU(0.2, inplace=True)

    self.linear2 = mod_conv_fc.Modulated_FC_Conv(in_channel=out_dim,
                                                 out_channel=out_dim,
                                                 style_dim=style_dim,
                                                 use_style_fc=True,
                                                 scale=None,
                                                 eps=1e-4)
    # torch.nn.init.kaiming_normal_(self.linear2.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    # torch.nn.init.kaiming_normal_(self.linear2.modulation.weight)
    self.style_dim_dict[f'{name_prefix}_1'] = self.linear2.style_dim
    self.act2 = nn.LeakyReLU(0.2, inplace=True)

    self.add_layer = AddLayer()
    pass

  def forward(self,
              x,
              style_dict,
              skip=False):
    x_orig = x

    style = style_dict[f'{self.name_prefix}_0']
    x = self.linear1(x, style)
    x = self.act1(x)

    style = style_dict[f'{self.name_prefix}_1']
    x = self.linear2(x, style)
    out = self.act2(x)

    if skip and out.shape[-1] == x_orig.shape[-1]:
      out = self.add_layer(out, x_orig)
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
           f"out_dim={self.out_dim}, " \
           f"style_dim={self.style_dim})"
    return repr


class ModToRGB(nn.Module):
  def __init__(self,
               in_dim,
               dim_rgb=3):
    super().__init__()
    self.in_dim = in_dim
    self.dim_rgb = dim_rgb

    # self.linear = nn.Linear(in_dim, dim_rgb)
    self.linear = mod_conv_fc.EqualLinear(in_dim=in_dim,
                                          out_dim=dim_rgb)
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


class INRNetwork_Skip_CLN(nn.Module):
  def __repr__(self):
    return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               input_dim,
               style_dim,
               hidden_dim=256,
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

    self.repr = f"input_dim={input_dim}, style_dim={style_dim}, hidden_dim={hidden_dim}"

    self.device = device
    self.rgb_dim = rgb_dim
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
        _film_block = ModLinearBlock(in_dim=_in_dim,
                                     out_dim=_out_dim,
                                     style_dim=style_dim,
                                     name_prefix=f'{name_prefix}_w{name}')
        self.style_dim_dict.update(_film_block.style_dim_dict)
        network[name] = _film_block

      _to_rgb = ModToRGB(in_dim=_out_dim, dim_rgb=3)
      to_rbgs[name] = _to_rgb

    self.network = nn.ModuleDict(network)
    self.to_rgbs = nn.ModuleDict(to_rbgs)
    self.to_rgbs.apply(inr_network.frequency_init(100))

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
    logging.getLogger('tl').info(self)
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
      skip = int(name) >= 32
      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(block,
                                     inputs_args=(x, style_dict, skip),
                                     name_prefix=f'block.{name}.')
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



@MODEL_REGISTRY.register(name_prefix=__name__)
class GeneratorNerfINR(nn.Module):
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
    super(GeneratorNerfINR, self).__init__()

    self.z_dim = z_dim
    self.grad_lambda = grad_lambda
    self.device = device

    # nerf net
    self.siren = NeRFNetwork_Small(**nerf_cfg)
    self.mapping_network_nerf = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_nerf_cfg,
         'head_dim_dict': self.siren.style_dim_dict,
         })

    # inr_net
    self.inr_net = INRNetwork_Skip_CLN(**{**inr_cfg,
                                          "input_dim": self.siren.rgb_dim})
    self.mapping_network_inr = multi_head_mapping.MultiHeadMappingNetwork(
      **{**mapping_inr_cfg,
         'head_dim_dict': self.inr_net.style_dim_dict,
         })

    torch_utils.print_number_params(
      models_dict={
        'siren': self.siren,
        'mapping_network_nerf': self.mapping_network_nerf,
        'inr_net': self.inr_net,
        'mapping_network_inr': self.mapping_network_inr,
        'G': self,
      })

    self.epoch = 0
    self.step = 0
    pass

  def get_optimizer(self,
                    lr,
                    equal_lr,
                    betas=(0, 0.999),
                    weight_decay=0.):
    optimizer = torch.optim.Adam(
      [
        {'params': chain(self.siren.parameters(),
                         self.mapping_network_nerf.parameters(),
                         self.mapping_network_inr.parameters()),
         'initial_lr': lr,
         'lr': lr},
        {'params': self.inr_net.parameters(),
         'initial_lr': equal_lr,
         'lr': equal_lr}
      ],
      betas=betas,
      weight_decay=weight_decay)

    num_params = 0
    for group in optimizer.param_groups:
      num_params += len(group['params'])
    assert num_params == len(list(self.parameters()))
    return optimizer

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
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_network_nerf,
                                   inputs_args=(z, ),
                                   submodels=['base_net'],
                                   name_prefix="nerf_mapping.")
      VerboseModel.forward_verbose(self.mapping_network_inr,
                                   inputs_args=(z,),
                                   submodels=['base_net'],
                                   name_prefix="inr_mapping.")
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

    # inr_fea = self.grad_lambda * pixels_fea + (1 - self.grad_lambda) * pixels_fea.detach()
    inr_fea = pixels_fea
    inr_img = self.inr_net(inr_fea, style_dict, img_size=1024)

    if return_aux_img:
      # aux rgb_branch
      # aux_fea = (1 - self.grad_lambda) * pixels_fea + self.grad_lambda * pixels_fea.detach()
      # aux_img = self.aux_to_rbg(aux_fea, img_size=256, style_dict=style_dict)
      aux_img = torch.zeros_like(inr_img)
    else:
      aux_img = None

    return inr_img, aux_img

  def mapping_network(self, z):
    style_dict = {}
    style_dict.update(self.mapping_network_nerf(z))
    style_dict.update(self.mapping_network_inr(z))
    return style_dict

  def staged_forward(self, *args, **kwargs):
    raise NotImplementedError

  def set_device(self, device):
    self.device = device
    pass

  def generate_avg_frequencies(self,
                               num_samples=10000,
                               device='cuda'):
    z = torch.randn((num_samples, self.z_dim), device=device)
    with torch.no_grad():
      style_dict = self.mapping_network(z)

    avg_styles = {}
    for name, style in style_dict.items():
      avg_styles[name] = style.mean(0, keepdim=True)
    self.avg_styles = avg_styles
    return avg_styles

  def get_truncated_freq_phase(self,
                               raw_style_dict,
                               avg_style_dict,
                               raw_lambda):

    truncated_style_dict = {}
    for name, avg_style in avg_style_dict.items():
      raw_style = raw_style_dict[name]
      truncated_style = avg_style + raw_lambda * (raw_style - avg_style)
      truncated_style_dict[name] = truncated_style
    return truncated_style_dict

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




