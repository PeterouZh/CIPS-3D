from itertools import chain
import math
import logging
import collections
from collections import OrderedDict
import tqdm
import random
import time
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
from tl2.proj.pytorch.examples.nerf import cam_params
from tl2.proj.pytorch.examples.nerf import volume_rendering
from tl2.proj.pytorch.examples.networks import nerf_net
from tl2.proj.pytorch.examples.networks import multi_head_mapping
from tl2.proj.pytorch.examples.networks import cips_net

from exp.pigan import pigan_utils
from exp.dev.nerf_inr.models.generator_nerf_inr import INRNetwork
from exp.dev.nerf_inr.models.generator_nerf_inr import GeneratorNerfINR as GeneratorNerfINR_base
from exp.comm import comm_utils
from exp.comm.models import nerf_network
from exp.comm.models import inr_network
from exp.comm.models import film_layer
from exp.comm.models import mod_conv_fc
# from exp.cips3d.models import multi_head_mapping


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


@MODEL_REGISTRY.register(name_prefix=__name__)
# class Generator_Diffcam(GeneratorNerfINR_base):
class Generator_Diffcam(nn.Module):

  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               nerf_cfg,
               mapping_shape_cfg,
               mapping_app_cfg,
               inr_cfg,
               mapping_inr_cfg,
               inr_block_end_index=None,
               device='cuda',
               **kwargs):
    super(Generator_Diffcam, self).__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'nerf_cfg': nerf_cfg,
      'mapping_shape_cfg': mapping_shape_cfg,
      'mapping_app_cfg': mapping_app_cfg,
      'inr_cfg': inr_cfg,
      'mapping_inr_cfg': mapping_inr_cfg,
      'inr_block_end_index': inr_block_end_index,
    })

    self.device = device
    self.inr_block_end_index = inr_block_end_index

    self.module_name_list = []

    # nerf_net
    self.nerf_net = nerf_net.NeRFNetwork_CIPS(**nerf_cfg)
    self.module_name_list.append('nerf_net')

    # mapping shape
    self.mapping_shape = multi_head_mapping.MultiHeadMappingNetwork(**{
      **mapping_shape_cfg,
      'head_dim_dict': self.nerf_net.style_dim_dict_shape
    })
    self.module_name_list.append('mapping_shape')

    # mapping appearance
    self.mapping_app = multi_head_mapping.MultiHeadMappingNetwork(**{
      **mapping_app_cfg,
      'head_dim_dict': self.nerf_net.style_dim_dict_app
    })
    self.module_name_list.append('mapping_app')

    _in_dim = nerf_cfg.app_net_cfg.out_dim

    # inr_net
    self.inr_net = cips_net.CIPSNet(**{
      **inr_cfg,
      "input_dim": _in_dim,
      'add_out_layer': True,
    })
    self.module_name_list.append('inr_net')

    self.mapping_inr = multi_head_mapping.MultiHeadMappingNetwork(**{
      **mapping_inr_cfg,
      'head_dim_dict': self.inr_net.style_dim_dict
    })
    self.module_name_list.append('mapping_inr')


    self.aux_to_rbg = nn.Sequential(
      nn.Linear(_in_dim, 3),
      nn.Tanh()
    )
    self.aux_to_rbg.apply(nerf_network.frequency_init(25))
    self.module_name_list.append('aux_to_rbg')


    logger = logging.getLogger('tl')
    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['G'] = self
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)

    pass

  def forward(self,
              zs,
              rays_o,
              rays_d,
              nerf_kwargs={},
              psi=1,
              return_aux_img=False,
              grad_points=None,
              forward_points=None,  # disable gradients
              **kwargs):
    """
    Generates images from a noise vector, rendering parameters, and camera distribution.
    Uses the hierarchical sampling scheme described in NeRF.

    :param zs: {k: (b, z_dim), ...}
    :param rays_o: (b, h, w, 3) in world space
    :param rays_d: (b, h, w, 3) in world space

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

    b, h, w, c = rays_o.shape
    rays_o = rearrange(rays_o, "b h w c -> b (h w) c")
    rays_d = rearrange(rays_d, "b h w c -> b (h w) c")

    if grad_points is not None and grad_points < h * w:
      imgs, ret_maps = self.part_grad_forward(
        rays_o=rays_o,
        rays_d=rays_d,
        style_dict=style_dict,
        nerf_kwargs=nerf_kwargs,
        return_aux_img=return_aux_img,
        grad_points=grad_points)

    else:
      imgs, ret_maps = self.whole_grad_forward(
        rays_o=rays_o,
        rays_d=rays_d,
        style_dict=style_dict,
        nerf_kwargs=nerf_kwargs,
        return_aux_img=return_aux_img,
        forward_points=forward_points)

    imgs = rearrange(imgs, "b (h w) c -> b c h w", h=h, w=w)

    ret_imgs = {}
    for name, v_map in ret_maps.items():
      if v_map.dim() == 3:
        v_map = rearrange(v_map, "b (h w) c -> b c h w", h=h, w=w)
      elif v_map.dim() == 2:
        v_map = rearrange(v_map, "b (h w) -> b h w", h=h, w=w)
      ret_imgs[name] = v_map

    return imgs, ret_imgs

  def get_rays_axis_angle(self,
                          R,
                          t,
                          fx,
                          fy,
                          H: int,
                          W: int,
                          N_rays: int = -1):
    """

    :param R: (b, 3)
    :param t: (b, 3)
    :param fx:
    :param fy:
    :param H:
    :param W:
    :param N_rays:
    :return

    - rays_o: (b, H, W, 3)
    - rays_d: (b, H, W, 3)
    - select_inds: (b, H, W)
    """

    rays_o, rays_d, select_inds = cam_params.get_rays(
      rot=R,
      trans=t,
      focal_x=fx,
      focal_y=fy,
      H=H,
      W=W,
      N_rays=N_rays,
      flatten=False)

    return rays_o, rays_d, select_inds

  def get_batch_style_dict(self, b, style_dict):
    ret_style_dict = {}
    for name, style in style_dict.items():
      ret_style_dict[name] = style[[b]]
    return ret_style_dict

  def whole_grad_forward(self,
                         rays_o,
                         rays_d,
                         style_dict,
                         nerf_kwargs,
                         return_aux_img=True,
                         forward_points=None,
                         **kwargs):

    if forward_points is not None and forward_points < rays_o.shape[1]: # no gradients
      # stage forward
      with torch.no_grad():
        batch_size = rays_o.shape[0]
        num_points = rays_o.shape[1]

        near = nerf_kwargs['near']
        far = nerf_kwargs['far']
        N_samples = nerf_kwargs['N_samples']
        perturb = self.training
        z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o,
                                                            rays_d=rays_d,
                                                            near=near,
                                                            far=far,
                                                            N_samples=N_samples,
                                                            perturb=perturb)

        batch_image_ddict = collections.defaultdict(list)
        for b in range(batch_size):
          image_ddict = collections.defaultdict(list)

          head = 0
          while head < num_points:
            tail = head + forward_points
            cur_style_dict = self.get_batch_style_dict(b=b, style_dict=style_dict)

            cur_inr_img, cur_ret_maps = self.points_forward(
              rays_o=rays_o[[b], head:tail], # (b, hxw, 3)
              rays_d=rays_d[[b], head:tail], # (b, hxw, 3)
              points=points[[b], head:tail], # (b, hxw, Nsamples, 3)
              z_vals=z_vals[[b], head:tail], # (b, hxw, Nsamples)
              style_dict=cur_style_dict,
              nerf_kwargs=nerf_kwargs,
              return_aux_img=return_aux_img)

            image_ddict['inr_img'].append(cur_inr_img)
            for k, v in cur_ret_maps.items():
              image_ddict[k].append(v)
            head += forward_points
          for k, v in image_ddict.items():
            one_image = torch.cat(v, dim=1)
            batch_image_ddict[k].append(one_image)
        ret_maps = {}
        for k, v in batch_image_ddict.items():
          ret_maps[k] = torch.cat(v, dim=0)
        imgs = ret_maps.pop('inr_img')

    else:
      near = nerf_kwargs['near']
      far = nerf_kwargs['far']
      N_samples = nerf_kwargs['N_samples']
      perturb = self.training
      z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o,
                                                          rays_d=rays_d,
                                                          near=near,
                                                          far=far,
                                                          N_samples=N_samples,
                                                          perturb=perturb)

      # transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
      # transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
      #                                                 "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)

      imgs, ret_maps = self.points_forward(
        rays_o=rays_o,
        rays_d=rays_d,
        points=points,
        z_vals=z_vals,
        style_dict=style_dict,
        nerf_kwargs=nerf_kwargs,
        return_aux_img=return_aux_img)

    return imgs, ret_maps

  def part_grad_forward(self,
                        rays_o,
                        rays_d,
                        style_dict,
                        nerf_kwargs,
                        return_aux_img,
                        grad_points):

    near = nerf_kwargs['near']
    far = nerf_kwargs['far']
    N_samples = nerf_kwargs['N_samples']
    perturb = self.training
    # z_vals: (b, hxw, Nsamples), points: (b, hxw, Nsamples, 3)
    z_vals, points = volume_rendering.ray_sample_points(rays_o=rays_o, # (b, hxw, 3)
                                                        rays_d=rays_d, # (b, hxw, 3)
                                                        near=near,
                                                        far=far,
                                                        N_samples=N_samples,
                                                        perturb=perturb)

    # transformed_points = rearrange(transformed_points, "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)
    # transformed_ray_directions_expanded = rearrange(transformed_ray_directions_expanded,
    #                                                 "b (h w s) c -> b (h w) s c", h=img_size, s=num_steps)

    batch_size = rays_o.shape[0]
    num_points = rays_o.shape[1]
    device = self.device
    assert num_points > grad_points
    idx_grad, idx_no_grad = torch_utils.batch_random_split_indices(bs=batch_size,
                                                                   num_points=num_points,
                                                                   grad_points=grad_points,
                                                                   device=device)
    # rand_idx = torch.randperm(num_points, device=device)
    # idx_grad = rand_idx[:grad_points]
    # idx_no_grad = rand_idx[grad_points:]

    inr_img_grad, ret_maps_grad = self.points_forward(
      rays_o=rays_o,
      rays_d=rays_d,
      points=points,
      z_vals=z_vals,
      style_dict=style_dict,
      nerf_kwargs=nerf_kwargs,
      return_aux_img=return_aux_img,
      idx_grad=idx_grad)

    with torch.no_grad():
      inr_img_no_grad, ret_maps_no_grad = self.points_forward(
        rays_o=rays_o,
        rays_d=rays_d,
        points=points,
        z_vals=z_vals,
        style_dict=style_dict,
        nerf_kwargs=nerf_kwargs,
        return_aux_img=return_aux_img,
        idx_grad=idx_no_grad)

    imgs = comm_utils.batch_scatter_points(idx_grad=idx_grad,
                                           points_grad=inr_img_grad,
                                           idx_no_grad=idx_no_grad,
                                           points_no_grad=inr_img_no_grad,
                                           num_points=num_points)
    ret_maps = {}
    for k in ret_maps_grad.keys():
      comp_map = comm_utils.batch_scatter_points(idx_grad=idx_grad,
                                                 points_grad=ret_maps_grad[k],
                                                 idx_no_grad=idx_no_grad,
                                                 points_no_grad=ret_maps_no_grad[k],
                                                 num_points=num_points)
      ret_maps[k] = comp_map
    return imgs, ret_maps

  def points_forward(self,
                     rays_o,
                     rays_d,
                     points,
                     z_vals,
                     style_dict,
                     nerf_kwargs,
                     return_aux_img,
                     idx_grad=None,
                     **kwargs):
    """

    :param rays_o: (b, hxw, 3)
    :param rays_d: (b, hxw, 3)
    :param points: (b, hxw, Nsamples, 3)
    :param z_vals: (b, hxw, Nsamples)
    :param style_dict:
    :param nerf_kwargs:
    :param return_aux_img:
    :param idx_grad: (b, N_grad, )
    :param kwargs:
    :return:
    """

    device = points.device

    viewdirs = volume_rendering.get_viewdirs(rays_d=rays_d)
    # viewdirs = viewdirs[..., None, :].expand_as(points)
    N_samples = nerf_kwargs['N_samples']

    if idx_grad is not None:
      rays_o = comm_utils.batch_gather_points(points=rays_o, idx_grad=idx_grad)
      rays_d = comm_utils.batch_gather_points(points=rays_d, idx_grad=idx_grad)
      points = comm_utils.batch_gather_points(points=points, idx_grad=idx_grad)
      z_vals = comm_utils.batch_gather_points(points=z_vals, idx_grad=idx_grad)


    points = rearrange(points, "b Nrays Nsamples c -> b (Nrays Nsamples) c")
    coarse_viewdirs = repeat(viewdirs, "b Nrays c -> b (Nrays Nsamples) c", Nsamples=N_samples)

    # Model prediction on course points
    coarse_output = self.nerf_net(
      x=points,  # b (Nrays Nsamples) c
      ray_directions=coarse_viewdirs, # b (Nrays Nsamples) c
      style_dict=style_dict)
    coarse_output = rearrange(
      coarse_output, "b (Nrays Nsamples) rgb_sigma -> b Nrays Nsamples rgb_sigma", Nsamples=N_samples)

    # Re-sample fine points alont camera rays, as described in NeRF
    if nerf_kwargs['N_importance'] > 0:

      with torch.no_grad():
        raw_sigma = coarse_output[..., -1]
        perturb = self.training
        fine_z_vals, fine_points = volume_rendering.get_fine_points(
          z_vals=z_vals,
          rays_o=rays_o,
          rays_d=rays_d,
          raw_sigma=raw_sigma,
          N_importance=nerf_kwargs['N_importance'],
          perturb=perturb,
          raw_noise_std=nerf_kwargs['raw_noise_std'],
          eps=nerf_kwargs['eps'])

      # Model prediction on re-sampled find points
      fine_points = rearrange(fine_points, "b Nrays Nsamples c -> b (Nrays Nsamples) c")
      fine_viewdirs = repeat(viewdirs, "b Nrays c -> b (Nrays Nsamples) c", Nsamples=nerf_kwargs['N_importance'])

      fine_output = self.nerf_net(
        x=fine_points,  # b (Nrays Nsamples) c
        ray_directions=fine_viewdirs,  # b (Nrays Nsamples) c
        style_dict=style_dict)
      fine_output = rearrange(
        fine_output, "b (Nrays Nsamples) rgb_sigma -> b Nrays Nsamples rgb_sigma", Nsamples=nerf_kwargs['N_importance'])

      # Combine course and fine points
      DIM_SAMPLES = 2
      all_z_vals = torch.cat([fine_z_vals, z_vals], dim=DIM_SAMPLES) # (b, N_rays, N_samples)
      _, indices = torch.sort(all_z_vals, dim=DIM_SAMPLES) # (b, N_rays, N_samples)
      # gather z_vals
      all_z_vals = torch.gather(all_z_vals, DIM_SAMPLES, indices) # (b, N_rays, N_samples)

      # (b, N_rays, N_samples, rgb_sigma)
      all_outputs = torch.cat([fine_output, coarse_output], dim=DIM_SAMPLES)
      view_shape = [*indices.shape, *(len(all_outputs.shape) - len(indices.shape)) * [1]]
      all_outputs = torch.gather(all_outputs, DIM_SAMPLES, indices.view(view_shape).expand_as(all_outputs))

    else:
      all_outputs = coarse_output
      all_z_vals = z_vals

    # Create images with NeRF
    all_raw_rgb = all_outputs[..., :-1]
    all_raw_sigma = all_outputs[..., -1]

    pixels_fea, ret_maps = volume_rendering.ray_integration(raw_rgb=all_raw_rgb,
                                                            raw_sigma=all_raw_sigma,
                                                            z_vals=all_z_vals,
                                                            rays_d=rays_d,
                                                            raw_noise_std=nerf_kwargs['raw_noise_std'],
                                                            eps=nerf_kwargs['eps'])

    # inr_net
    inr_img = self.inr_net(pixels_fea, style_dict, block_end_index=self.inr_block_end_index)

    if return_aux_img:
      # aux rgb_branch
      aux_img = self.aux_to_rbg(pixels_fea)
      ret_maps['aux_img'] = aux_img

    return inr_img, ret_maps

  def z_sampler(self,
                shape,
                device,
                dist='gaussian'):
    if dist == 'gaussian':
      z = torch.randn(shape, device=device)
    elif dist == 'uniform':
      z = torch.rand(shape, device=device) * 2 - 1
    return z

  def get_zs(self,
             b,
             batch_split=1):
    z_shape = self.z_sampler(shape=(b, self.mapping_shape.z_dim), device=self.device)
    z_app = self.z_sampler(shape=(b, self.mapping_app.z_dim), device=self.device)
    z_inr = self.z_sampler(shape=(b, self.mapping_inr.z_dim), device=self.device)

    if batch_split > 1:
      zs_list = []
      z_shape_list = z_shape.split(b // batch_split)
      z_app_list = z_app.split(b // batch_split)
      z_inr_list = z_inr.split(b // batch_split)
      for z_shape_, z_app_, z_inr_ in zip(z_shape_list, z_app_list, z_inr_list):
        zs_ = {
          'z_shape': z_shape_,
          'z_app': z_app_,
          'z_inr': z_inr_,
        }
        zs_list.append(zs_)
      return zs_list
    else:
      zs = {
        'z_shape': z_shape,
        'z_app': z_app,
        'z_inr': z_inr,
      }
      return zs

  def mapping_network(self,
                      z_shape,
                      z_app,
                      z_inr):
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.mapping_shape,
                                   inputs_args=(z_shape,),
                                   submodels=['base_net'],
                                   name_prefix='mapping_shape.')
      VerboseModel.forward_verbose(self.mapping_app,
                                   inputs_args=(z_app,),
                                   submodels=['base_net'],
                                   name_prefix='mapping_app.')
      VerboseModel.forward_verbose(self.mapping_inr,
                                   inputs_args=(z_inr,),
                                   submodels=['base_net', ],
                                   input_padding=50,
                                   name_prefix='mapping_inr.')

    style_dict = {}
    style_dict.update(self.mapping_shape(z_shape))
    style_dict.update(self.mapping_app(z_app))
    style_dict.update(self.mapping_inr(z_inr))

    return style_dict

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

  def generate_avg_frequencies(self,
                               num_samples=10000,
                               device='cuda'):
    """Calculates average frequencies and phase shifts"""

    # z = torch.randn((num_samples, self.z_dim), device=device)
    zs = self.get_zs(num_samples)
    with torch.no_grad():
      style_dict = self.mapping_network(**zs)

    avg_styles = {}
    for name, style in style_dict.items():
      avg_styles[name] = style.mean(0, keepdim=True)

    # self.avg_styles = avg_styles
    return avg_styles

  def staged_forward(self, *args, **kwargs):
    raise NotImplementedError

  def set_device(self, device):
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
class GeneratorNerfINR_freeze_NeRF(Generator_Diffcam):

  def load_nerf_ema(self, G_ema):
    ret = self.nerf_net.load_state_dict(G_ema.nerf_net.state_dict())
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
      coarse_output = self.nerf_net(
        x=transformed_points,  # (b, n x s, 3)
        style_dict=style_dict,
        ray_directions=transformed_ray_directions_expanded,
      )
    coarse_output = rearrange(coarse_output, "b (n s) rgb_sigma -> b n s rgb_sigma", s=num_steps)

    # Re-sample fine points alont camera rays, as described in NeRF
    if hierarchical_sample:
      fine_points, fine_z_vals = self.get_fine_points_and_direction(
        coarse_output=coarse_output,
        z_vals=z_vals,
        dim_rgb=self.nerf_net.rgb_dim,
        clamp_mode=clamp_mode,
        nerf_noise=nerf_noise,
        num_steps=num_steps,
        transformed_ray_origins=transformed_ray_origins,
        transformed_ray_directions=transformed_ray_directions
      )

      # Model prediction on re-sampled find points
      with torch.no_grad():
        fine_output = self.nerf_net(
          x=fine_points,  # (b, n x s, 3)
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
      dim_rgb=self.nerf_net.rgb_dim,
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





