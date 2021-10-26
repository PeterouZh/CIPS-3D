from einops import rearrange
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


class FiLMLayerEqualFC(nn.Module):
  def __init__(self,
               input_dim,
               hidden_dim):
    super().__init__()
    # self.layer = nn.Linear(input_dim, hidden_dim)
    self.layer = EqualLinear(input_dim, hidden_dim)
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


class Modulated_FC_Conv(nn.Module):
  def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=1,
        style_dim=None,
        use_style_fc=False,
        demodulate=True,
        activation=None):
    """

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param style_dim: =in_channel
    :param use_style_fc:
    :param demodulate:
    :param activation: FusedLeakyReLU
    """
    super().__init__()

    self.eps = 1e-8
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.kernel_size = kernel_size
    self.style_dim = style_dim
    self.use_style_fc = use_style_fc
    self.demodulate = demodulate
    self.activation = activation

    fan_in = in_channel * kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = kernel_size // 2

    self.weight = nn.Parameter(
      torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
    )
    self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    if use_style_fc:
      self.modulation = EqualLinear(style_dim, in_channel, bias_init=0)
    else:
      self.style_dim = in_channel

    if activation is not None:
      self.act_layer = nn.LeakyReLU(0.2)
    pass

  def __repr__(self):
    return (
      f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
      f'use_style_fc={self.use_style_fc}, activation={self.activation})'
    )

  def forward(self,
              x,
              style):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    if x.dim() == 2:
      input = rearrange(x, "b c -> b c 1 1")
    elif x.dim() == 3:
      input = rearrange(x, "b n c -> b c n 1")
    elif x.dim() == 4:
      input = x
    else:
      assert 0

    batch, in_channel, height, width = input.shape

    if self.use_style_fc:
      style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
      style = style + 1.
    else:
      style = rearrange(style, 'b c -> b 1 c 1 1')
      style = style + 1.

    weight = self.scale * self.weight * style
    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
      weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

    weight = weight.view(
      batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
    )

    input = input.reshape(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channel, height, width)

    out = out + self.bias

    if self.activation is not None:
      out = self.act_layer(out)

    if x.dim() == 2:
      out = rearrange(out, "b c 1 1 -> b c")
    elif x.dim() == 3:
      out = rearrange(out, "b c n 1 -> b n c")

    return out



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)



class EqualLinear(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               bias=True,
               bias_init=0,
               lr_mul=1.,
               activation=None
               ):
    """

    :param in_dim:
    :param out_dim:
    :param bias:
    :param bias_init:
    :param lr_mul: 0.01
    :param activation: None: Linear; fused_leaky_relu
    """
    super().__init__()

    self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

    else:
      self.bias = None

    self.activation = activation
    if self.activation is not None:
      self.act_layer = nn.LeakyReLU(0.2)

    self.scale = (1 / math.sqrt(in_dim)) * lr_mul
    self.lr_mul = lr_mul
    pass

  def forward(self,
              input):

    out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

    if self.activation:
      out = self.act_layer(out)

    return out

  def __repr__(self):
    return (
      f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}), activation={self.activation}'
    )




class MultiHead_MappingNetwork_EqualFC(nn.Module):
  def __init__(self,
               z_dim,
               hidden_dim,
               base_layers,
               head_layers,
               head_dim_dict,
               ):
    super().__init__()
    self.head_dim_dict = head_dim_dict

    self.norm_layer = PixelNorm()

    # base net
    base_net = []
    for i in range(base_layers):
      if i == 0:
        in_dim = z_dim
      else:
        in_dim = hidden_dim
      out_dim = hidden_dim

      if i != base_layers - 1:
        act = 'fused_leaky_relu'
      else:
        act = None

      hidden_layer = EqualLinear(in_dim=in_dim,
                                 out_dim=out_dim,
                                 bias=True,
                                 lr_mul=0.01,
                                 activation=act)

      base_net.append(hidden_layer)
    self.base_net = nn.Sequential(*base_net)

    # head net
    for name, head_dim in head_dim_dict.items():
      head_net = []
      in_dim = hidden_dim
      for i in range(head_layers):
        if i == head_layers - 1:
          out_dim = head_dim
          act = None
        else:
          out_dim = hidden_dim
          act = 'fused_leaky_relu'

        hidden_layer = EqualLinear(in_dim=in_dim,
                                   out_dim=out_dim,
                                   bias=True,
                                   lr_mul=0.01,
                                   activation=act)
        head_net.append(hidden_layer)
      self.add_module(name, nn.Sequential(*head_net))

    self.print_number_params()
    pass

  def forward(self, z):

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.norm_layer,
                                   inputs_args=(z, ),
                                   name_prefix='norm_layer')
    z = self.norm_layer(z)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.base_net,
                                   inputs_args=(z, ),
                                   name_prefix='base_net.')
    base_fea = self.base_net(z)

    out_dict = {}
    for name, head_dim in self.head_dim_dict.items():
      head_net = getattr(self, name)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(head_net,
                                     inputs_args=(base_fea, ),
                                     name_prefix=f"{name}.")
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
