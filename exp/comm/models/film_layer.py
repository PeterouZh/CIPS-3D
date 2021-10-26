from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tl2.proj.pytorch import init_func


def frequency_init(freq):
  def init(m):
    with torch.no_grad():
      if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

  return init


class LinearScale(nn.Module):
  def __init__(self,
               scale,
               bias):
    super(LinearScale, self).__init__()
    self.scale_v = scale
    self.bias_v = bias
    pass

  def forward(self, x):
    out = x * self.scale_v + self.bias_v
    return out

  def __repr__(self):
    repr = f"{self.__class__.__name__}(" \
           f"scale_v={self.scale_v}," \
           f"bias_v={self.bias_v})"
    return repr


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
    self.linear.apply(frequency_init(25))

    self.gain_scale = LinearScale(scale=15, bias=30)
    # Prepare gain and bias layers
    if use_style_fc:
      self.gain_fc = which_linear(style_dim, out_dim)
      self.bias_fc = which_linear(style_dim, out_dim)
      self.gain_fc.weight.data.mul_(0.25)
      self.bias_fc.weight.data.mul_(0.25)
      # self.gain_fc.apply(init_func.kaiming_leaky_init)
      # self.bias_fc.apply(init_func.kaiming_leaky_init)
      # with torch.no_grad():
      #   self.gain_fc.weight.data.mul_(0.25)
    else:
      self.style_dim = out_dim * 2


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
    out = torch.sin(gain * x + bias)
    return out

  def __repr__(self):
    s = f'{self.__class__.__name__}(' \
        f'in_dim={self.in_dim}, ' \
        f'out_dim={self.out_dim}, ' \
        f'style_dim={self.style_dim}, ' \
        f'use_style_fc={self.use_style_fc}, ' \
        f')'
    return s


class FiLMLayer_PreSin(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               style_dim,
               use_style_fc=True,
               which_linear=nn.Linear,
               **kwargs):
    super(FiLMLayer_PreSin, self).__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.style_dim = style_dim
    self.use_style_fc = use_style_fc

    self.linear = which_linear(in_dim, out_dim)
    nn.init.uniform_(self.linear.weight, -np.sqrt(9 / in_dim), np.sqrt(9 / in_dim))

    # Prepare gain and bias layers
    if use_style_fc:
      self.gain_fc = which_linear(style_dim, out_dim)
      self.bias_fc = which_linear(style_dim, out_dim)
      self.gain_fc.weight.data.mul_(0.25)
      self.gain_fc.bias.data.fill_(1)
      self.bias_fc.weight.data.mul_(0.25)
    else:
      self.style_dim = out_dim * 2

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
      bias = self.bias_fc(style)
    else:
      style = rearrange(style, "b (n c) -> b n c", n=2)
      gain, bias = style.unbind(dim=1)

    if x.dim() == 3:
      gain = rearrange(gain, "b c -> b 1 c")
      bias = rearrange(bias, "b c -> b 1 c")
    elif x.dim() == 2:
      pass
    else:
      assert 0

    x = self.linear(x)
    x = torch.sin(x)
    out = gain * x + bias
    return out

  def __repr__(self):
    s = f'{self.__class__.__name__}(' \
        f'in_dim={self.in_dim}, ' \
        f'out_dim={self.out_dim}, ' \
        f'style_dim={self.style_dim}, ' \
        f'use_style_fc={self.use_style_fc}, ' \
        f')'
    return s
