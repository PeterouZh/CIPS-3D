import itertools
import copy
import sys
import os
import math
import unittest
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F

# from exp.comm.op import FusedLeakyReLU, fused_leaky_relu
from tl2.proj.pytorch import init_func


class Modulated_FC_Conv(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=1,
        style_dim=None,
        use_style_fc=False,
        demodulate=True,
        scale=None,
        use_group_conv=False,
        eps=1e-8,
        **kwargs):
    """

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param style_dim: =in_channel
    :param use_style_fc:
    :param demodulate:
    """
    super().__init__()

    self.eps = eps
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.kernel_size = kernel_size
    self.style_dim = style_dim
    self.use_style_fc = use_style_fc
    self.demodulate = demodulate
    self.use_group_conv = use_group_conv

    if scale is not None:
      self.scale = scale
    else:
      fan_in = in_channel * kernel_size ** 2
      self.scale = 1 / math.sqrt(fan_in)
    self.padding = kernel_size // 2

    if use_group_conv:
      self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
    else:
      assert kernel_size == 1
      self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))

    if use_style_fc:
      self.modulation = EqualLinear(style_dim, in_channel, bias_init=1, lr_mul=1., scale=scale)
    else:
      self.style_dim = in_channel

    self.repr = f"in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, " \
                f"style_dim={style_dim}, use_style_fc={use_style_fc}, demodulate={demodulate}, " \
                f"scale={self.scale:.3f}, use_group_conv={use_group_conv}"
    pass

  def forward_bmm(self,
                  x,
                  style,
                  weight):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    assert x.shape[0] == style.shape[0]
    if x.dim() == 2:
      input = rearrange(x, "b c -> b 1 c")
    elif x.dim() == 3:
      input = x
    else:
      assert 0

    batch, N, in_channel = input.shape

    if self.use_style_fc:
      style = self.modulation(style).view(-1, in_channel, 1)
      style = style + 1.
    else:
      style = rearrange(style, 'b c -> b c 1')
      # style = style + 1.

    # (1, in, out) * (b in 1) -> (b, in, out)
    weight = self.scale * weight * style
    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([1, ]) + self.eps)  # (b, out)
      weight = weight * demod.view(batch, 1, self.out_channel)  # (b, in, out) * (b, 1, out) -> (b, in, out)
    # (b, n, in) * (b, in, out) -> (b, n, out)
    out = torch.bmm(input, weight)

    if x.dim() == 2:
      out = rearrange(out, "b 1 c -> b c")
    elif x.dim() == 3:
      # out = rearrange(out, "b n c -> b n c")
      pass
    return out

  def forward_group_conv(self,
                         x,
                         style):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    assert x.shape[0] == style.shape[0]
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
      style = self.modulation(style).view(-1, 1, in_channel, 1, 1)
      style = style + 1.
    else:
      style = rearrange(style, 'b c -> b 1 c 1 1')
      # style = style + 1.
    # (1, out, in, ks, ks) * (b, 1, in, 1, 1) -> (b, out, in, ks, ks)
    weight = self.scale * self.weight * style
    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps) # (b, out)
      weight = weight * demod.view(batch, self.out_channel, 1, 1, 1) # (b, out, in, ks, ks) * (b, out, 1, 1, 1)
    # (b*out, in, ks, ks)
    weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
    # (1, b*in, h, w)
    input = input.reshape(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channel, height, width)

    if x.dim() == 2:
      out = rearrange(out, "b c 1 1 -> b c")
    elif x.dim() == 3:
      out = rearrange(out, "b c n 1 -> b n c")

    return out

  def forward(self,
              x,
              style,
              force_bmm=False):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    if self.use_group_conv:
      if force_bmm:
        weight = rearrange(self.weight, "1 out in 1 1 -> 1 in out")
        out = self.forward_bmm(x=x, style=style, weight=weight)
      else:
        out = self.forward_group_conv(x=x, style=style)
    else:
      out = self.forward_bmm(x=x, style=style, weight=self.weight)
    return out

  def _load_from_state_dict(self,
                            state_dict,
                            prefix,
                            local_metadata,
                            strict,
                            missing_keys,
                            unexpected_keys,
                            error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
      hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
      key = prefix + name
      if key in state_dict:
        input_param = state_dict[key]
        # This is used to avoid copying uninitialized parameters into
        # non-lazy modules, since they dont have the hook to do the checks
        # in such case, it will error when accessing the .shape attribute.
        is_param_lazy = torch.nn.parameter.is_lazy(param)
        # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
        if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
          input_param = input_param[0]

        # if not is_param_lazy and input_param.shape != param.shape:
        #   # local shape should match the one in checkpoint
        #   error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
        #                     'the shape in current model is {}.'
        #                     .format(key, input_param.shape, param.shape))
        #   continue
        try:
          with torch.no_grad():
            param.copy_(input_param)
        except Exception as ex:
          print('While copying the parameter named "{}", '
                'whose dimensions in the model are {} and '
                'whose dimensions in the checkpoint are {}, '
                'an exception occurred : {}.\n'
                'Try reshape loading...'
                .format(key, param.size(), input_param.size(), ex.args))
          try:
            with torch.no_grad():
              input_param = rearrange(input_param, "1 out in 1 1 -> 1 in out")
              param.copy_(input_param)
          except Exception as ex:
            error_msgs.append('While copying the parameter named "{}", '
                              'whose dimensions in the model are {} and '
                              'whose dimensions in the checkpoint are {}, '
                              'an exception occurred : {}.'
                              .format(key, param.size(), input_param.size(), ex.args))
      elif strict:
        missing_keys.append(key)

    if strict:
      for key in state_dict.keys():
        if key.startswith(prefix):
          input_name = key[len(prefix):]
          input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
          if input_name not in self._modules and input_name not in local_state:
            unexpected_keys.append(key)
    pass

  def reset_parameters(self) -> None:
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if hasattr(self, 'modulation'):
      nn.init.kaiming_uniform_(self.modulation.weight, a=math.sqrt(5))
      if self.modulation.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.modulation.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.modulation.bias, -bound, bound)


class EqualLinear(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               bias=True,
               bias_init=0,
               lr_mul=1.,
               scale=None,
               norm_weight=False,
               **kwargs
               ):
    """

    :param in_dim:
    :param out_dim:
    :param bias:
    :param bias_init:
    :param lr_mul: 0.01
    """
    super().__init__()

    self.lr_mul = lr_mul
    self.norm_weight = norm_weight

    self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
    if bias:
      self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
    else:
      self.bias = None

    if scale is not None:
      self.scale = scale
    else:
      self.scale = (1 / math.sqrt(in_dim)) * lr_mul

    self.repr = f"in_dim={in_dim}, out_dim={out_dim}, bias={bias}, bias_init={bias_init}, " \
                f"lr_mul={lr_mul}, scale={self.scale:.6f}, norm_weight={norm_weight}"
    pass

  def forward(self,
              input):
    """

    :param input: (b c), (b, n, c)
    :return:
    """
    if self.norm_weight:
      demod = torch.rsqrt(self.weight.pow(2).sum([1, ], keepdim=True) + 1e-8)
      weight = self.weight * demod
    else:
      weight = self.weight
    out = F.linear(input, weight * self.scale, bias=self.bias * self.lr_mul)
    return out

  def __repr__(self):
    return f"{self.__class__.__name__}({self.repr})"


class EqualConv2d(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               in_channel,
               out_channel,
               kernel_size,
               stride=1,
               padding=0,
               bias=True):
      super().__init__()

      self.stride = stride
      self.padding = padding

      self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
      self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
      if bias:
          self.bias = nn.Parameter(torch.zeros(out_channel))
      else:
          self.bias = None

      self.repr = f"in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, " \
                  f"stride={stride}, padding={padding}, bias={bias}, scale={self.scale}"
      pass

  def forward(self, input):
      out = F.conv2d(
          input,
          self.weight * self.scale,
          bias=self.bias,
          stride=self.stride,
          padding=self.padding)
      return out


class SinAct(nn.Module):
  def __init__(self, ):
    super(SinAct, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class SinStyleMod(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=1,
        style_dim=None,
        use_style_fc=False,
        demodulate=True,
        use_group_conv=False,
        eps=1e-8,
        **kwargs):
    """

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param style_dim: =in_channel
    :param use_style_fc:
    :param demodulate:
    """
    super().__init__()

    self.eps = eps
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.kernel_size = kernel_size
    self.style_dim = style_dim
    self.use_style_fc = use_style_fc
    self.demodulate = demodulate
    self.use_group_conv = use_group_conv

    self.padding = kernel_size // 2

    if use_group_conv:
      self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
      torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    else:
      assert kernel_size == 1
      self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))
      torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    if use_style_fc:
      # self.modulation = EqualLinear(style_dim, in_channel, bias_init=1, lr_mul=1., scale=scale)
      self.modulation = nn.Linear(style_dim, in_channel)
      self.modulation.apply(init_func.kaiming_leaky_init)
      # self.modulation.weight.data.div_(0.01)
    else:
      self.style_dim = in_channel

    self.sin = SinAct()
    self.norm = nn.LayerNorm(in_channel)

    self.repr = f"in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, " \
                f"style_dim={style_dim}, use_style_fc={use_style_fc}, demodulate={demodulate}, " \
                f"use_group_conv={use_group_conv}"
    pass

  def forward_bmm(self,
                  x,
                  style,
                  weight):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    assert x.shape[0] == style.shape[0]
    if x.dim() == 2:
      input = rearrange(x, "b c -> b 1 c")
    elif x.dim() == 3:
      input = x
    else:
      assert 0

    batch, N, in_channel = input.shape

    if self.use_style_fc:
      # style = self.sin(style)
      style = self.modulation(style)
      # style = self.norm(style)
      style = style.view(-1, in_channel, 1)
    else:
      # style = self.norm(style)
      style = rearrange(style, 'b c -> b c 1')
      # style = style + 1.

    # (1, in, out) * (b in 1) -> (b, in, out)
    weight = weight * (style + 1)

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([1, ]) + self.eps)  # (b, out)
      weight = weight * demod.view(batch, 1, self.out_channel)  # (b, in, out) * (b, 1, out) -> (b, in, out)
    # (b, n, in) * (b, in, out) -> (b, n, out)
    out = torch.bmm(input, weight)

    if x.dim() == 2:
      out = rearrange(out, "b 1 c -> b c")
    elif x.dim() == 3:
      # out = rearrange(out, "b n c -> b n c")
      pass
    return out

  def forward_group_conv(self,
                         x,
                         style):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    assert x.shape[0] == style.shape[0]
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
      style = self.modulation(style).view(-1, 1, in_channel, 1, 1)
      style = style + 1.
    else:
      style = rearrange(style, 'b c -> b 1 c 1 1')
      # style = style + 1.
    # (1, out, in, ks, ks) * (b, 1, in, 1, 1) -> (b, out, in, ks, ks)
    weight = self.weight * style
    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps) # (b, out)
      weight = weight * demod.view(batch, self.out_channel, 1, 1, 1) # (b, out, in, ks, ks) * (b, out, 1, 1, 1)
    # (b*out, in, ks, ks)
    weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
    # (1, b*in, h, w)
    input = input.reshape(1, batch * in_channel, height, width)
    out = F.conv2d(input, weight, padding=self.padding, groups=batch)
    _, _, height, width = out.shape
    out = out.view(batch, self.out_channel, height, width)

    if x.dim() == 2:
      out = rearrange(out, "b c 1 1 -> b c")
    elif x.dim() == 3:
      out = rearrange(out, "b c n 1 -> b n c")

    return out

  def forward(self,
              x,
              style,
              force_bmm=False):
    """

    :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
    :param style: (b, in_c)
    :return:
    """
    if self.use_group_conv:
      if force_bmm:
        weight = rearrange(self.weight, "1 out in 1 1 -> 1 in out")
        out = self.forward_bmm(x=x, style=style, weight=weight)
      else:
        out = self.forward_group_conv(x=x, style=style)
    else:
      out = self.forward_bmm(x=x, style=style, weight=self.weight)
    return out

