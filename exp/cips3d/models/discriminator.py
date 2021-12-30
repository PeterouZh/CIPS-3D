from einops import rearrange
import collections
import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.pytorch import torch_utils
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2 import tl2_utils

from exp.comm.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from .diffaug import DiffAugment


class EqualConv2d(nn.Module):
  def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
  ):
    super().__init__()

    self.weight = nn.Parameter(
      torch.randn(out_channel, in_channel, kernel_size, kernel_size)
    )
    self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

    self.stride = stride
    self.padding = padding

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channel))

    else:
      self.bias = None

  def forward(self, input):
    out = F.conv2d(
      input,
      self.weight * self.scale,
      bias=self.bias,
      stride=self.stride,
      padding=self.padding,
    )
    return out

  def __repr__(self):
    return (
      f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
      f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
    )


def make_kernel(k):
  k = torch.tensor(k, dtype=torch.float32)

  if k.ndim == 1:
    k = k[None, :] * k[:, None]

  k /= k.sum()

  return k

class Blur(nn.Module):
  def __init__(self, kernel, pad, upsample_factor=1):
    super().__init__()

    kernel = make_kernel(kernel)

    if upsample_factor > 1:
      kernel = kernel * (upsample_factor ** 2)

    self.register_buffer('kernel', kernel)

    self.pad = pad

  def forward(self, input):
    out = upfirdn2d(input, self.kernel, pad=self.pad)
    return out


class EqualConvTranspose2d(nn.Module):
  def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
  ):
    super().__init__()

    self.weight = nn.Parameter(
      torch.randn(in_channel, out_channel, kernel_size, kernel_size)
    )
    self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

    self.stride = stride
    self.padding = padding

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channel))

    else:
      self.bias = None

  def forward(self, input):
    out = F.conv_transpose2d(
      input,
      self.weight * self.scale,
      bias=self.bias,
      stride=self.stride,
      padding=self.padding,
    )

    return out

  def __repr__(self):
    return (
      f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
      f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
    )


class ScaledLeakyReLU(nn.Module):
  def __init__(self, negative_slope=0.2):
    super().__init__()

    self.negative_slope = negative_slope

  def forward(self, input):
    out = F.leaky_relu(input, negative_slope=self.negative_slope)
    return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
  def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        upsample=False,
        padding="zero",
  ):
    # layers = []
    layers = collections.OrderedDict()

    self.padding = 0
    stride = 1

    if downsample:
      factor = 2
      p = (len(blur_kernel) - factor) + (kernel_size - 1)
      pad0 = (p + 1) // 2
      pad1 = p // 2

      # layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
      layers['down_blur'] = Blur(blur_kernel, pad=(pad0, pad1))

      stride = 2

    if upsample:
      up_conv = EqualConvTranspose2d(
        in_channel,
        out_channel,
        kernel_size,
        padding=0,
        stride=2,
        bias=bias and not activate,
      )
      # layers.append(up_conv)
      layers['up_conv'] = up_conv

      factor = 2
      p = (len(blur_kernel) - factor) - (kernel_size - 1)
      pad0 = (p + 1) // 2 + factor - 1
      pad1 = p // 2 + 1

      # layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
      layers['up_blur'] = Blur(blur_kernel, pad=(pad0, pad1))

    else:
      if not downsample:
        if padding == "zero":
          self.padding = (kernel_size - 1) // 2

        elif padding == "reflect":
          padding = (kernel_size - 1) // 2

          if padding > 0:
            # layers.append(nn.ReflectionPad2d(padding))
            layers['pad'] = nn.ReflectionPad2d(padding)

          self.padding = 0

        elif padding != "valid":
          raise ValueError('Padding should be "zero", "reflect", or "valid"')

      equal_conv = EqualConv2d(
        in_channel,
        out_channel,
        kernel_size,
        padding=self.padding,
        stride=stride,
        bias=bias and not activate,
      )
      # layers.append(equal_conv)
      layers['equal_conv'] = equal_conv

    if activate:
      if bias:
        # layers.append(FusedLeakyReLU(out_channel))
        layers['flrelu'] = FusedLeakyReLU(out_channel)

      else:
        # layers.append(ScaledLeakyReLU(0.2))
        layers['slrelu'] = ScaledLeakyReLU(0.2)

    super().__init__(layers)
    pass

class ResBlock(nn.Module):
  def __init__(self,
               in_channel,
               out_channel,
               blur_kernel=[1, 3, 3, 1],
               kernel_size=3,
               downsample=True,
               first_downsample=False):
    super().__init__()

    if first_downsample:
      self.conv1 = ConvLayer(in_channel, in_channel, kernel_size, downsample=downsample)
      self.conv2 = ConvLayer(in_channel, out_channel, kernel_size)
    else:
      self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
      self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)

    self.skip = ConvLayer(
      in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
    )

  def forward(self, input):
    out = self.conv1(input)
    out = self.conv2(out)

    skip = self.skip(input)
    out = (out + skip) / math.sqrt(2)

    return out

class EqualLinear(nn.Module):
  def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
  ):
    super().__init__()

    self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

    else:
      self.bias = None

    self.activation = activation

    self.scale = (1 / math.sqrt(in_dim)) * lr_mul
    self.lr_mul = lr_mul

  def forward(self, input):
    if self.activation:
      out = F.linear(input, self.weight * self.scale)
      out = fused_leaky_relu(out, self.bias * self.lr_mul)

    else:
      out = F.linear(
        input, self.weight * self.scale, bias=self.bias * self.lr_mul
      )

    return out

  def __repr__(self):
    return (
      f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
    )


@MODEL_REGISTRY.register(name_prefix=__name__)
class Discriminator(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               size,
               channel_multiplier=2,
               blur_kernel=[1, 3, 3, 1],
               input_size=3,
               n_first_layers=0,
               **kwargs):
    super().__init__()

    self.repr = f"size={size}, channel_multiplier={channel_multiplier}, n_first_layers={n_first_layers}"

    self.size = size
    self.input_size = input_size

    self.epoch = 0
    self.step = 0

    channels = {
      4: 512,
      8: 512,
      16: 512,
      32: 512,
      64: 256 * channel_multiplier,
      128: 128 * channel_multiplier,
      256: 64 * channel_multiplier,
      512: 32 * channel_multiplier,
      1024: 16 * channel_multiplier,
    }

    convs = []
    _conv_layer = ConvLayer(input_size, channels[size], 1)
    convs.append(_conv_layer)

    _first_layers = [ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)]
    convs.extend(_first_layers)

    log_size = int(math.log(size, 2))

    in_channel = channels[size]

    for i in range(log_size, 2, -1):
      out_channel = channels[2 ** (i - 1)]

      convs.append(ResBlock(in_channel, out_channel, blur_kernel))

      in_channel = out_channel

    self.convs = nn.Sequential(*convs)

    self.stddev_group = 4
    self.stddev_feat = 1

    self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
    self.final_linear = nn.Sequential(
      EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
      EqualLinear(channels[4], 1),
    )

    torch_utils.print_number_params(models_dict={
      "convs": self.convs,
      "final_conv": self.final_conv,
      "final_linear": self.final_linear,
      "D": self
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              *args,
              **kwargs):
    assert input.shape[-1] == self.size

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.convs,
                                   inputs_args=(input, ),
                                   submodels=['0', '1', ],
                                   name_prefix="convs.")
    out = self.convs(input)

    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    # (4, 1, 1, 512//1, 4, 4)
    stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
    # (1, 1, 512//1, 4, 4)
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    # (1, 1, 1, 1)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    # (4, 1, 4, 4)
    stddev = stddev.repeat(group, 1, height, width)
    # (4, 513, 4, 4)
    out = torch.cat([out, stddev], 1)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_conv,
                                   inputs_args=(out,),
                                   name_prefix="final_conv.")
    out = self.final_conv(out)
    out = out.view(batch, -1)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_linear,
                                   inputs_args=(out,),
                                   name_prefix="final_linear.")
    out = self.final_linear(out)

    latent, position = None, None
    return out, latent, position


@MODEL_REGISTRY.register(name_prefix=__name__)
class Discriminator_MultiScale(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               diffaug,
               max_size,
               channel_multiplier=2,
               blur_kernel=[1, 3, 3, 1],
               input_size=3,
               first_downsample=False,
               channels=None,
               stddev_group=4,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'diffaug': diffaug,
      'max_size': max_size,
      'channel_multiplier': channel_multiplier,
      'first_downsample': first_downsample,
      'stddev_group': stddev_group,
    })

    self.epoch = 0
    self.step = 0

    self.diffaug = diffaug
    self.max_size = max_size
    self.input_size = input_size
    self.stddev_group = stddev_group

    self.module_name_list = []

    if channels is None:
      channels = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
      }

    self.conv_in = nn.ModuleDict()
    self.module_name_list.append('conv_in')
    for name, channel_ in channels.items():
      self.conv_in[f"{name}"] = ConvLayer(input_size, channel_, 1)

    # _first_layers = [ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)]
    # convs.extend(_first_layers)

    self.convs = nn.ModuleDict()
    self.module_name_list.append('convs')
    log_size = int(math.log(max_size, 2))
    in_channel = channels[max_size]
    for i in range(log_size, 2, -1):
      out_channel = channels[2 ** (i - 1)]
      self.convs[f"{2 ** i}"] = ResBlock(in_channel, out_channel, blur_kernel, first_downsample=first_downsample)
      in_channel = out_channel

    self.stddev_feat = 1

    if self.stddev_group > 1:
      self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
    else:
      self.final_conv = ConvLayer(in_channel, channels[4], 3)
    self.module_name_list.append('final_conv')

    self.final_linear = nn.Sequential(
      EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
      EqualLinear(channels[4], 1))
    self.module_name_list.append('final_linear')

    models_dict = {}
    for name in self.module_name_list:
      models_dict[name] = getattr(self, name)
    models_dict['D'] = self
    logger = logging.getLogger('tl')
    torch_utils.print_number_params(models_dict=models_dict, logger=logger)
    logger.info(self)
    pass

  def diff_aug_img(self, img):
    img = DiffAugment(img, policy='color,translation,cutout')
    return img

  def forward(self,
              input,
              alpha):
    # assert input.shape[-1] == self.size
    if self.diffaug:
      input = self.diff_aug_img(input)

    size = input.shape[-1]
    log_size = int(math.log(size, 2))

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.conv_in[f"{2 ** log_size}"],
                                   inputs_args=(input,),
                                   name_prefix=f"conv_in[{2 ** log_size}].")
    cur_size_out = self.conv_in[f"{2 ** log_size}"](input)
    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.convs[f"{2 ** log_size}"],
                                   inputs_args=(cur_size_out,),
                                   name_prefix=f"convs[{2 ** log_size}].")
    cur_size_out = self.convs[f"{2 ** log_size}"](cur_size_out)

    if alpha < 1:
      down_input = F.interpolate(input, scale_factor=0.5, mode='bilinear')
      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.conv_in[f"{2 ** (log_size - 1)}"],
                                     inputs_args=(down_input,),
                                     name_prefix=f"conv_in[{2 ** (log_size - 1)}].")
      down_size_out = self.conv_in[f"{2 ** (log_size - 1)}"](down_input)

      out = alpha * cur_size_out + (1 - alpha) * down_size_out
    else:
      out = cur_size_out

    for i in range(log_size - 1, 2, -1):
      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(self.convs[f"{2 ** i}"],
                                     inputs_args=(out, ),
                                     name_prefix=f"convs[{2 ** i}].")
      out = self.convs[f"{2 ** i}"](out)

    batch, channel, height, width = out.shape

    if self.stddev_group > 0:
      group = min(batch, self.stddev_group)
      # (4, 2, 1, 512//1, 4, 4)
      stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
      # (2, 1, 512//1, 4, 4)
      stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
      # (2, 1, 1, 1)
      stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
      # (8, 1, 4, 4)
      stddev = stddev.repeat(group, 1, height, width)
      # (8, 513, 4, 4)
      out = torch.cat([out, stddev], 1)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_conv,
                                   inputs_args=(out,),
                                   name_prefix="final_conv.")
    out = self.final_conv(out)
    out = out.view(batch, -1)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_linear,
                                   inputs_args=(out,),
                                   name_prefix="final_linear.")
    out = self.final_linear(out)

    latent, position = None, None
    return out, latent, position


@MODEL_REGISTRY.register(name_prefix=__name__)
class Discriminator_MultiScale_Aux(nn.Module):

  def __repr__(self):
    return tl2_utils.get_class_repr(self)

  def __init__(self,
               diffaug,
               max_size,
               channel_multiplier=2,
               first_downsample=False,
               stddev_group=0,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'diffaug': diffaug,
      'max_size': max_size,
      'channel_multiplier': channel_multiplier,
      'first_downsample': first_downsample,
      'stddev_group': stddev_group,
    })

    self.epoch = 0
    self.step = 0

    self.main_disc = Discriminator_MultiScale(diffaug=diffaug,
                                              max_size=max_size,
                                              channel_multiplier=channel_multiplier,
                                              first_downsample=first_downsample,
                                              stddev_group=stddev_group)

    channel_multiplier = 2
    channels = {
      4: 128 * channel_multiplier,
      8: 128 * channel_multiplier,
      16: 128 * channel_multiplier,
      32: 128 * channel_multiplier,
      64: 128 * channel_multiplier,
      128: 128 * channel_multiplier,
      256: 64 * channel_multiplier,
      512: 32 * channel_multiplier,
      1024: 16 * channel_multiplier,
    }
    self.aux_disc = Discriminator_MultiScale(diffaug=diffaug,
                                             max_size=max_size,
                                             channel_multiplier=channel_multiplier,
                                             first_downsample=True,
                                             channels=channels,
                                             stddev_group=stddev_group)
    logger = logging.getLogger('tl')
    torch_utils.print_number_params(models_dict={
      "main_disc": self.main_disc,
      "aux_disc": self.aux_disc,
      "D": self
    }, logger=logger)
    logger.info(self)
    pass

  def forward(self,
              input,
              use_aux_disc=False,
              **kwargs):
    alpha = 1.
    if use_aux_disc:
      b = input.shape[0] // 2
      main_input = input[:b]
      aux_input = input[b:]
      main_out, latent, position = self.main_disc(main_input, alpha)
      aux_out, _, _ = self.aux_disc(aux_input, alpha)
      out = torch.cat([main_out, aux_out], dim=0)
    else:
      out, latent, position = self.main_disc(input, alpha)

    return out, latent, position