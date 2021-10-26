import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch import torch_utils, init_func

from exp.comm.models import mod_conv_fc


class AddCoords(nn.Module):
  """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

  def __init__(self, with_r=False):
    super().__init__()
    self.with_r = with_r

  def forward(self, input_tensor):
    """
    Args:
        input_tensor: shape(batch, channel, x_dim, y_dim)
    """
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat([
      input_tensor,
      xx_channel.type_as(input_tensor),
      yy_channel.type_as(input_tensor)], dim=1)

    if self.with_r:
      rr = torch.sqrt(
        torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
      ret = torch.cat([ret, rr], dim=1)

    return ret


class CoordConv(nn.Module):
  """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

  def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
    super().__init__()
    self.addcoords = AddCoords(with_r=with_r)
    in_size = in_channels + 2
    if with_r:
      in_size += 1
    self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class ResidualCCBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3):
    super().__init__()
    p = kernel_size // 2
    self.network = nn.Sequential(
      CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
      nn.LeakyReLU(0.2, inplace=True),
      CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
      nn.LeakyReLU(0.2, inplace=True)
    )
    self.network.apply(kaiming_leaky_init)

    self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    y = (y + identity) / math.sqrt(2)
    return y


class AdapterBlock(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, input):
        return self.model(input)


@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    self.layers = nn.ModuleList(
      [
        ResidualCCBlock(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock(256, 400),  # 3 32x32 -> 16x16
        ResidualCCBlock(400, 400),  # 2 16x16 -> 8x8
        ResidualCCBlock(400, 400),  # 1 8x8 -> 4x4
        ResidualCCBlock(400, 400),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400)
      ])
    self.final_layer = nn.Conv2d(400, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      x = layer(x)

    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position


class ResidualCCBlock_FirstDown(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3):
    super().__init__()
    p = kernel_size // 2
    self.network = nn.Sequential(
      CoordConv(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p),
      nn.LeakyReLU(0.2, inplace=True),
      CoordConv(planes, planes, kernel_size=kernel_size, stride=1, padding=p),
      nn.LeakyReLU(0.2, inplace=True)
    )
    self.network.apply(kaiming_leaky_init)

    self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    y = (y + identity) / math.sqrt(2)
    return y


@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator_FirstDown(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    self.layers = nn.ModuleList(
      [
        ResidualCCBlock_FirstDown(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock_FirstDown(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock_FirstDown(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock_FirstDown(256, 400),  # 3 32x32 -> 16x16
        ResidualCCBlock_FirstDown(400, 400),  # 2 16x16 -> 8x8
        ResidualCCBlock_FirstDown(400, 400),  # 1 8x8 -> 4x4
        ResidualCCBlock_FirstDown(400, 400),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400)
      ])
    self.final_layer = nn.Conv2d(400, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.fromRGB[start],
                                   inputs_args=(input, ),
                                   submodels=['model', ],
                                   name_prefix=f"fromRGB[{start}].")
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'network.0', 'network.2'],
                                     name_prefix=f"layers[{start + i}].")
      x = layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x, ),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position


class SinAct(nn.Module):
  def __init__(self, ):
    super(SinAct, self).__init__()

  def forward(self, x):
    return torch.sin(x)


class ResidualCCBlock_FirstDown_SinAct(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3):
    super().__init__()
    p = kernel_size // 2
    self.network = nn.Sequential(
      CoordConv(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p),
      SinAct(),
      # nn.LeakyReLU(0.2, inplace=True),
      CoordConv(planes, planes, kernel_size=kernel_size, stride=1, padding=p),
      SinAct(),
      # nn.LeakyReLU(0.2, inplace=True)
    )
    self.network.apply(kaiming_leaky_init)

    self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    y = (y + identity) / math.sqrt(2)
    return y


@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator_FirstDown_SinAct(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    self.layers = nn.ModuleList(
      [
        ResidualCCBlock_FirstDown_SinAct(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock_FirstDown_SinAct(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock_FirstDown_SinAct(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock_FirstDown_SinAct(256, 400),  # 3 32x32 -> 16x16
        ResidualCCBlock_FirstDown_SinAct(400, 400),  # 2 16x16 -> 8x8
        ResidualCCBlock_FirstDown_SinAct(400, 400),  # 1 8x8 -> 4x4
        ResidualCCBlock_FirstDown_SinAct(400, 400),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400),
        AdapterBlock(400)
      ])
    self.final_layer = nn.Conv2d(400, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.fromRGB[start],
                                   inputs_args=(input, ),
                                   submodels=['model', ],
                                   name_prefix=f"fromRGB[{start}].")
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'network.0', 'network.2'],
                                     name_prefix=f"layers[{start + i}].")
      x = layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x, ),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position



class CoordConvSinAct(nn.Module):
  """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

  def __init__(self,
               in_channels,
               out_channels,
               channels_per_group=16,
               **kwargs):
    super().__init__()

    self.coord_conv = nn.Conv2d(2, out_channels, **kwargs)
    self.sin_act = SinAct()

    self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
    # self.norm = nn.GroupNorm(num_groups=out_channels//channels_per_group, num_channels=out_channels)
    pass

  def forward(self, input):

    batch, _, H, W = input.shape
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=input.device),
                          torch.linspace(-1, 1, H, device=input.device))
    x = x.T  # (HxW, ) [[-1, ..., 1], ...]
    y = y.T  # (HxW, ) [[-1, ..., 1]^T, ...]

    xy = torch.stack((x, y), dim=0)
    xy = xy.expand((batch, -1, -1, -1))
    xy_fea = self.coord_conv(xy)
    xy_fea = self.sin_act(xy_fea)

    out = self.conv(input)
    # out = self.norm(out)
    out = xy_fea + out
    return out

class ResidualCCBlock_FirstDown_CoordConvSinAct(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3):
    super().__init__()
    p = kernel_size // 2
    self.network = nn.Sequential(
      CoordConvSinAct(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p),
      nn.LeakyReLU(0.2, inplace=True),
      CoordConvSinAct(planes, planes, kernel_size=kernel_size, stride=1, padding=p),
      nn.LeakyReLU(0.2, inplace=True)
    )
    self.network.apply(kaiming_leaky_init)

    self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    y = (y + identity) / math.sqrt(2)
    return y

@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator_FirstDown_CoordConvSinAct(nn.Module):
  """
  Coord_Conv_Sin (good);

  """
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    max_channel = 400
    self.layers = nn.ModuleList(
      [
        ResidualCCBlock_FirstDown_CoordConvSinAct(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock_FirstDown_CoordConvSinAct(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock_FirstDown_CoordConvSinAct(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock_FirstDown_CoordConvSinAct(256, max_channel),  # 3 32x32 -> 16x16
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 2 16x16 -> 8x8
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 1 8x8 -> 4x4
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel)
      ])
    self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.fromRGB[start],
                                   inputs_args=(input, ),
                                   submodels=['model', ],
                                   name_prefix=f"fromRGB[{start}].")
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'network.0', 'network.2'],
                                     input_padding=50,
                                     name_prefix=f"layers[{start + i}].")
      x = layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x, ),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position

@MODEL_REGISTRY.register(name_prefix=__name__)
class Discriminator_CoordConvSinAct(nn.Module):
  """
  Coord_Conv_Sin (good);
  Support 512 and 1024;

  """
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    max_channel = 400
    self.layers = nn.ModuleList(
      [
        ResidualCCBlock_FirstDown_CoordConvSinAct(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock_FirstDown_CoordConvSinAct(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock_FirstDown_CoordConvSinAct(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock_FirstDown_CoordConvSinAct(256, max_channel),  # 3 32x32 -> 16x16
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 2 16x16 -> 8x8
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 1 8x8 -> 4x4
        ResidualCCBlock_FirstDown_CoordConvSinAct(max_channel, max_channel),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel)
      ])
    self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.fromRGB[start],
                                   inputs_args=(input, ),
                                   submodels=['model', ],
                                   name_prefix=f"fromRGB[{start}].")
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'network.0', 'network.2'],
                                     input_padding=50,
                                     name_prefix=f"layers[{start + i}].")
      x = layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x, ),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position


class CoordConvSinAct_EqualLR(nn.Module):
  """
  Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               **kwargs):
    super().__init__()

    self.coord_conv = mod_conv_fc.EqualConv2d(2, out_channels, kernel_size, stride, padding=padding)
    self.sin_act = SinAct()

    self.conv = mod_conv_fc.EqualConv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    pass

  def forward(self, input):

    batch, _, H, W = input.shape
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=input.device),
                          torch.linspace(-1, 1, H, device=input.device))
    x = x.T  # (HxW, ) [[-1, ..., 1], ...]
    y = y.T  # (HxW, ) [[-1, ..., 1]^T, ...]

    xy = torch.stack((x, y), dim=0)
    xy = xy.expand((batch, -1, -1, -1))
    xy_fea = self.coord_conv(xy)
    xy_fea = self.sin_act(xy_fea)

    out = self.conv(input)
    out = xy_fea + out
    return out

class ResidualCCBlockFirstDown_EqualLR(nn.Module):
  def __init__(self,
               inplanes,
               planes,
               stride=2,
               kernel_size=3,
               skip=True):
    super().__init__()
    self.skip = skip

    p = kernel_size // 2

    conv1 = CoordConvSinAct_EqualLR(in_channels=inplanes,
                                    out_channels=planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=p)
    conv2 = CoordConvSinAct_EqualLR(in_channels=planes,
                                    out_channels=planes,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=p)

    self.network = nn.Sequential(
      conv1,
      nn.LeakyReLU(0.2, inplace=True),
      conv2,
      nn.LeakyReLU(0.2, inplace=True),
    )
    # self.network.apply(kaiming_leaky_init)

    if skip:
      self.proj = mod_conv_fc.EqualConv2d(inplanes, planes, 1, stride)

    pass

  def forward(self, input):
    y = self.network(input)

    if self.skip:
      identity = self.proj(input)
      y = (y + identity) / math.sqrt(2)

    return y

@MODEL_REGISTRY.register(name_prefix=__name__)
class DiscriminatorMultiScale_EqualLR(nn.Module):
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):
    """

    :param kwargs:
    """
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.dim_z = dim_z

    logger = logging.getLogger('tl')
    self.epoch = 0
    self.step = 0
    self.layers = nn.ModuleList(
      [
        # 1024
        ResidualCCBlockFirstDown_EqualLR(3, 32),
        # 512
        ResidualCCBlockFirstDown_EqualLR(32, 64),
        # 256
        ResidualCCBlockFirstDown_EqualLR(64, 128),
        # 128
        ResidualCCBlockFirstDown_EqualLR(128, 256),
        # 64
        ResidualCCBlockFirstDown_EqualLR(256, 512),
        # 32
        ResidualCCBlockFirstDown_EqualLR(512, 512),
        # 16
        ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1),
        # 16
        ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1),
        # 16
        ResidualCCBlockFirstDown_EqualLR(512, 512, stride=1),
        # 16
        # ResidualCCBlock(400, 400),
        # 2
      ])

    final_layer = nn.Linear(512, 1 + dim_z + 2)
    self.final_layer = final_layer

    torch_utils.print_number_params(models_dict={'layers': self.layers,
                                                 'final_layer': self.final_layer,
                                                 'D': self})
    logger.info(self)
    pass

  def forward(self,
              x,
              alpha,
              **kwargs):

    img_size = x.shape[-1]
    if img_size < 128:
      x = F.upsample_bilinear(x, size=128)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers):

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x.clone(),),
                                     submodels=['network', 'network.0', 'network.2'],
                                     name_prefix=f"layers[{i}].",
                                     input_padding=50)
      x = layer(x)

    x = F.adaptive_avg_pool2d(x, output_size=1)
    x = x.flatten(1)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1+self.dim_z]
    position = x[..., 1+self.dim_z:]

    return prediction, latent, position


class ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3):
    super().__init__()
    p = kernel_size // 2
    self.network = nn.Sequential(
      CoordConvSinAct_EqualLR(inplanes, planes, kernel_size=kernel_size, stride=2, padding=p),
      nn.LeakyReLU(0.2, inplace=True),
      CoordConvSinAct_EqualLR(planes, planes, kernel_size=kernel_size, stride=1, padding=p),
      nn.LeakyReLU(0.2, inplace=True)
    )
    self.network.apply(kaiming_leaky_init)

    self.proj = mod_conv_fc.EqualConv2d(inplanes, planes, 1, 2)
    pass

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    y = (y + identity) / math.sqrt(2)
    return y

@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator_FirstDown_CoordConvSinAct_EqualLR(nn.Module):
  """
  CoordConv + GroupNorm
  """
  def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

  def __init__(self,
               dim_z=0,
               **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
    super().__init__()

    self.repr = f"dim_z={dim_z}"

    self.epoch = 0
    self.step = 0

    self.dim_z = dim_z

    max_channel = 400
    self.layers = nn.ModuleList(
      [
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(32, 64),  # 6 256x256 -> 128x128
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(64, 128),  # 5 128x128 -> 64x64
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(128, 256),  # 4 64x64 -> 32x32
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(256, max_channel),  # 3 32x32 -> 16x16
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel),  # 2 16x16 -> 8x8
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel),  # 1 8x8 -> 4x4
        ResidualCCBlock_FirstDown_CoordConvSinAct_EqualLR(max_channel, max_channel),  # 7 4x4 -> 2x2
      ])

    self.fromRGB = nn.ModuleList(
      [
        AdapterBlock(32),
        AdapterBlock(64),
        AdapterBlock(128),
        AdapterBlock(256),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel),
        AdapterBlock(max_channel)
      ])
    self.final_layer = nn.Conv2d(max_channel, 1 + self.dim_z + 2, 2)
    self.img_size_to_layer = {
      2: 7,
      4: 6,
      8: 5,
      16: 4,
      32: 3,
      64: 2,
      128: 1,
      256: 0
    }

    torch_utils.print_number_params(models_dict={
      'layers': self.layers,
      'fromRGB': self.fromRGB,
      'final_layer': self.final_layer,
      'D': self,
    })
    logging.getLogger('tl').info(self)
    pass

  def forward(self,
              input,
              alpha,
              options=None,
              **kwargs):

    start = self.img_size_to_layer[input.shape[-1]]

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.fromRGB[start],
                                   inputs_args=(input, ),
                                   submodels=['model', ],
                                   name_prefix=f"fromRGB[{start}].")
    x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      if i == 1 and alpha < 1:
        x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'network.0', 'network.2'],
                                     input_padding=50,
                                     name_prefix=f"layers[{start + i}].")
      x = layer(x)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x, ),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1 + self.dim_z]
    position = x[..., 1 + self.dim_z:]

    return prediction, latent, position
