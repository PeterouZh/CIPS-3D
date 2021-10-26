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
    pass

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

  def __init__(self,
               in_channels,
               out_channels,
               with_r=False,
               **kwargs):
    super().__init__()
    self.addcoords = AddCoords(with_r=with_r)
    in_size = in_channels + 2
    if with_r:
      in_size += 1
    self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
    pass

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret

  def __repr__(self):
    repr = f"{self.__class__.__name__}(conv={str(self.conv)})"
    return repr


def kaiming_leaky_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
    torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class ResidualCCBlock(nn.Module):
  def __init__(self,
               inplanes,
               planes,
               kernel_size=3):
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
    pass

  def forward(self, input):
    y = self.network(input)

    identity = self.proj(input)

    # y = (y + identity) / math.sqrt(2)
    y = y + identity
    return y


class AdapterBlock(nn.Module):
  def __init__(self, output_channels):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(3, output_channels, 1, padding=0),
      nn.LeakyReLU(0.2, inplace=True)
    )
    pass

  def forward(self, input):
    out = self.model(input)
    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
class CCSEncoderDiscriminator(nn.Module):
  def __init__(self,
               dim_z=256,
               **kwargs):
    """
    # from 4 * 2^0 to 4 * 2^7 4 -> 512

    :param kwargs:
    """
    super().__init__()

    self.dim_z = dim_z

    logger = logging.getLogger('tl')
    self.epoch = 0
    self.step = 0
    self.layers = nn.ModuleList(
      [
        # 1024
        ResidualCCBlock(16, 32),
        # 512
        ResidualCCBlock(32, 64),
        # 256
        ResidualCCBlock(64, 128),
        # 128
        ResidualCCBlock(128, 256),
        # 64
        ResidualCCBlock(256, 400),
        # 32
        ResidualCCBlock(400, 400),
        # 16
        ResidualCCBlock(400, 400),
        # 8
        ResidualCCBlock(400, 400),
        # 4
        # ResidualCCBlock(400, 400),
        # 2
      ])
    self.base_scale = int(math.log2(4))

    self.first_layer = nn.ModuleList([
      AdapterBlock(16),
      AdapterBlock(32),
      AdapterBlock(64),
      AdapterBlock(128),
      AdapterBlock(256),
      AdapterBlock(400),
      AdapterBlock(400),
      AdapterBlock(400),
      # AdapterBlock(400),
    ])
    assert len(self.layers) == len(self.first_layer)

    # self.first_layer = nn.ModuleDict({
    #   "-9": AdapterBlock(16),
    #   "-8": AdapterBlock(32),
    #   "-7": AdapterBlock(64),
    #   "-6": AdapterBlock(128),
    #   "-5": AdapterBlock(256),
    #   "-4": AdapterBlock(400),
    #   "-3": AdapterBlock(400),
    #   "-2": AdapterBlock(400),
    #   "-1": AdapterBlock(400),
    # })

    # self.final_layer = nn.ModuleDict({
    #   "-9": nn.Conv2d(64, 1 + dim_z + 2, 2),
    #   "-8": nn.Conv2d(64, 1 + dim_z + 2, 2),
    #   "-7": nn.Conv2d(64, 1 + dim_z + 2, 2),
    #   "-6": nn.Conv2d(128, 1 + dim_z + 2, 2),
    #   "-5": nn.Conv2d(256, 1 + dim_z + 2, 2),
    #   "-4": nn.Conv2d(400, 1 + dim_z + 2, 2),
    #   "-3": nn.Conv2d(400, 1 + dim_z + 2, 2),
    #   "-2": nn.Conv2d(400, 1 + dim_z + 2, 2),
    #   "-1": nn.Conv2d(400, 1 + dim_z + 2, 2),
    # })
    self.final_layer = nn.Conv2d(400, 1 + dim_z + 2, 2)

    torch_utils.print_number_params(models_dict={'layers': self.layers,
                                                 'first_layer': self.first_layer,
                                                 'final_layer': self.final_layer,
                                                 'D': self},
                                    logger=logger)
    pass

  def forward(self,
              x,
              alpha,
              **kwargs):

    img_size = x.shape[-1]
    start = int(math.log2(img_size)) - self.base_scale
    start = min(start, len(self.layers))

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.first_layer[-start],
                                   inputs_args=(x.clone(), ),
                                   submodels=['model'],
                                   name_prefix=f"first_layer[-{start}].")
    x = self.first_layer[-start](x)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[-start:]):

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'proj'],
                                     name_prefix=f"layers[{len(self.layers) - start + i}].")
      x = layer(x)

    x = F.adaptive_avg_pool2d(x, output_size=2)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer,
                                   inputs_args=(x,),
                                   name_prefix=f"final_layer.")
    x = self.final_layer(x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1+self.dim_z]
    position = x[..., 1+self.dim_z:]

    return prediction, latent, position


class ResidualCCBlockFirstDown(nn.Module):
  def __init__(self,
               inplanes,
               planes,
               stride=2,
               kernel_size=3,
               spectral_norm=True,
               skip=True):
    super().__init__()
    self.skip = skip

    p = kernel_size // 2

    conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=p, stride=stride)
    # if spectral_norm:
    #   conv1 = nn.utils.spectral_norm(conv1)

    conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=p)
    if spectral_norm:
      conv2 = nn.utils.spectral_norm(conv2)

    self.network = nn.Sequential(
      conv1,
      nn.LeakyReLU(0.2, inplace=True),
      # nn.ReLU(inplace=True),
      conv2,
      nn.LeakyReLU(0.2, inplace=True),
      # nn.ReLU(inplace=True),
    )
    self.network.apply(kaiming_leaky_init)

    if skip:
      proj = nn.Conv2d(inplanes, planes, 1, stride=stride)
      if spectral_norm:
        proj = nn.utils.spectral_norm(proj)
      self.proj = proj
    else:
      self.proj = nn.Identity()
    pass

  def forward(self, input):
    y = self.network(input)

    if self.skip:
      identity = self.proj(input)
    else:
      identity = 0

    # y = (y + identity) / math.sqrt(2)
    y = y + identity
    return y


@MODEL_REGISTRY.register(name_prefix=__name__)
class DiscriminatorMultiScale(nn.Module):
  def __init__(self,
               dim_z=256,
               spectral_norm=False,
               **kwargs):
    """
    # from 4 * 2^0 to 4 * 2^7 4 -> 512

    :param kwargs:
    """
    super().__init__()

    self.dim_z = dim_z
    self.spectral_norm = spectral_norm

    logger = logging.getLogger('tl')
    self.epoch = 0
    self.step = 0
    self.layers = nn.ModuleList(
      [
        # 1024
        ResidualCCBlockFirstDown(3, 16, spectral_norm=spectral_norm),
        # 512
        ResidualCCBlockFirstDown(16, 32, spectral_norm=spectral_norm),
        # 256
        ResidualCCBlockFirstDown(32, 64, spectral_norm=spectral_norm),
        # 128
        ResidualCCBlockFirstDown(64, 128, spectral_norm=spectral_norm),
        # 64
        ResidualCCBlockFirstDown(128, 256, spectral_norm=spectral_norm),
        # 32
        ResidualCCBlockFirstDown(256, 512, spectral_norm=spectral_norm),
        # 16
        ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm),
        # 16
        ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm),
        # 16
        ResidualCCBlockFirstDown(512, 512, stride=1, spectral_norm=spectral_norm),
        # 16
        # ResidualCCBlock(400, 400),
        # 2
      ])
    self.layers.apply(init_func.kaiming_leaky_init)

    # self.first_layer = AdapterBlock(16)
    # self.final_layer = nn.Conv2d(512, 1 + dim_z + 2, 2)

    final_layer = nn.Linear(512, 1 + dim_z + 2)
    # if spectral_norm:
    #    final_layer = nn.utils.spectral_norm(final_layer)
    self.final_layer = final_layer

    torch_utils.print_number_params(models_dict={'layers': self.layers,
                                                 # 'first_layer': self.first_layer,
                                                 'final_layer': self.final_layer,
                                                 'D': self},
                                    logger=logger)
    logger.info(self)
    pass

  def forward(self,
              x,
              alpha,
              **kwargs):

    img_size = x.shape[-1]
    if img_size < 128:
      x = F.upsample_bilinear(x, size=128)


    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.first_layer,
    #                                inputs_args=(x.clone(), ),
    #                                submodels=['model'],
    #                                name_prefix=f"first_layer.")
    # x = self.first_layer(x)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers):

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x.clone(),),
                                     submodels=['network', 'proj'],
                                     name_prefix=f"layers[{i}].")
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

  def __repr__(self):
    repr = f"{self.__class__.__name__}(" \
           f"dim_z={self.dim_z}, " \
           f"spectral_norm={self.spectral_norm})"
    return repr


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

    # conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=p, stride=stride)
    # conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=p)
    conv1 = mod_conv_fc.EqualConv2d(inplanes, planes, kernel_size, stride, padding=p)
    conv2 = mod_conv_fc.EqualConv2d(planes, planes, kernel_size, 1, padding=p)

    self.network = nn.Sequential(
      conv1,
      nn.LeakyReLU(0.2, inplace=True),
      # nn.ReLU(inplace=True),
      conv2,
      nn.LeakyReLU(0.2, inplace=True),
      # nn.ReLU(inplace=True),
    )
    # self.network.apply(kaiming_leaky_init)

    if skip:
      # proj = nn.Conv2d(inplanes, planes, 1, stride=stride)
      proj = mod_conv_fc.EqualConv2d(inplanes, planes, 1, stride)
      self.proj = proj
    else:
      self.proj = nn.Identity()
    pass

  def forward(self, input):
    y = self.network(input)

    if self.skip:
      identity = self.proj(input)
    else:
      identity = 0

    # y = (y + identity) / math.sqrt(2)
    y = y + identity
    return y


@MODEL_REGISTRY.register(name_prefix=__name__)
class DiscriminatorMultiScale_EqualLR(nn.Module):
  def __init__(self,
               dim_z=256,
               **kwargs):
    """
    # from 4 * 2^0 to 4 * 2^7 4 -> 512

    :param kwargs:
    """
    super().__init__()

    self.dim_z = dim_z

    logger = logging.getLogger('tl')
    self.epoch = 0
    self.step = 0
    self.layers = nn.ModuleList(
      [
        # 1024
        ResidualCCBlockFirstDown_EqualLR(3, 16),
        # 512
        ResidualCCBlockFirstDown_EqualLR(16, 32),
        # 256
        ResidualCCBlockFirstDown_EqualLR(32, 64),
        # 128
        ResidualCCBlockFirstDown_EqualLR(64, 128),
        # 64
        ResidualCCBlockFirstDown_EqualLR(128, 256),
        # 32
        ResidualCCBlockFirstDown_EqualLR(256, 512),
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
    # self.layers.apply(init_func.kaiming_leaky_init)

    # self.first_layer = AdapterBlock(16)
    # self.final_layer = nn.Conv2d(512, 1 + dim_z + 2, 2)

    final_layer = nn.Linear(512, 1 + dim_z + 2)
    self.final_layer = final_layer

    torch_utils.print_number_params(models_dict={'layers': self.layers,
                                                 # 'first_layer': self.first_layer,
                                                 'final_layer': self.final_layer,
                                                 'D': self},
                                    logger=logger)
    logger.info(self)
    pass

  def forward(self,
              x,
              alpha,
              **kwargs):

    img_size = x.shape[-1]
    if img_size < 128:
      x = F.upsample_bilinear(x, size=128)


    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.first_layer,
    #                                inputs_args=(x.clone(), ),
    #                                submodels=['model'],
    #                                name_prefix=f"first_layer.")
    # x = self.first_layer(x)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers):

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x.clone(),),
                                     submodels=['network', 'proj'],
                                     name_prefix=f"layers[{i}].")
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

  def __repr__(self):
    repr = f"{self.__class__.__name__}(" \
           f"dim_z={self.dim_z})"
    return repr