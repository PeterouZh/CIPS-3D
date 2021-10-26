import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.pytorch.pytorch_hook import VerboseModel
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pytorch import torch_utils


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
        ResidualCCBlock(3, 32),     
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

    self.final_layer = nn.ModuleDict({
      "1": nn.Conv2d(64, 1 + dim_z + 2, 2), # 16
      "2": nn.Conv2d(128, 1 + dim_z + 2, 2), # 32
      "3": nn.Conv2d(256, 1 + dim_z + 2, 2), # 64
      "4": nn.Conv2d(400, 1 + dim_z + 2, 2),
      "5": nn.Conv2d(400, 1 + dim_z + 2, 2),
      "6": nn.Conv2d(400, 1 + dim_z + 2, 2),
      "7": nn.Conv2d(400, 1 + dim_z + 2, 2),
    })

    torch_utils.print_number_params(models_dict={'layers': self.layers,
                                                 'final_layer': self.final_layer,
                                                 'D': self},
                                    logger=logger)
    pass

  def forward(self,
              x,
              alpha,
              **kwargs):

    # start = self.img_size_to_layer[input.shape[-1]]
    start = 0
    # if global_cfg.tl_debug:
    #   VerboseModel.forward_verbose(self.fromRGB[0],
    #                                inputs_args=(input, ),
    #                                submodels=['model'],
    #                                name_prefix=f"fromRGB[{start}].")
    # x = self.fromRGB[start](input)

    if kwargs.get('instance_noise', 0) > 0:
      x = x + torch.randn_like(x) * kwargs['instance_noise']

    for i, layer in enumerate(self.layers[start:]):
      # if i == 1 and alpha < 1:
      #   skip = F.interpolate(input, scale_factor=0.5, mode='nearest')
      #   x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](skip)

      if global_cfg.tl_debug:
        VerboseModel.forward_verbose(layer,
                                     inputs_args=(x,),
                                     submodels=['network', 'proj'],
                                     name_prefix=f"layers[{i}].")
      x = layer(x)
      if x.shape[-1] <= 2:
        break

    x = F.adaptive_avg_pool2d(x, output_size=2)

    if global_cfg.tl_debug:
      VerboseModel.forward_verbose(self.final_layer[f"{i}"],
                                   inputs_args=(x,),
                                   name_prefix=f"final_layer[{i}].")
    x = self.final_layer[f"{i}"](x).reshape(x.shape[0], -1)

    prediction = x[..., 0:1]
    latent = x[..., 1:1+self.dim_z]
    position = x[..., 1+self.dim_z:]

    return prediction, latent, position
