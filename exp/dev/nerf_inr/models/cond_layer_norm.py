import os
import sys
import unittest
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLN(nn.Module):
  def __init__(self,
               in_dim,
               use_style_fc=False,
               style_dim=None,
               which_linear=nn.Linear,
               spectral_norm=False,
               eps=1e-5,
               **kwargs):
    super(CLN, self).__init__()

    self.in_dim = in_dim
    self.use_style_fc = use_style_fc
    self.style_dim = style_dim

    self.spectral_norm = spectral_norm
    # Prepare gain and bias layers

    if use_style_fc:
      self.gain = which_linear(style_dim, in_dim)
      self.bias = which_linear(style_dim, in_dim)
      if spectral_norm:
        self.gain = nn.utils.spectral_norm(self.gain)
        self.bias = nn.utils.spectral_norm(self.bias)
    else:
      self.style_dim = in_dim * 2

    self.eps = eps
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

    if self.use_style_fc:
      gain = self.gain(style) + 1.
      bias = self.bias(style)
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

    out = F.layer_norm(x,
                       normalized_shape=(self.in_dim,),
                       weight=None,
                       bias=None,
                       eps=self.eps)

    out = out * gain + bias
    return out

  def __repr__(self):
    s = f'{self.__class__.__name__}(in_dim={self.in_dim}, style_dim={self.style_dim})'
    return s


class Testing_CLN(unittest.TestCase):
  """

  """
  def test__build_CLN(self, debug=True):
    """
    Usage:
        proj_root=pi-GAN-exp
        python tl2_lib/tl2/modelarts/scripts/copy_tool.py \
          -s s3://bucket-7001/ZhouPeng/codes/$proj_root -d /cache/$proj_root -t copytree -b /cache/$proj_root/code.zip
        cd /cache/$proj_root
        cp tl2_lib/tl2/modelarts/sources/pip.conf.modelarts /root/.pip/pip.conf
        cp tl2_lib/tl2/modelarts/sources/sources.list.modelarts /etc/apt/sources.list
        pip install -e tl2_lib

        pip install -e pytorch_ema_lib
        export CUDA_VISIBLE_DEVICES=6,7
        export TIME_STR=1
        export PORT=12346
        export PYTHONPATH=.:./piGAN_lib
        python -c "from exp.tests.test_pigan import Testing_train;\
          Testing_train().test_train_celeba(debug=False)" \
          --tl_opts curriculum.0.batch_size 14

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
    # tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    # print(f'tl_opts:\n {tl_opts}')

    # debug = False

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                {"--tl_debug" if debug else ""}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)

    in_dim = 128
    style_dim = 128 * 2
    num_seq = 10

    x = torch.randn(2, num_seq, in_dim).requires_grad_()
    y = torch.randn(2, style_dim).requires_grad_()

    net = CLN(in_dim=in_dim)
    print(net)
    out = net(x, y)
    loss = out.mean()
    loss.backward()

    x = torch.randn(2, in_dim)
    y = torch.randn(2, style_dim)
    out = net(x, y)
    pass