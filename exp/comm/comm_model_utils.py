import logging
import random
import numpy as np
import math

import torch
import torch.nn as nn


class PosEmbedding(nn.Module):
  def __init__(self,
               max_logscale,
               N_freqs,
               logscale=True,
               multi_pi=False,):
    """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
    """
    super().__init__()

    self.N_freqs = N_freqs
    self.funcs = [torch.sin, torch.cos]

    if logscale:
      self.freqs = 2 ** torch.linspace(0, max_logscale, N_freqs)
    else:
      self.freqs = torch.linspace(1, 2 ** max_logscale, N_freqs)
    if multi_pi:
      self.freqs = self.freqs * math.pi
    pass

  def get_out_dim(self):
    outdim = 3 + 3 * 2 * self.N_freqs
    return outdim

  def forward(self, x):
    """
    Inputs:
        x: (B, 3)

    Outputs:
        out: (B, 6*N_freqs+3)
    """
    out = [x]
    for freq in self.freqs:
      for func in self.funcs:
        out += [func(freq * x)]

    return torch.cat(out, -1)



class EMA(object):
  def __init__(self,
               source,
               target,
               decay=0.9999,
               start_itr=0):
    """
    # Simple wrapper that applies EMA to a model. Could be better done in 1.0 using
    # the parameters() and buffers() module functions, but for now this works
    # with state_dicts using .copy_

    :param source: model
    :param target: ema model
    :param decay:
    :param start_itr:
    """
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr

    logger = logging.getLogger('tl')

    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    logger.info(f'Initializing EMA [{decay}] parameters to be source parameters...')
    self.update_target_dict(source_state_dict=self.source_dict)
    pass

  def update_target_dict(self, source_state_dict):
    """
    Reset the ema model weights.

    :param source_state_dict:
    :return:
    """
    with torch.no_grad():
      for key in source_state_dict:
        self.target_dict[key].data.copy_(source_state_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!
    pass

  def update(self, itr=None, source_dict=None):
    """
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.

    :param itr:
    :return:
    """

    if itr is not None and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      if source_dict is None:
        source_dict = self.source_dict

      for key in source_dict:
        self.target_dict[key].data.copy_(
          self.target_dict[key].data * decay + source_dict[key].data * (1 - decay))
    pass



