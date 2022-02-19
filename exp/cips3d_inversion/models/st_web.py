import math
import pathlib
import collections
from pathlib import Path
import logging
import os
import sys

import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, os.getcwd())

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as tv_f

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.proj.streamlit import SessionState
from tl2.proj.streamlit import st_utils
from tl2.proj.logger.logger_utils import get_file_logger
from tl2 import tl2_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.proj.pytorch import torch_utils
from tl2.proj.pytorch.examples.nerf import cam_params
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)


def interpolate_sphere(z1,
                       z2,
                       t):
  p = (z1 * z2).sum(dim=-1, keepdim=True)
  p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
  p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
  omega = torch.acos(p)
  s1 = torch.sin((1 - t) * omega) / torch.sin(omega)
  s2 = torch.sin(t * omega) / torch.sin(omega)
  z = s1 * z1 + s2 * z2
  return z


@MODEL_REGISTRY.register(name_prefix=__name__)
class STModel(object):
  def __init__(self):

    pass

  def interpolate_z_web(self,
                        cfg,
                        outdir,
                        saved_suffix_state=None,
                        **kwargs):

    network_pkl = st_utils.selectbox_v1('network_pkl', options_dict=cfg.network_pkl,
                                        default_key=cfg.default_network_pkl, sidebar=True)

    st_utils.st_set_sep('video kwargs')
    bs = st_utils.number_input('bs', cfg.bs, sidebar=True)
    img_size = st_utils.number_input('img_size', cfg.img_size, sidebar=True)
    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)

    st_utils.st_set_sep('interp kwargs')
    z_mode = st_utils.selectbox('z_mode', options=cfg.z_mode, default_value=cfg.default_z_mode, sidebar=True)
    interp_mode = st_utils.selectbox('interp_mode', options=cfg.interp_mode, sidebar=True)

    st_utils.st_set_sep('NeRF kwargs')
    nerf_N_samples = st_utils.number_input('nerf_N_samples', cfg.nerf_N_samples, sidebar=True)
    nerf_N_importance = st_utils.number_input('nerf_N_importance', cfg.nerf_N_importance, sidebar=True)
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    seed = st_utils.get_seed(seeds=cfg.seeds)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = 'cuda'
    torch_utils.init_seeds(seed=seed)

    G_kwargs = cfg.G_kwargs.clone()
    G_kwargs['nerf_kwargs']['h_stddev'] = 0.
    G_kwargs['nerf_kwargs']['v_stddev'] = 0.
    G_kwargs['nerf_kwargs']['N_samples'] = nerf_N_samples
    G_kwargs['nerf_kwargs']['N_importance'] = nerf_N_importance

    G = build_model(cfg.G_cfg).cuda()
    Checkpointer(G).load_state_dict_from_file(network_pkl)

    H = W = img_size
    cam_param = cam_params.CamParams.from_config(H0=H, W0=W).cuda()

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=bs, intr=intr, **G_kwargs.nerf_kwargs)

    st_image = st.empty()
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/video.mp4", fps=fps, hd_video=True)

    G.eval()
    zs = G.get_zs(bs)
    zs_list = [G.get_zs(bs) for _ in range(N_samples)]

    for idx in range(N_samples):
      zs1 = zs_list[idx]
      zs2 = zs_list[(idx + 1) % N_samples]
      ts = np.linspace(0, 1, N_step)
      for t in ts:
        z1_ = zs1[z_mode]
        z2_ = zs2[z_mode]
        if interp_mode == 'slerp':
          z_interp_ = interpolate_sphere(z1_, z2_, t)
        elif interp_mode == 'lerp':
          z_interp_ = torch.lerp(z1_, z2_, t)
        else:
          assert 0
        zs[z_mode] = z_interp_

        with torch.set_grad_enabled(False):
          imgs, ret_imgs = G(zs=zs,
                             rays_o=rays_o,
                             rays_d=rays_d,
                             forward_points=forward_points ** 2,  # disable gradients
                             return_aux_img=True,
                             **G_kwargs)
        img = make_grid(imgs, nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)
        img_pil = tv_f.to_pil_image(img)
        img_str = f"{idx}-{(idx+1)%N_samples}/{N_samples}, t={t:.2f}"
        pil_utils.add_text(img_pil, img_str, size=img_pil.size[0]//18)

        st_utils.st_image(img_pil, caption=f"{imgs.shape}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        video_f.write(img_pil)

    video_f.release(st_video=True)

    pass

  def sampling_yaw_web(self,
                       cfg,
                       outdir,
                       saved_suffix_state=None,
                       **kwargs):

    network_pkl = st_utils.selectbox_v1('network_pkl', options_dict=cfg.network_pkl,
                                        default_key=cfg.default_network_pkl, sidebar=True)
    network_pkl_model = st_utils.selectbox_v1('network_pkl_model', options_dict=cfg.network_pkl_model,
                                              default_key=cfg.default_network_pkl_model, sidebar=True)
    use_network_pkl_model = st_utils.checkbox('use_network_pkl_model', cfg.use_network_pkl_model)

    st_utils.st_set_sep('video kwargs')
    bs = st_utils.number_input('bs', cfg.bs, sidebar=True)
    img_size = st_utils.number_input('img_size', cfg.img_size, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)

    st_utils.st_set_sep('NeRF kwargs')
    nerf_N_samples = st_utils.number_input('nerf_N_samples', cfg.nerf_N_samples, sidebar=True)
    nerf_N_importance = st_utils.number_input('nerf_N_importance', cfg.nerf_N_importance, sidebar=True)
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    st_utils.st_set_sep('pose kwargs')
    yaw_min = st_utils.number_input('yaw_min', cfg.yaw_min, sidebar=True, format="%.3f")
    yaw_max = st_utils.number_input('yaw_max', cfg.yaw_max, sidebar=True, format="%.3f")
    N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)

    seed = st_utils.get_seed(seeds=cfg.seeds)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = 'cuda'
    torch_utils.init_seeds(seed=seed)

    G_kwargs = cfg.G_kwargs.clone()
    G_kwargs['nerf_kwargs']['h_stddev'] = 0.
    G_kwargs['nerf_kwargs']['v_stddev'] = 0.
    G_kwargs['nerf_kwargs']['N_samples'] = nerf_N_samples
    G_kwargs['nerf_kwargs']['N_importance'] = nerf_N_importance

    if use_network_pkl_model:
      G = torch.load(network_pkl_model).cuda()
    else:
      G = build_model(cfg.G_cfg).cuda()
      Checkpointer(G).load_state_dict_from_file(network_pkl)

    H = W = img_size
    cam_param = cam_params.CamParams.from_config(H0=H, W0=W).cuda()
    intr = cam_param(mode='get_intrinsic')


    st_image = st.empty()
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/video.mp4", fps=fps, hd_video=True)

    G.eval()
    zs = G.get_zs(bs)

    yaw_list = np.linspace(yaw_min, yaw_max, N_step)
    for idx, yaw_ in enumerate(yaw_list):
      rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
        device=device, bs=bs, intr=intr,
        **{**G_kwargs.nerf_kwargs,
           'h_mean': yaw_,
           'h_stddev': 0,
           'v_stddev': 0})

      with torch.set_grad_enabled(False):
        imgs, ret_imgs = G(zs=zs,
                           rays_o=rays_o,
                           rays_d=rays_d,
                           forward_points=forward_points ** 2,  # disable gradients
                           return_aux_img=True,
                           **G_kwargs)
      img = make_grid(imgs, nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)
      img_pil = tv_f.to_pil_image(img)
      img_str = f"{idx}/{N_step}, yaw={yaw_:.2f}"
      pil_utils.add_text(img_pil, img_str, size=img_pil.size[0] // 18)

      st_utils.st_image(img_pil, caption=f"{imgs.shape}",
                        debug=global_cfg.tl_debug, st_empty=st_image)
      video_f.write(img_pil)

    video_f.release(st_video=True)

    pass
