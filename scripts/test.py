import copy
from pathlib import Path
import math
import numpy as np
import os
import tqdm
import streamlit as st
import torch
import sys

sys.path.insert(0, os.getcwd())

from tl2.proj.fvcore import TLCfgNode, global_cfg
from tl2.proj.fvcore import build_model
from tl2.proj.argparser import argparser_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.logger.logging_utils_v2 import get_logger

from exp.comm import comm_utils

class CIPS_3D_Demo(object):
  def __init__(self):

    pass

  def model(self,
            cfg,
            outdir,
            debug=False,
            **kwargs):

    network_pkl = st_utils.selectbox('network_pkl', cfg.network_pkl)
    model_pkl_input = st_utils.text_input('model_pkl', "", sidebar=False)

    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)

    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    num_frames = st_utils.number_input('num_frames', cfg.num_frames, sidebar=True)

    num_samples_translate = st_utils.number_input('num_samples_translate', cfg.num_samples_translate, sidebar=True)
    translate_dist = st_utils.number_input('translate_dist', 0.04, sidebar=True)

    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)
    max_fov = st_utils.number_input('max_fov', cfg.max_fov, sidebar=True)
    alpha_pi_div = st_utils.number_input('alpha_pi_div', cfg.alpha_pi_div, sidebar=True)

    # seed
    seed = st_utils.get_seed(cfg.seeds_gallery)

    # trajectory
    trajectory_mode = st_utils.selectbox('trajectory_mode', cfg.trajectory_mode, sidebar=True)
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    # ****************************************************************************
    # if not debug:
    #   if not st.sidebar.button("run_web"):
    #     return

    device = torch.device('cuda')

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')
    generator = build_model(cfg=cfg.G_cfg).to(device)
    Checkpointer(generator).load_state_dict_from_file(model_pkl)


   
    curriculum = comm_utils.get_metadata_from_json(metafile=cfg.metadata,
                                                   num_steps=num_steps,
                                                   image_size=image_size,
                                                   psi=psi)

    if trajectory_mode == 'circle_near_far':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(
        alpha=math.pi / alpha_pi_div, num_samples=num_frames, periods=2)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = []
      for idx, t in enumerate(np.linspace(0, 1, num_frames)):
        fov_list.append(fov + t * (max_fov - fov))
      fov_list.extend(fov_list[::-1])

    elif trajectory_mode == 'translate_circle_near_far':
      xyz, lookup, yaws, pitchs, num_samples_translate = comm_utils.get_translate_circle_camera_pos_and_lookup(
        num_samples_translate=num_samples_translate,
        translate_dist=translate_dist,
        alpha=math.pi / alpha_pi_div,
        num_samples=num_frames,
        periods=2)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * num_samples_translate * 2
      for idx, t in enumerate(np.linspace(0, 1, num_frames)):
        fov_list.append(fov + t * (max_fov - fov))
      fov_list.extend(fov_list[-num_frames:][::-1])
      assert len(fov_list) == len(xyz)

    elif trajectory_mode == 'circle':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / alpha_pi_div,
                                                                              num_samples=num_frames,
                                                                              periods=2)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    elif trajectory_mode == 'yaw':
      xyz, lookup, yaws, pitchs = comm_utils.get_yaw_camera_pos_and_lookup(num_samples=num_frames, )
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    st_image = st.empty()
    output_name = Path(f'seed_{seed}.mp4')
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name}", fps=fps)

    torch.manual_seed(seed)
    zs = generator.get_zs(1)

    with torch.no_grad():
      for idx in tqdm.tqdm(range(len(xyz))):
        curriculum['h_mean'] = 0
        curriculum['v_mean'] = 0
        curriculum['h_stddev'] = 0
        curriculum['v_stddev'] = 0

        cur_camera_pos = xyz[[idx]]
        cur_camera_lookup = lookup[[idx]]
        yaw = yaws[idx]
        pitch = pitchs[idx]
        fov = fov_list[idx]
        curriculum['fov'] = fov

        frame, depth_map = generator.forward_camera_pos_and_lookup(
          zs=zs,
          return_aux_img=False,
          forward_points=forward_points ** 2,
          camera_pos=cur_camera_pos,
          camera_lookup=cur_camera_lookup,
          **curriculum)
        frame_pil = comm_utils.to_pil(frame)

        st_utils.st_image(frame_pil, caption=f"{frame_pil.size}, seed={seed}",
                          debug=debug, st_empty=st_image)
        video_f.write(merged_frame)
    
      video_f.release(st_video=True)

    pass



def main(outdir,
         cfg_file,
         command,
         debug,
         **kwargs
         ):

  os.makedirs(outdir, exist_ok=True)
  global_cfg.tl_debug = False

  command_cfg = TLCfgNode.load_yaml_with_command(cfg_filename=cfg_file, command=command)

  logger = get_logger(filename=f"{outdir}/log.txt", logger_names=['st'], stream=False)
  logger.info(f"command_cfg:\n {command_cfg.dump()}")

  st_model = CIPS_3D_Demo()

  mode = st_utils.selectbox(label='mode', options=command_cfg.mode, sidebar=True)
  getattr(st_model, mode)(cfg=command_cfg.get(mode, {}),
                          outdir=outdir,
                          debug=debug)

  pass


def build_args():
  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_str(parser, name='outdir', default='results/model_interpolation')
  argparser_utils.add_argument_str(parser, name='cfg_file', default='configs/web_demo.yaml')
  argparser_utils.add_argument_str(parser, name='command', default='model_interpolation')
  argparser_utils.add_argument_bool(parser, name='debug', default=False)

  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)
  return args

if __name__ == '__main__':

  args = build_args()

  main(**vars(args))


