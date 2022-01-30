import copy
import json
from pathlib import Path
import math
import numpy as np
import os
import tqdm
from PIL import Image
import streamlit as st
import collections

import torch
from torchvision import transforms
import torchvision.transforms.functional as trans_f
from torchvision.utils import save_image, make_grid

from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.launch.launch_utils import global_cfg
from tl2.proj.streamlit import st_utils
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2 import tl2_utils

# import curriculums
from exp.dev.nerf_inr import curriculums
from exp.pigan import pigan_utils
from exp.comm import comm_utils


def generate_img(G, z, **kwargs):
  with torch.no_grad():
    img, depth_map = G.staged_forward(z, **kwargs)
    tensor_img = img.detach()

    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)
    img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
  return img, tensor_img, depth_map


def tensor_to_PIL(img):
  img = img.squeeze() * 0.5 + 0.5
  return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


class FrequencyInterpolator:
  def __init__(self, generator, z1, z2, psi=0.5):
    avg_frequencies, avg_phase_shifts = generator.generate_avg_frequencies()
    raw_frequencies1, raw_phase_shifts1 = generator.siren.mapping_network(z1)
    self.truncated_frequencies1 = avg_frequencies + psi * (raw_frequencies1 - avg_frequencies)
    self.truncated_phase_shifts1 = avg_phase_shifts + psi * (raw_phase_shifts1 - avg_phase_shifts)
    raw_frequencies2, raw_phase_shifts2 = generator.siren.mapping_network(z2)
    self.truncated_frequencies2 = avg_frequencies + psi * (raw_frequencies2 - avg_frequencies)
    self.truncated_phase_shifts2 = avg_phase_shifts + psi * (raw_phase_shifts2 - avg_phase_shifts)

  def forward(self, t):
    frequencies = self.truncated_frequencies1 * (1 - t) + self.truncated_frequencies2 * t
    phase_shifts = self.truncated_phase_shifts1 * (1 - t) + self.truncated_phase_shifts2 * t

    return frequencies, phase_shifts


class Projector(object):

  def project_image(self,
                    outdir,
                    model_pkl,
                    image_path,
                    image_size,
                    n_iterations,
                    max_batch_size,
                    num_frames,
                    fps,
                    lock_view_dependence=True,
                    seed=0,
                    st_web=False,
                    st_log_every=10,
                    ):

    device = torch.device('cuda')
    image_path = Path(image_path)

    # generator = torch.load(model_pkl, map_location=torch.device(device))
    # ema_file = model_pkl.split('generator')[0] + 'ema.pth'
    # ema = torch.load(ema_file, map_location=device)
    # ema.copy_to(generator.parameters())
    # generator.set_device(device)
    # generator.eval()
    generator = pigan_utils.load_generator_ema(model_pkl=model_pkl, device=device)

    if seed is not None:
      torch.manual_seed(seed)

    gt_image_pil = Image.open(image_path).convert('RGB')
    gt_image_pil = gt_image_pil.resize((image_size, image_size), Image.LANCZOS)
    if st_web:
      st_utils.st_image(gt_image_pil, caption=f"gt_image_{image_path.stem} {gt_image_pil.size}",
                        debug=global_cfg.tl_debug, outdir=outdir)

    transform = transforms.Compose(
      [
        # transforms.Resize(256),
        # transforms.CenterCrop(256),
        # transforms.Resize((image_size, image_size), interpolation=0),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
      ])
    gt_image = transform(gt_image_pil).to(device)

    options = {
      'img_size': image_size,
      'fov': 12,
      'ray_start': 0.88,
      'ray_end': 1.12,
      'num_steps': 24,
      'h_stddev': 0,
      'v_stddev': 0,
      'h_mean': torch.tensor(math.pi / 2).to(device),
      'v_mean': torch.tensor(math.pi / 2).to(device),
      'hierarchical_sample': False,
      'sample_dist': None,
      'clamp_mode': 'relu',
      'nerf_noise': 0,
    }

    render_options = {
      'img_size': 256,
      'fov': 12,
      'ray_start': 0.88,
      'ray_end': 1.12,
      'num_steps': 48,
      'h_stddev': 0,
      'v_stddev': 0,
      'v_mean': math.pi / 2,
      'hierarchical_sample': True,
      'sample_dist': None,
      'clamp_mode': 'relu',
      'nerf_noise': 0,
      'last_back': True,
    }

    z = torch.randn((10000, 256), device=device)
    with torch.no_grad():
      frequencies, phase_shifts = generator.siren.mapping_network(z)
    w_frequencies = frequencies.mean(0, keepdim=True)
    w_phase_shifts = phase_shifts.mean(0, keepdim=True)

    w_frequency_offsets = torch.zeros_like(w_frequencies)
    w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)
    w_frequency_offsets.requires_grad_()
    w_phase_shift_offsets.requires_grad_()

    frames = []
    n_iterations_pose = 0

    # os.makedirs('debug', exist_ok=True)
    # save_image(gt_image, f"{outdir}/gt.jpg", normalize=True)

    optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.75)
    # l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    if st_web:
      st_linechart_lr = st_utils.LineChart(x_label='step', y_label='lr')
      st_linechart_mse_loss = st_utils.LineChart(x_label='step', y_label='mse_loss')
      st_frame = st.empty()
      video_frame = cv2_utils.ImageioVideoWriter(f"{outdir}/frame.mp4", fps=fps)

    pbar = tqdm.tqdm(range(n_iterations))
    for i in pbar:
      noise_w_frequencies = 0.03 * torch.randn_like(w_frequencies) * (n_iterations - i) / n_iterations
      noise_w_phase_shifts = 0.03 * torch.randn_like(w_phase_shifts) * (n_iterations - i) / n_iterations
      frame, _ = generator.forward_with_frequencies(
        w_frequencies + noise_w_frequencies + w_frequency_offsets,
        w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets,
        **options)
      loss = mse_loss(frame.squeeze(), gt_image)
      loss = loss.mean()

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      scheduler.step()

      if st_web and i % st_log_every == 0:
        st_linechart_mse_loss.write(i, loss.item())
        cur_lr = scheduler.get_last_lr()[0]
        st_linechart_lr.write(i, cur_lr)

        gt_img_pil_text = pil_utils.add_text(
          img=gt_image_pil, text=f"step: {i}", size=gt_image_pil.size[0]//10, clone=True)
        frame_pil = pigan_utils.tensor_to_PIL(img_tensor=frame)
        merged_pil = pil_utils.merge_image_pil([gt_img_pil_text, frame_pil], nrow=2)
        st_utils.st_image(merged_pil, caption=f"{frame_pil.size}", debug=global_cfg.tl_debug,
                          st_empty=st_frame)
        video_frame.write(merged_pil)

      if global_cfg.tl_debug: break
      pass

    if st_web:
      video_frame.release(st_video=True)
      st_multi_view = st.empty()
      video_multi_view = cv2_utils.ImageioVideoWriter(f"{outdir}/multi_view.mp4", fps=1)

    # multi-view
    multi_view_outdir = f"{outdir}/multi_view"
    os.makedirs(multi_view_outdir, exist_ok=True)
    # save_image(frame, f"{outdir}/{i}.jpg", normalize=True)
    with torch.no_grad():
      for angle in [-0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7]:
        img, _ = generator.staged_forward_with_frequencies(
          w_frequencies + w_frequency_offsets,
          w_phase_shifts + w_phase_shift_offsets,
          h_mean=(math.pi / 2 + angle),
          max_batch_size=max_batch_size,
          lock_view_dependence=lock_view_dependence,
          **render_options)
        # save_image(img, f"{outdir}/{i}_{angle}.jpg", normalize=True)
        if st_web:
          gt_img_pil_text = pil_utils.add_text(
            img=gt_image_pil, text=f"angle={angle}", size=gt_image_pil.size[0] // 5, clone=True)
          img_angle_pil = pigan_utils.tensor_to_PIL(img)
          merged_pil = pil_utils.merge_image_pil([gt_img_pil_text, img_angle_pil], nrow=2)
          st_utils.st_image(merged_pil, caption=f"angle_{angle}_{img_angle_pil.size}", debug=global_cfg.tl_debug,
                            st_empty=st_multi_view)
          video_multi_view.write(merged_pil)

        if global_cfg.tl_debug: break
    if st_web:
      video_multi_view.release(st_video=True)

    trajectory = []
    for t in np.linspace(0, 1, 24):
      pitch = 0.2 * t
      yaw = 0
      trajectory.append((pitch, yaw))
    for t in np.linspace(0, 1, num_frames):
      pitch = 0.2 * np.cos(t * 2 * math.pi)
      yaw = 0.4 * np.sin(t * 2 * math.pi)
      trajectory.append((pitch, yaw))

    # writer = skvideo.io.FFmpegWriter(
    #   os.path.join(cfg.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    if st_web:
      video_recons = cv2_utils.ImageioVideoWriter(f"{outdir}/reconstructed.mp4", fps=fps)
      st_recons = st.empty()

    depths = []
    pbar = tqdm.tqdm(trajectory)
    with torch.no_grad():
      for idx, (pitch, yaw) in enumerate(pbar):
        render_options['h_mean'] = yaw + 3.14 / 2
        render_options['v_mean'] = pitch + 3.14 / 2

        frame, depth_map = generator.staged_forward_with_frequencies(
          w_frequencies + w_frequency_offsets,
          w_phase_shifts + w_phase_shift_offsets,
          max_batch_size=max_batch_size,
          lock_view_dependence=lock_view_dependence,
          **render_options)
        frame_pil = tensor_to_PIL(frame)
        if st_web:
          gt_img_pil_text = pil_utils.add_text(
            img=gt_image_pil, text=f"{idx}: pitch={pitch:.3f}\nyaw={yaw:.3f}", size=gt_image_pil.size[0] // 10,
            clone=True)
          merged_pil = pil_utils.merge_image_pil([gt_img_pil_text, frame_pil], nrow=2)
          st_utils.st_image(merged_pil, caption=f"{frame_pil.size}", debug=global_cfg.tl_debug,
                            st_empty=st_recons)
          video_recons.write(merged_pil)

        depths.append(depth_map.unsqueeze(0).expand(-1, 3, -1, -1).squeeze().permute(1, 2, 0).cpu().numpy())
        if global_cfg.tl_debug: break
    if st_web:
      video_recons.release(st_video=True)

    pass


@MODEL_REGISTRY.register(name_prefix=__name__)
class CIPS_3D_Web(object):
  def __init__(self):

    pass

  def render_video_web(self,
                       cfg,
                       outdir,
                       saved_suffix_state=None,
                       **kwargs):

    # mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    ckpt_dir = st_utils.selectbox_v1('ckpt_dir', options_dict=cfg.ckpt_dir,
                                     default_key=cfg.default_ckpt_dir, sidebar=True)
    ckpt_dir_input = st_utils.text_input('ckpt_dir', "", sidebar=False)
    if ckpt_dir_input:
      ckpt_dir = ckpt_dir_input

    load_model_directly = st_utils.checkbox('load_model_directly', cfg.load_model_directly, sidebar=True)
    return_aux_img = st_utils.checkbox('return_aux_img', cfg.return_aux_img, sidebar=True)

    st_utils.st_set_sep(msg='G_kwargs', sidebar=True)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    last_back = st_utils.checkbox('last_back', True, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', False, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    num_frames = st_utils.number_input('num_frames', cfg.num_frames, sidebar=True)

    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)
    max_fov = st_utils.number_input('max_fov', cfg.max_fov, sidebar=True)
    alpha_pidiv = st_utils.number_input('alpha_pidiv', cfg.alpha_pidiv, sidebar=True)
    periods = st_utils.number_input('periods', cfg.periods, sidebar=True)
    # seed
    seed = st_utils.get_seed(seeds=cfg.seeds)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)
    # trajectory
    trajectory_mode = st_utils.selectbox('trajectory_mode', cfg.trajectory_mode, sidebar=True)
    yaw_range = st_utils.parse_list_from_st_text_input('yaw_range', cfg.yaw_range, sidebar=True)
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    num_samples_translate = st_utils.number_input('num_samples_translate', 30, sidebar=True)
    translate_dist = st_utils.number_input('translate_dist', 0.04, sidebar=True)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = torch.device('cuda')

    print(f"Load from {ckpt_dir}")
    if load_model_directly:
      generator = torch.load(f"{ckpt_dir}/G_ema_model.pth")
    else:
      generator = build_model(cfg=cfg.G_cfg).to(device)
      Checkpointer(generator).load_state_dict_from_file(f"{ckpt_dir}/G_ema.pth")

    curriculum = cfg.G_kwargs.to_dict()
    curriculum['num_steps'] = num_steps
    curriculum['img_size'] = image_size
    curriculum['psi'] = psi
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['last_back'] = last_back
    curriculum['nerf_noise'] = 0

    up = None
    if trajectory_mode == 'yaw':
      xyz, lookup, yaws, pitchs = comm_utils.get_yaw_camera_pos_and_lookup(num_samples=num_frames, )
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    elif trajectory_mode == 'circle':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / alpha_pidiv,
                                                                              num_samples=num_frames,
                                                                              periods=periods)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    elif trajectory_mode == 'circle_near_far':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / alpha_pidiv,
                                                                              num_samples=num_frames,
                                                                              periods=2)
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
        alpha=math.pi / alpha_pidiv,
        num_samples=num_frames,
        periods=2)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * num_samples_translate * 2
      for idx, t in enumerate(np.linspace(0, 1, num_frames)):
        fov_list.append(fov + t * (max_fov - fov))
      fov_list.extend(fov_list[-num_frames:][::-1])
      assert len(fov_list) == len(xyz)
    elif trajectory_mode == 'translate_rotate':
      xyz, lookup, up, yaws, pitchs, num_samples_translate = comm_utils.get_translate_rotate(
        num_samples_translate=num_samples_translate,
        translate_dist=translate_dist,
        alpha=math.pi / alpha_pidiv,
        num_samples=num_frames,
        periods=1)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      up = torch.from_numpy(up).to(device)

      fov_list = [fov] * len(xyz)

    st_image = st.empty()
    # output_name = f'seed_{seed}.mp4'
    psi_str = f"{psi:.2f}".replace(".", "_")
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/seed_{seed}_psi_{psi_str}_inr_nerf.mp4", fps=fps,
                                           hd_video=hd_video)
    video_nerf_f = cv2_utils.ImageioVideoWriter(f"{outdir}/seed_{seed}_psi_{psi_str}_nerf.mp4", fps=fps,
                                                hd_video=hd_video)
    video_inr_f = cv2_utils.ImageioVideoWriter(f"{outdir}/seed_{seed}_psi_{psi_str}_inr.mp4", fps=fps,
                                               hd_video=hd_video)

    torch.manual_seed(seed)
    # z = torch.randn(1, generator.z_dim, device=device)
    zs = generator.get_zs(1)

    with torch.no_grad():
      for idx in tqdm.tqdm(range(len(xyz))):
        curriculum['h_mean'] = 0
        curriculum['v_mean'] = 0
        curriculum['h_stddev'] = 0
        curriculum['v_stddev'] = 0

        cur_camera_pos = xyz[[idx]]
        cur_camera_lookup = lookup[[idx]]
        if up is not None:
          up_vector = up[[idx]]
        else:
          up_vector = None

        yaw = yaws[idx]
        pitch = pitchs[idx]
        fov = fov_list[idx]
        curriculum['fov'] = fov

        if return_aux_img:
          frame, depth_map = generator.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=True,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            up_vector=up_vector,
            **curriculum)
          frame = (frame.squeeze() + 1) * 0.5

          img_inr, img_nerf = frame.unbind(0)
          img_inr_pil = trans_f.to_pil_image(img_inr)
          img_nerf_pil = trans_f.to_pil_image(img_nerf)
          video_inr_f.write(img_inr_pil, )
          video_nerf_f.write(img_nerf_pil)

          frame_tensor = make_grid(frame, nrow=2, normalize=True, scale_each=True)
          frame_pil = trans_f.to_pil_image(frame_tensor)
          pil_utils.add_text(frame_pil, f"pitch {pitch:.2f}, yaw {yaw:.2f}, fov {fov:.2f}",
                             size=frame_pil.size[0] // 18)
        else:
          frame, depth_map = generator.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=False,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            up_vector=up_vector,
            **curriculum)
          frame = (frame.squeeze() + 1) * 0.5
          frame_pil = trans_f.to_pil_image(frame)
          pil_utils.add_text(frame_pil, f"pitch: {pitch:.2f}\nyaw: {yaw:.2f}\nfov: {fov:.2f}",
                             size=frame_pil.size[0] // 15)

        st_utils.st_image(frame_pil, caption=f"{frame_pil.size}, seed={seed}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        video_f.write(frame_pil)
        if global_cfg.tl_debug:
          break

      video_f.release(st_video=True)
      if return_aux_img:
        video_inr_f.release(st_video=True)
        video_nerf_f.release(st_video=True)
    pass

  def render_image_web(self,
                       cfg,
                       outdir,
                       saved_suffix_state=None,
                       **kwargs):

    mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    lock_view_dependence = st_utils.checkbox('lock_view_dependence', cfg.lock_view_dependence, sidebar=True)
    ray_step_multiplier = st_utils.number_input('ray_step_multiplier', cfg.ray_step_multiplier, sidebar=True)
    seeds = st_utils.parse_list_from_st_text_input('seeds', cfg.seeds)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = torch.device('cuda')

    curriculum = getattr(curriculums, mode)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * ray_step_multiplier
    curriculum['img_size'] = image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    model_pkl = cfg.models_dict[mode]
    generator = torch.load(model_pkl, map_location=torch.device(device))
    ema_file = model_pkl.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    face_angles = [-0.5, -0.25, 0., 0.25, 0.5]

    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    img_pils = []
    for seed in tqdm.tqdm(seeds):
      images = []
      for i, yaw in enumerate(face_angles):
        curriculum['h_mean'] = yaw
        torch.manual_seed(seed)
        z = torch.randn((1, 256), device=device)
        img, tensor_img, depth_map = generate_img(generator, z, **curriculum)
        images.append(tensor_img)
      # save_image(torch.cat(images), os.path.join(outdir, f'grid_{seed}.png'), normalize=True)
      merged_tensor = make_grid(torch.cat(images), nrow=len(images), normalize=True, scale_each=True)
      merged_pil = trans_f.to_pil_image(merged_tensor)
      img_pils.append(merged_pil)
      if global_cfg.tl_debug:
        break
    img_pil = pil_utils.merge_image_pil(img_pils, nrow=1)
    st_utils.st_image(img_pil, caption=f"{img.shape}, seed: {seed}", debug=global_cfg.tl_debug, outdir=outdir)
    pass

  def render_layer_swapping_web(self,
                                cfg,
                                outdir,
                                saved_suffix_state=None,
                                **kwargs):

    # mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    network_pkl = st_utils.selectbox('network_pkl', cfg.network_pkl)
    target_network_pkl = st_utils.selectbox('target_network_pkl', cfg.network_pkl)
    model_pkl_input = st_utils.text_input('model_pkl', "", sidebar=False)

    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    last_back = st_utils.checkbox('last_back', True, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', False, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    num_frames = st_utils.number_input('num_frames', cfg.num_frames, sidebar=True)
    num_samples_translate = st_utils.number_input('num_samples_translate', 30, sidebar=True)
    translate_dist = st_utils.number_input('translate_dist', 0.04, sidebar=True)

    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)
    max_fov = st_utils.number_input('max_fov', cfg.max_fov, sidebar=True)
    alpha_pidiv = st_utils.number_input('alpha_pidiv', cfg.alpha_pidiv, sidebar=True)
    periods = st_utils.number_input('periods', cfg.periods, sidebar=True)
    # seed
    seed = st_utils.get_seed(cfg.seeds)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)
    psi_target = st_utils.number_input('psi_target', 1., sidebar=True)
    # trajectory
    trajectory_mode = st_utils.selectbox('trajectory_mode', cfg.trajectory_mode, sidebar=True)
    yaw_range = st_utils.parse_list_from_st_text_input('yaw_range', cfg.yaw_range, sidebar=True)
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    return_aux_img = st_utils.checkbox('return_aux_img', False, sidebar=True)

    swapped_layers = st_utils.parse_list_from_st_text_input('swapped_layers',
                                                            "[1024, 512, 256, 128, 64, 32, 16, 8, 4]")
    gamma_target = st_utils.number_input('gamma_target', 0.45, sidebar=True)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = torch.device('cuda')

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')
    generator = build_model(cfg=cfg.G_cfg).to(device)
    Checkpointer(generator).load_state_dict_from_file(model_pkl)

    target_G = copy.deepcopy(generator)
    _, target_model_pkl = target_network_pkl.split(':')
    target_model_pkl = target_model_pkl.strip(' ')
    if model_pkl_input:
      target_model_pkl = model_pkl_input
    Checkpointer(target_G).load_state_dict_from_file(target_model_pkl)

    swapped_G = copy.deepcopy(generator)

    comm_utils.inr_layer_swapping(swapped_net=swapped_G.inr_net, target_net=target_G.inr_net,
                                  gamma_target=gamma_target, swapped_layers=swapped_layers)

    metadata = f"{os.path.dirname(model_pkl)}/metadata.json"
    if os.path.exists(metadata):
      curriculum = comm_utils.get_metadata_from_json(metafile=metadata,
                                                     num_steps=num_steps,
                                                     image_size=image_size,
                                                     psi=psi,
                                                     last_back=last_back,
                                                     )

    else:
      curriculum = getattr(curriculums, mode)
      curriculum['num_steps'] = num_steps
      curriculum['img_size'] = image_size
      curriculum['psi'] = psi
      curriculum['v_stddev'] = 0
      curriculum['h_stddev'] = 0
      curriculum['last_back'] = last_back
      curriculum['nerf_noise'] = 0
      curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    if trajectory_mode == 'yaw':
      xyz, lookup, yaws, pitchs = comm_utils.get_yaw_camera_pos_and_lookup(num_samples=num_frames, )
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    elif trajectory_mode == 'circle':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / alpha_pidiv,
                                                                              num_samples=num_frames,
                                                                              periods=periods)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * len(xyz)

    elif trajectory_mode == 'circle_near_far':
      xyz, lookup, yaws, pitchs = comm_utils.get_circle_camera_pos_and_lookup(alpha=math.pi / alpha_pidiv,
                                                                              num_samples=num_frames,
                                                                              periods=2)
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
        alpha=math.pi / alpha_pidiv,
        num_samples=num_frames,
        periods=2)
      xyz = torch.from_numpy(xyz).to(device)
      lookup = torch.from_numpy(lookup).to(device)
      fov_list = [fov] * num_samples_translate * 2
      for idx, t in enumerate(np.linspace(0, 1, num_frames)):
        fov_list.append(fov + t * (max_fov - fov))
      fov_list.extend(fov_list[-num_frames:][::-1])
      assert len(fov_list) == len(xyz)

    st_image = st.empty()
    output_name = Path(f'seed_{seed}.mp4')
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name}", fps=fps, hd_video=hd_video)
    video_f_source = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name.stem}_source.mp4", fps=fps, hd_video=hd_video)
    video_f_target = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name.stem}_target.mp4", fps=fps, hd_video=hd_video)
    video_f_mixed = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name.stem}_mixed.mp4", fps=fps, hd_video=hd_video)

    torch.manual_seed(seed)
    # z = torch.randn(1, generator.z_dim, device=device)
    zs = generator.get_zs(1)

    with torch.no_grad():

      # frontal face
      curriculum['h_mean'] = math.pi * 0.5
      curriculum['v_mean'] = math.pi * 0.5
      curriculum['h_stddev'] = 0
      curriculum['v_stddev'] = 0
      curriculum['fov'] = fov

      frontal_faces = []
      # source image
      frame, _ = generator(zs=zs,
                           return_aux_img=False,
                           forward_points=forward_points ** 2,
                           **curriculum)
      frame_pil = comm_utils.to_pil(frame)
      video_f_source.write(frame_pil)
      frontal_faces.append(frame_pil)

      frame, _ = target_G(zs=zs,
                          return_aux_img=False,
                          forward_points=forward_points ** 2,
                          **curriculum)
      frame_pil = comm_utils.to_pil(frame)
      video_f_target.write(frame_pil)
      frontal_faces.append(frame_pil)

      frame, _ = swapped_G(zs=zs,
                           return_aux_img=False,
                           forward_points=forward_points ** 2,
                           **curriculum)
      frame_pil = comm_utils.to_pil(frame)
      video_f_mixed.write(frame_pil)
      frontal_faces.append(frame_pil)
      frontal_faces_pil = pil_utils.merge_image_pil(frontal_faces, nrow=3, )
      st.image(frontal_faces_pil, caption=f"{frame_pil.size}, seed: {seed}")

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

        if return_aux_img:
          frame, depth_map = generator.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=True,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            **curriculum)
          frame_tensor = make_grid(frame, nrow=2, normalize=True, scale_each=True)
          frame_pil = trans_f.to_pil_image(frame_tensor)
          pil_utils.add_text(frame_pil, f"pitch {pitch:.2f}, yaw {yaw:.2f}, fov {fov:.2f}",
                             size=frame_pil.size[0] // 18)
          merged_frame = frame_pil
        else:
          frame, depth_map = generator.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=False,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            **curriculum)
          frame_pil = comm_utils.to_pil(frame)
          video_f_source.write(frame_pil)
          pil_utils.add_text(frame_pil, f"pitch: {pitch:.2f}\nyaw: {yaw:.2f}\nfov: {fov:.2f}",
                             size=frame_pil.size[0] // 15)

          # target image
          frame_target, _ = target_G.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=False,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            **{**curriculum, 'psi': psi_target})
          frame_target_pil = comm_utils.to_pil(frame_target)
          video_f_target.write(frame_target_pil)
          pil_utils.add_text(frame_target_pil, f"pitch: {pitch:.2f}\nyaw: {yaw:.2f}\nfov: {fov:.2f}",
                             size=frame_target_pil.size[0] // 15)

          # swapped image
          frame_swapped, _ = swapped_G.forward_camera_pos_and_lookup(
            zs=zs,
            return_aux_img=False,
            forward_points=forward_points ** 2,
            camera_pos=cur_camera_pos,
            camera_lookup=cur_camera_lookup,
            **curriculum)
          frame_swapped_pil = comm_utils.to_pil(frame_swapped)
          video_f_mixed.write(frame_swapped_pil)
          pil_utils.add_text(frame_swapped_pil, f"pitch: {pitch:.2f}\nyaw: {yaw:.2f}\nfov: {fov:.2f}",
                             size=frame_swapped_pil.size[0] // 15)

          merged_frame = pil_utils.merge_image_pil([frame_pil, frame_target_pil, frame_swapped_pil], nrow=2, )
        st_utils.st_image(merged_frame, caption=f"{frame_pil.size}, seed={seed}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        video_f.write(merged_frame)
        if global_cfg.tl_debug:
          break

      video_f.release(st_video=True)
      video_f_source.release(st_video=True)
      video_f_target.release(st_video=True)
      video_f_mixed.release(st_video=True)
    pass

  def render_interpolate_layers_web(self,
                                    cfg,
                                    outdir,
                                    saved_suffix_state=None,
                                    **kwargs):

    # mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    network_pkl = st_utils.selectbox('network_pkl', cfg.network_pkl)
    target_network_pkl = st_utils.selectbox('target_network_pkl', cfg.network_pkl)
    model_pkl_input = st_utils.text_input('model_pkl', "", sidebar=False)

    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    last_back = st_utils.checkbox('last_back', True, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', False, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)
    fov = st_utils.number_input('fov', cfg.fov, sidebar=True)

    # seed
    seed = st_utils.get_seed(cfg.seeds)

    # trajectory
    forward_points = st_utils.number_input('forward_points', cfg.forward_points, sidebar=True)

    swapped_layers = st_utils.parse_list_from_st_text_input('swapped_layers',
                                                            "[1024, 512, 256, 128, 64, 32, 16, 8, 4]")
    num_interp = st_utils.number_input('num_interp', cfg.num_interp, sidebar=True)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = torch.device('cuda')

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')
    generator = build_model(cfg=cfg.G_cfg).to(device)
    Checkpointer(generator).load_state_dict_from_file(model_pkl)

    target_G = copy.deepcopy(generator)
    _, target_model_pkl = target_network_pkl.split(':')
    target_model_pkl = target_model_pkl.strip(' ')
    if model_pkl_input:
      target_model_pkl = model_pkl_input
    Checkpointer(target_G).load_state_dict_from_file(target_model_pkl)

    metadata = f"{os.path.dirname(model_pkl)}/metadata.json"
    if os.path.exists(metadata):
      curriculum = comm_utils.get_metadata_from_json(metafile=metadata,
                                                     num_steps=num_steps,
                                                     image_size=image_size,
                                                     psi=psi,
                                                     last_back=last_back,
                                                     )

    else:
      curriculum = getattr(curriculums, mode)
      curriculum['num_steps'] = num_steps
      curriculum['img_size'] = image_size
      curriculum['psi'] = psi
      curriculum['v_stddev'] = 0
      curriculum['h_stddev'] = 0
      curriculum['last_back'] = last_back
      curriculum['nerf_noise'] = 0
      curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    st_image = st.empty()
    output_name = Path(f'seed_{seed}.mp4')
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name}", fps=fps, hd_video=hd_video)

    torch.manual_seed(seed)
    # z = torch.randn(1, generator.z_dim, device=device)
    zs = generator.get_zs(1)

    with torch.no_grad():

      curriculum['h_mean'] = math.pi * 0.5
      curriculum['v_mean'] = math.pi * 0.5
      curriculum['h_stddev'] = 0
      curriculum['v_stddev'] = 0
      curriculum['fov'] = fov

      # source image
      frame, g_pos = generator(zs=zs,
                               return_aux_img=False,
                               forward_points=forward_points ** 2,
                               **curriculum)

      frame_pil = comm_utils.to_pil(frame)

      # target image
      frame_target, _ = target_G(zs=zs,
                                 return_aux_img=False,
                                 forward_points=forward_points ** 2,
                                 **curriculum)
      frame_target_pil = comm_utils.to_pil(frame_target)
      frame_source_target = pil_utils.merge_image_pil([frame_pil, frame_target_pil], nrow=2)

      pil_utils.add_text(frame_source_target,
                         f"pitch: {curriculum['v_mean']:.2f}\nyaw: {curriculum['h_mean']:.2f}\nfov: {fov:.2f}",
                         size=frame_target_pil.size[0] // 15)
      st_utils.st_image(frame_source_target, caption=f"target, seed: {seed}, {frame_target_pil.size}", debug=global_cfg.tl_debug)

      for cur_gamma_target in tqdm.tqdm(np.linspace(0, 1, num_interp)):
        # interp layers
        interp_G = copy.deepcopy(generator)
        comm_utils.inr_layer_swapping(swapped_net=interp_G.inr_net, target_net=target_G.inr_net,
                                      gamma_target=cur_gamma_target, swapped_layers=swapped_layers, verbose=False)

        frame_swapped, _ = interp_G(zs=zs,
                                    return_aux_img=False,
                                    forward_points=forward_points ** 2,
                                    **curriculum)
        frame_swapped_pil = comm_utils.to_pil(frame_swapped)

        st_utils.st_image(frame_swapped_pil, caption=f"{frame_swapped_pil.size}, seed={seed}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        video_f.write(frame_swapped_pil)
        if global_cfg.tl_debug:
          break

    video_f.release(st_video=True)

    pass

  def render_video_interpolation_web(self,
                                     cfg,
                                     outdir,
                                     saved_suffix_state=None,
                                     **kwargs):

    mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    lock_view_dependence = st_utils.checkbox('lock_view_dependence', cfg.lock_view_dependence, sidebar=True)
    ray_step_multiplier = st_utils.number_input('ray_step_multiplier', cfg.ray_step_multiplier, sidebar=True)
    seeds = st_utils.parse_list_from_st_text_input('seeds', cfg.seeds)
    trajectory_mode = st_utils.selectbox('trajectory_mode', cfg.trajectory_mode, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    psi = st_utils.number_input('psi', cfg.psi, sidebar=True)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    device = torch.device('cuda')

    curriculum = getattr(curriculums, mode)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * ray_step_multiplier
    curriculum['img_size'] = image_size
    curriculum['psi'] = psi
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['num_frames'] = cfg.num_frames
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    model_pkl = cfg.models_dict[mode]
    generator = torch.load(model_pkl, map_location=torch.device(device))
    ema_file = model_pkl.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    if trajectory_mode == 'front':
      trajectory = []
      for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi / 2
        yaw = 0.4 * np.sin(t * 2 * math.pi) + math.pi / 2

        fov = curriculum['fov'] + 5 + np.sin(t * 2 * math.pi) * 5

        trajectory.append((t, pitch, yaw, fov))
    elif trajectory_mode == 'orbit':
      trajectory = []
      for t in np.linspace(0, 1, curriculum['num_frames']):
        pitch = 0.2 * np.cos(t * 2 * math.pi) + math.pi / 4
        yaw = t * 2 * math.pi
        fov = curriculum['fov']

        trajectory.append((t, pitch, yaw, fov))

    output_name = f'interp.mp4'
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/{output_name}", fps=fps)
    st_image = st.empty()
    # writer = skvideo.io.FFmpegWriter(os.path.join(cfg.output_dir, output_name),
    #                                  outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    # print(seeds)

    for i, seed in enumerate(seeds):
      frames = []
      depths = []

      torch.manual_seed(seed)
      z_current = torch.randn(1, 256, device=device)

      torch.manual_seed(seeds[(i + 1) % len(seeds)])
      z_next = torch.randn(1, 256, device=device)

      frequencyInterpolator = FrequencyInterpolator(generator, z_current, z_next, psi=cfg.psi)

      with torch.no_grad():
        for t, pitch, yaw, fov in tqdm.tqdm(trajectory):
          curriculum['h_mean'] = yaw  # + 3.14/2
          curriculum['v_mean'] = pitch  # + 3.14/2
          curriculum['fov'] = fov
          curriculum['h_stddev'] = 0
          curriculum['v_stddev'] = 0
          frame, depth_map = generator.staged_forward_with_frequencies(*frequencyInterpolator.forward(t),
                                                                       max_batch_size=cfg.max_batch_size,
                                                                       depth_map=cfg.depth_map, **curriculum)
          #             frame, depth_map = generator.staged_forward(z, max_batch_size=opt.max_batch_size, depth_map=opt.depth_map, **curriculum)
          frame_pil = tensor_to_PIL(frame)
          video_f.write(frame_pil)
          st_image.image(frame_pil, caption=f"{frame_pil.size}, seed: {seed}-{seeds[(i + 1) % len(seeds)]}")

          # frames.append(frame_pil)
        # for frame in frames:
        #   writer.writeFrame(np.array(frame))

    # writer.close()
    video_f.release(st_video=True)

  def inverse_render_web(self,
                         cfg,
                         outdir,
                         saved_suffix_state=None,
                         **kwargs):

    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      header = f"{k}_s"
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=header, )
      image_list_kwargs[header]['image_path'] = image_path
    source_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)

    image_path = image_list_kwargs[source_k]['image_path']
    image_path = st_utils.text_input('image_path', image_path)
    if not image_path:
      image_path = image_list_kwargs[source_k]['image_path']

    img_pil = Image.open(image_path)
    st.image(img_pil, caption=f"{img_pil.size}")

    network_pkl = st_utils.selectbox('network_pkl', options=cfg.network_pkl, )
    # mode = st_utils.selectbox('curriculum', options=cfg.mode, index=0)
    image_size = st_utils.number_input('image_size', cfg.image_size, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    n_iterations = st_utils.number_input('n_iterations', cfg.n_iterations, sidebar=True)
    st_log_every = st_utils.number_input('st_log_every', cfg.st_log_every, sidebar=True)
    lock_view_dependence = st_utils.checkbox('lock_view_dependence', cfg.lock_view_dependence, sidebar=True)

    # ****************************************************************************
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    mode, model_pkl = network_pkl.split(':')
    model_pkl = model_pkl.strip(' ')

    projector = Projector()
    projector.project_image(
      outdir=outdir,
      model_pkl=model_pkl,
      image_path=image_path,
      image_size=image_size,
      n_iterations=n_iterations,
      max_batch_size=cfg.max_batch_size,
      num_frames=cfg.num_frames,
      fps=fps,
      lock_view_dependence=lock_view_dependence,
      st_web=True,
      st_log_every=st_log_every,
    )

    pass
