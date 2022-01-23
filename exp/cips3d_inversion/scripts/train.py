from PIL import Image
import collections
import shutil
import traceback
from functools import partial
import pprint
import logging
import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import copy
from einops import repeat, rearrange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import modelarts_utils, moxing_utils
from tl2.proj.fvcore import build_model
from tl2.proj.logger.textlogger import summary_dict2txtfig, summary_defaultdict2txtfig, global_textlogger
from tl2 import tl2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch.examples.dataset_stylegan3.dataset import get_training_dataloader, to_norm_tensor
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.proj.pytorch.examples.nerf import cam_params

from exp.dev.nerf_inr import curriculums
from exp.pigan import datasets
from exp.comm import comm_model_utils
from exp.cips3d.scripts.setup_evaluation import setup_evaluation
from exp.cips3d_inversion.scripts.gen_images import gen_images
from exp.cips3d.scripts.eval_fid import eval_fid

def setup_ddp(rank, world_size, port):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = port

  # initialize the process group
  # dist.init_process_group("gloo", rank=rank, world_size=world_size)
  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  pass


def cleanup():
  dist.destroy_process_group()


def saved_models(model_dict,
                 info_msg,
                 G,
                 G_ema,
                 G_kwargs,
                 cam_param,
                 fixed_z,
                 img_size,
                 device,
                 saved_dir=None):
  if saved_dir is None:
    ckpt_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep(name='ckpt', use_circle_number=True)
    saved_dir = ckpt_max2keep.step_and_ret_circle_dir(global_cfg.tl_ckptdir)
  os.makedirs(saved_dir, exist_ok=True)

  global_cfg.dump_to_file_with_command(f"{saved_dir}/config_command.yaml", global_cfg.tl_command)

  torch_utils.save_models(save_dir=saved_dir, model_dict=model_dict)
  tl2_utils.write_info_msg(saved_dir, info_msg)

  save_images(saved_dir=saved_dir,
              G=G,
              G_ema=G_ema,
              cam_param=cam_param,
              G_kwargs=G_kwargs,
              fixed_z=fixed_z,
              img_size=img_size,
              device=device)

  torch.cuda.empty_cache()

  pass


@torch.no_grad()
def save_images(saved_dir,
                G,
                G_ema,
                cam_param,
                G_kwargs,
                fixed_z,
                img_size,
                device,
                use_amp_G=False,
                ):
  G.eval()
  G_ema.eval()

  bs = len(list(fixed_z.values())[0])
  G_kwargs = copy.deepcopy(G_kwargs)
  H = W = img_size

  # intr = cam_param.get_intrinsic(H, W)
  rays_o, rays_d = cam_param.get_rays_of_pose_avg(H=H, W=W, bs=bs)

  with torch.cuda.amp.autocast(use_amp_G):
    copied_metadata = copy.deepcopy(G_kwargs)

    Gz, ret_imgs = G(zs=fixed_z,
                     rays_o=rays_o,
                     rays_d=rays_d,
                     forward_points=256 ** 2,
                     return_aux_img=True,
                     **copied_metadata)
    Gz_aux = ret_imgs['aux_img']
    Gz = torch.cat([Gz, Gz_aux], dim=0)
    save_image(Gz, f"{saved_dir}/0Gz.jpg", nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)

    Gema_z, ret_imgs = G_ema(zs=fixed_z,
                             rays_o=rays_o,
                             rays_d=rays_d,
                             forward_points=256 ** 2,
                             return_aux_img=True,
                             **copied_metadata)
    Gema_z_aux = ret_imgs['aux_img']
    Gema_z = torch.cat([Gema_z, Gema_z_aux], dim=0)
    save_image(Gema_z, f"{saved_dir}/0Gz_ema.jpg", nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)

    copied_metadata['psi'] = 0.7
    Gema_trunc, ret_imgs = G_ema(zs=fixed_z,
                                 rays_o=rays_o,
                                 rays_d=rays_d,
                                 forward_points=256 ** 2,
                                 return_aux_img=True,
                                 **copied_metadata)
    Gema_trunc_aux = ret_imgs['aux_img']
    Gema_trunc = torch.cat([Gema_trunc, Gema_trunc_aux], dim=0)
    save_image(Gema_trunc, f"{saved_dir}/0G_trunc_ema.jpg",
               nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)

  # with torch.cuda.amp.autocast(use_amp_G):
  #   copied_metadata = copy.deepcopy(G_kwargs)
  #   copied_metadata['h_stddev'] = 0
  #   copied_metadata['v_stddev'] = 0
  #   copied_metadata['h_mean'] = math.pi * 0.5 + 0.5
  #   Gz_tilted = G(fixed_z, return_aux_img=True, forward_points=256 ** 2, **copied_metadata)[0]
  #   save_image(Gz_tilted, f"{saved_dir}/0Gz_tilted.jpg",
  #              nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)
  #
  #   Gema_z_tilted = G_ema(fixed_z, return_aux_img=True, forward_points=256 ** 2, **copied_metadata)[0]
  #   save_image(Gema_z_tilted, f"{saved_dir}/0Gz_tilted_ema.jpg",
  #              nrow=int(math.sqrt(bs)), normalize=True, scale_each=True)

  # Monitor mirror symmetry
  # bs = min(20, bs)
  # sub_fixed_z = {}
  # for name, z_ in fixed_z.items():
  #   sub_fixed_z[name] = z_[:bs]
  # fixed_z = sub_fixed_z
  #
  # with torch.cuda.amp.autocast(use_amp_G):
  #   copied_metadata = copy.deepcopy(G_kwargs)
  #   copied_metadata['h_stddev'] = 0
  #   copied_metadata['v_stddev'] = 0
  #   copied_metadata['h_mean'] = 1.44
  #   Gema_flip1 = G_ema(fixed_z, return_aux_img=True, forward_points=256 ** 2, **copied_metadata)[0]
  #
  #   copied_metadata['h_mean'] = 1.70
  #   Gema_flip2 = G_ema(fixed_z, return_aux_img=True, forward_points=256 ** 2, **copied_metadata)[0]
  #
  #   Gema_flip = torch.cat([Gema_flip1, Gema_flip2])
  #   save_image(Gema_flip, f"{saved_dir}/0G_flip_ema.jpg", nrow=bs//2, normalize=True, scale_each=True)

  pass


def get_curriculum(curriculum_name):
  curriculum = getattr(curriculums, curriculum_name)

  # update curriculum_step
  for curriculum_step in curriculum.keys():
    if type(curriculum_step) == int:
      # stage
      curriculum_step_str = str(curriculum_step)
      if curriculum_step_str in global_cfg.curriculum:
        curriculum[curriculum_step].update(global_cfg.curriculum.get(curriculum_step_str).to_dict())
    else:
      # update attrs
      if curriculum_step in global_cfg.curriculum:
        curriculum[curriculum_step] = global_cfg.curriculum[curriculum_step]
  for new_attr, value in global_cfg.curriculum.get('new_attrs', {}).items():
    assert new_attr not in curriculum
    curriculum[new_attr] = value
  return curriculum


def build_optimizer(generator_ddp,
                    discriminator_ddp,
                    cam_param_ddp):

  optimizer_G = torch.optim.Adam(
    params=[{'params': generator_ddp.parameters(),
             'initial_lr': global_cfg.gen_lr}],
    lr=global_cfg.gen_lr,
    betas=global_cfg.betas,
    weight_decay=0)
  optimizer_D = torch.optim.Adam(
    params=[{'params': discriminator_ddp.parameters(),
             'initial_lr': global_cfg.disc_lr}],
    lr=global_cfg.disc_lr,
    betas=global_cfg.betas,
    weight_decay=0)

  optimizer_cam = torch.optim.Adam(
    params=[{'params': cam_param_ddp.parameters(),
             'initial_lr': global_cfg.cam_lr}],
    lr=global_cfg.cam_lr,
    betas=global_cfg.betas,
    weight_decay=0)

  # if global_cfg.tl_resume and global_cfg.load_optimizers:
  #   model_dict = {
  #     'optimizer_G': optimizer_G,
  #     'optimizer_D': optimizer_D,
  #     'scaler_G': scaler_G,
  #     'scaler_D': scaler_D,
  #   }
  #   torch_utils.load_models(save_dir=resume_dir, model_dict=model_dict, strict=False, rank=rank)

  # After load optimizer.
  # optimizer_G.param_groups[0]['initial_lr'] = global_cfg.gen_lr
  # optimizer_G.param_groups[0]['lr'] = global_cfg.gen_lr
  # optimizer_G.param_groups[0]['betas'] = metadata['betas']
  # optimizer_D.param_groups[0]['initial_lr'] = global_cfg.disc_lr
  # optimizer_D.param_groups[0]['lr'] = global_cfg.disc_lr
  # optimizer_D.param_groups[0]['betas'] = metadata['betas']

  return optimizer_G, optimizer_D, optimizer_cam


def train(rank,
          world_size,
          opt):
  setup_ddp(rank, world_size, opt.port)

  update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
  if rank == 0:
    moxing_utils.setup_tl_outdir_obs(global_cfg)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

  logger = logging.getLogger('tl')
  device = torch.device(rank)
  torch_utils.init_seeds(seed=global_cfg.seed, rank=rank)

  # curriculum = get_curriculum(curriculum_name=opt.curriculum)

  scaler_G = torch.cuda.amp.GradScaler(enabled=global_cfg.use_amp_G)
  scaler_D = torch.cuda.amp.GradScaler(enabled=global_cfg.use_amp_D)

  """dataset """
  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_training_dataset)
  dataset = build_model(global_cfg.data_cfg, kwargs_priority=True, resize_resolution=global_cfg.img_size)
  # imgs, label, idx = dataset[0] # [0, 255]
  data_loader = get_training_dataloader(dataset=dataset, rank=rank, num_gpus=world_size,
                                        batch_size=global_cfg.batch_size * world_size,
                                        num_workers=global_cfg.num_workers,
                                        shuffle=True, sampler_seed=0)
  data_loader_iter = iter(data_loader)
  # batch_data = next(data_loader_iter)

  H = W = global_cfg.img_size
  cam_param = cam_params.CamParams.from_config(num_imgs=len(dataset), H0=H, W0=W).to(device)
  cam_param_ddp = DDP(cam_param, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)

  generator = build_model(cfg=global_cfg.G_cfg).to(device)
  discriminator = build_model(cfg=global_cfg.D_cfg, kwargs_priority=True, diffaug=global_cfg.diffaug).to(device)
  G_ema = copy.deepcopy(generator)
  ema_start_itr = global_cfg.get('ema_start_itr', 1000)
  ema_model = comm_model_utils.EMA(source=generator, target=G_ema, decay=0.999, start_itr=ema_start_itr)

  # ddp
  generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
  discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
  generator = generator_ddp.module
  generator.set_device(device)

  optimizer_G, optimizer_D, optimizer_cam = build_optimizer(generator_ddp=generator_ddp,
                                                            discriminator_ddp=discriminator_ddp,
                                                            cam_param_ddp=cam_param_ddp)

  state_dict = {
    'cur_fid': np.inf,
    'best_fid': np.inf,
    'worst_fid': 0,
    'step': 0,
  }

  model_dict = {
    'cam_param': cam_param_ddp.module,
    'generator': generator_ddp.module,
    'G_ema': G_ema,
    'discriminator': discriminator_ddp.module,
    # 'optimizer_G': optimizer_G,
    # 'optimizer_D': optimizer_D,
    'state_dict': state_dict,
  }

  if global_cfg.tl_resume or global_cfg.load_finetune:
    if global_cfg.tl_resume:
      resume_dir = f"{global_cfg.tl_resumedir}/ckptdir/resume"
      torch_utils.load_models(save_dir=resume_dir, model_dict=model_dict, strict=False, rank=rank)
    elif global_cfg.load_finetune:
      moxing_utils.copy_data(rank=rank, global_cfg=global_cfg,
                             datapath_obs=global_cfg.finetune_dir, datapath=global_cfg.finetune_dir)
      finetune_model_dict = {
        'cam_param': cam_param_ddp.module,
        'generator': generator_ddp.module,
        'G_ema': G_ema,
        'discriminator': discriminator_ddp.module,
      }
      torch_utils.load_models(save_dir=global_cfg.finetune_dir, model_dict=finetune_model_dict,
                              strict=False, rank=rank)
    else:
      assert 0

    if global_cfg.get('load_nerf_ema', False):
      generator.load_nerf_ema(G_ema)

    if global_cfg.load_G_ema:
      ema_model.update_target_dict(G_ema.state_dict())
    else:
      ema_model.update_target_dict(generator.state_dict())

    if global_cfg.reset_best_fid:
      state_dict['best_fid'] = np.inf
    logger.info(pprint.pformat(state_dict))

  if global_cfg.tl_debug:
    global_cfg.fixed_z_bs = 4
  fixed_z = generator.get_zs(global_cfg.fixed_z_bs)
  dummy_tensor = torch.tensor([0], device=device)
  # ----------
  #  Training
  # ----------

  start_itr = state_dict['step']
  pbar = tl2_utils.TL_tqdm(total=global_cfg.total_iters, start=start_itr)


  summary_ddict = collections.defaultdict(dict)

  nerf_noise_disable = global_cfg.get('nerf_noise_disable', False)

  G_kwargs = global_cfg.G_kwargs.to_dict()

  for step in range(start_itr, global_cfg.total_iters):
    pbar.update()
    summary_ddict.clear()

    # for i, (imgs, _) in enumerate(dataloader):
      # real_imgs = imgs.to(device, non_blocking=True)
    imgs, _, imgs_idx = next(data_loader_iter)
    real_imgs = to_norm_tensor(imgs, device=device)
    imgs_idx = imgs_idx.to(device)

    generator_ddp.train()
    discriminator_ddp.train()

    if nerf_noise_disable:
      nerf_noise = 0.
    else:
      nerf_noise = max(0, 1. - state_dict['step'] / 5000.)
    G_kwargs['nerf_kwargs']['raw_noise_std'] = nerf_noise

    if global_cfg.get('warmup_D', False):
      alpha = min(1, step / global_cfg.fade_steps)
    else:
      alpha = 1.

    '''TRAIN DISCRIMINATOR'''
    torch_utils.requires_grad(generator_ddp, False)
    torch_utils.requires_grad(discriminator_ddp, True)

    aux_reg = global_cfg.train_aux_img and (step % global_cfg.update_aux_every == 0)

    with torch.cuda.amp.autocast(global_cfg.use_amp_D):
      # Generate images for discriminator training
      with torch.no_grad():
        # z = z_sampler((real_imgs.shape[0], metadata['latent_dim']),
        #               device=device,
        #               dist=metadata['z_dist'])
        zs_list = generator.get_zs(real_imgs.shape[0], batch_split=global_cfg.batch_split)
        if global_cfg.batch_split == 1:
          zs_list = [zs_list]
        gen_imgs = []
        gen_imgs_aux = []
        if global_cfg.img_size >= 256 and global_cfg.forward_points is not None:
          forward_points = global_cfg.forward_points ** 2
        else:
          forward_points = None

        imgs_idx_list = imgs_idx.chunk(global_cfg.batch_split)
        for subset_z, sub_imgs_idx in zip(zs_list, imgs_idx_list):
          R, t, fx, fy = cam_param(sub_imgs_idx)
          rays_o, rays_d, _ = generator.get_rays_axis_angle(R=R, t=t, fx=fx, fy=fy, H=H, W=W, N_rays=-1)
          g_imgs, ret_imgs = generator_ddp(zs=subset_z,
                                           rays_o=rays_o,
                                           rays_d=rays_d,
                                           return_aux_img=aux_reg,
                                           forward_points=forward_points,
                                           grad_points=None,
                                           **G_kwargs)

          # find a bug when batch_split > 1 :  g_imgs: [Gz, Gz_aux]
          gen_imgs.append(g_imgs)
          if aux_reg:
            g_imgs_aux = ret_imgs['aux_img']
            gen_imgs_aux.append(g_imgs_aux)

        gen_imgs = torch.cat(gen_imgs + gen_imgs_aux, axis=0)
      # end torch.no_grad
      if aux_reg:
        real_imgs = torch.cat([real_imgs, real_imgs], dim=0)
      real_imgs.requires_grad_()
      r_preds, _, _ = discriminator_ddp(real_imgs, alpha=alpha, use_aux_disc=aux_reg, summary_ddict=summary_ddict)

    d_regularize = step % global_cfg.d_reg_every == 0

    if global_cfg.r1_lambda > 0 and d_regularize:
      # Gradient penalty
      grad_real = torch.autograd.grad(
        outputs=scaler_D.scale(r_preds.sum()),
        inputs=real_imgs,
        create_graph=True)
      inv_scale = 1. / scaler_D.get_scale()
      grad_real = [p * inv_scale for p in grad_real][0]

    with torch.cuda.amp.autocast(global_cfg.use_amp_D):
      if global_cfg.r1_lambda > 0 and d_regularize:
        # grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        # grad_penalty = 0.5 * global_cfg.r1_lambda * global_cfg.d_reg_every * grad_penalty + 0 * r_preds[0]
        grad_penalty = grad_real.flatten(start_dim=1).square().sum(dim=1, keepdim=True)
        grad_penalty = 0.5 * global_cfg.r1_lambda * grad_penalty * global_cfg.d_reg_every  + 0. * r_preds
      else:
        grad_penalty = dummy_tensor

      g_preds, _, _ = discriminator_ddp(gen_imgs, alpha=alpha, use_aux_disc=aux_reg)

      d_loss = (torch.nn.functional.softplus(g_preds) +
                torch.nn.functional.softplus(-r_preds) +
                grad_penalty).mean()

      if rank == 0:
        with torch.no_grad():
          summary_ddict['D_logits']['D_logits_real'] = r_preds.mean().item()
          summary_ddict['D_logits']['D_logits_fake'] = g_preds.mean().item()
          summary_ddict['grad_penalty']['grad_penalty'] = grad_penalty.mean().item()

    optimizer_D.zero_grad()
    scaler_D.scale(d_loss).backward()
    scaler_D.unscale_(optimizer_D)
    try:
      D_total_norm = torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(),
                                                    global_cfg.grad_clip,
                                                    # error_if_nonfinite=True, # torch >= 1.9
                                                    )
      summary_ddict['D_total_norm']['D_total_norm'] = D_total_norm.item()
    except:
      summary_ddict['D_total_norm']['D_total_norm'] = np.nan
      logger.info(traceback.format_exc())
      saved_models(model_dict=model_dict,
                   info_msg=f"step: {state_dict['step']}",
                   G=generator, G_ema=G_ema, G_kwargs=G_kwargs,
                   fixed_z=fixed_z, img_size=global_cfg.img_size,
                   saved_dir=f"{global_cfg.tl_ckptdir}/D_crupted")
      # exit(0)
      optimizer_D.zero_grad()

    scaler_D.step(optimizer_D)
    scaler_D.update()

    ''' TRAIN GENERATOR '''
    torch_utils.requires_grad(generator_ddp, True)
    torch_utils.requires_grad(discriminator_ddp, False)

    # z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
    zs_list = generator.get_zs(imgs.shape[0], batch_split=global_cfg.batch_split)
    if global_cfg.batch_split == 1:
      zs_list = [zs_list]

    if global_cfg.grad_points is not None:
      grad_points = global_cfg.grad_points ** 2
    else:
      grad_points = None

    imgs_idx_list = imgs_idx.chunk(global_cfg.batch_split)
    for subset_z, sub_imgs_idx in zip(zs_list, imgs_idx_list):
      with torch.cuda.amp.autocast(global_cfg.use_amp_G):
        R, t, fx, fy = cam_param(sub_imgs_idx)
        rays_o, rays_d, _ = generator.get_rays_axis_angle(R=R, t=t, fx=fx, fy=fy, H=H, W=W, N_rays=-1)
        gen_imgs, ret_imgs = generator_ddp(zs=subset_z,
                                           rays_o=rays_o,
                                           rays_d=rays_d,
                                           return_aux_img=aux_reg,
                                           grad_points=grad_points,
                                           forward_points=None,
                                           **G_kwargs)
        if aux_reg:
          g_imgs_aux = ret_imgs['aux_img']
          gen_imgs = torch.cat([gen_imgs, g_imgs_aux], dim=0)

        with torch.cuda.amp.autocast(global_cfg.use_amp_D):
          g_preds, _, _ = discriminator_ddp(gen_imgs.to(torch.float32), alpha=alpha, use_aux_disc=aux_reg)
        g_loss = torch.nn.functional.softplus(-g_preds).mean()
      scaler_G.scale(g_loss).backward()
    # end accumulate gradients
    scaler_G.unscale_(optimizer_G)
    scaler_G.unscale_(optimizer_cam)
    try:
      G_total_norm = torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(),
                                                    global_cfg.grad_clip,
                                                    # metadata.get('grad_clip', 0.3),
                                                    # error_if_nonfinite=True, # torch >= 1.9
                                                    )
      summary_ddict['G_total_norm']['G_total_norm'] = G_total_norm.item()
    except:
      summary_ddict['G_total_norm']['G_total_norm'] = np.nan
      logger.info(traceback.format_exc())
      saved_models(model_dict=model_dict,
                   info_msg=f"step: {state_dict['step']}",
                   G=generator, G_ema=G_ema, G_kwargs=G_kwargs,
                   fixed_z=fixed_z, img_size=global_cfg.img_size,
                   saved_dir=f"{global_cfg.tl_ckptdir}/G_crupted")
      # exit(0)
      optimizer_G.zero_grad()
      optimizer_cam.zero_grad()
    scaler_G.step(optimizer_G)
    scaler_G.step(optimizer_cam)
    scaler_G.update()

    optimizer_G.zero_grad()
    optimizer_cam.zero_grad()

    # update ema
    ema_model.update(itr=state_dict['step'], source_dict=generator_ddp.module.state_dict())

    if (rank == 0 and (step + 1) % global_cfg.log_every == 0) or global_cfg.tl_debug:
      summary_ddict['lr']['G_lr'] = torch_utils.get_optimizer_lr(optimizer_G)
      summary_ddict['lr']['D_lr'] = torch_utils.get_optimizer_lr(optimizer_D)
      summary_ddict['lr']['cam_lr'] = torch_utils.get_optimizer_lr(optimizer_cam)
      summary_ddict['img_size']['img_size'] = global_cfg.img_size
      summary_ddict['batch_size']['batch_size'] = global_cfg.batch_size * world_size
      summary_ddict['grad_points']['grad_points'] = grad_points if grad_points is not None else global_cfg.img_size ** 2
      summary_ddict['scaler']['scaler_G'] = scaler_G.get_scale()
      summary_ddict['scaler']['scaler_D'] = scaler_D.get_scale()
      summary_ddict['r1_lambda']['r1_lambda'] = global_cfg.r1_lambda
      summary_ddict['grad_clip']['grad_clip'] = global_cfg.grad_clip
      summary_ddict['nerf_noise']['nerf_noise'] = nerf_noise
      summary_ddict['train_aux_img']['train_aux_img'] = int(global_cfg.train_aux_img)
      summary_ddict['alpha']['alpha'] = alpha
      summary_ddict['diffaug']['diffaug'] = int(global_cfg.diffaug)
      if step > 1000:
        summary_defaultdict2txtfig(summary_ddict, prefix='train', step=state_dict['step'], textlogger=global_textlogger)
      summary_str = tl2_utils.get_print_dict_str(summary_ddict, outdir=global_cfg.tl_outdir,
                                                 suffix_str=pbar.get_string())
      print(summary_str)


    state_dict['step'] += 1
    if step == 0 or (step + 1) % global_cfg.eval_every == 0 or global_cfg.tl_debug:
      # output real images
      setup_evaluation(rank=rank,
                       world_size=world_size,
                       data_cfg=global_cfg.data_cfg,
                       real_dir=f"{global_cfg.tl_outdir}/exp/fid/real",
                       img_size=global_cfg.img_size,
                       num_imgs=global_cfg.num_images_real_eval,
                       del_fid_real_images=global_cfg.del_fid_real_images,
                       shuffle=True)
      global_cfg.del_fid_real_images = False
      ddp_utils.d2_synchronize()

      # output fake images
      gen_images(rank=rank,
                 world_size=world_size,
                 generator=G_ema,
                 cam_param=cam_param,
                 G_kwargs=global_cfg.G_kwargs.to_dict(),
                 fake_dir=f"{global_cfg.tl_outdir}/exp/fid/fake",
                 num_imgs=global_cfg.num_images_gen_eval,
                 img_size=global_cfg.img_size,
                 batch_size=global_cfg.eval_batch_size,
                 device=device)
      ddp_utils.d2_synchronize()

      moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_inception_v3)
      global_cfg.obs_inception_v3.disable = True
      if rank == 0:
        metric_dict = eval_fid(real_dir=f"{global_cfg.tl_outdir}/exp/fid/real",
                               fake_dir=f"{global_cfg.tl_outdir}/exp/fid/fake")
        logger.info(f"\nstep: {state_dict['step']}, {pprint.pformat(metric_dict)}\n")
        summary_dict2txtfig(metric_dict, prefix='eval', step=state_dict['step'], textlogger=global_textlogger)
        state_dict['cur_fid'] = metric_dict['FID']

        if state_dict['best_fid'] > metric_dict['FID']:
          state_dict['best_fid'] = metric_dict['FID']
          saved_models(model_dict=model_dict,
                       info_msg=f"step: {state_dict['step']}\n"
                                f"cur_fid: {state_dict['cur_fid']}\n"
                                f"best_fid: {state_dict['best_fid']}",
                       G=generator, G_ema=G_ema, G_kwargs=global_cfg.G_kwargs.to_dict(),
                       cam_param=cam_param,
                       fixed_z=fixed_z, img_size=global_cfg.img_size,
                       saved_dir=f"{global_cfg.tl_ckptdir}/best_fid",
                       device=device)

        info_msg = f"step: {state_dict['step']}\n" \
                   f"cur_fid: {state_dict['cur_fid']}\n" \
                   f"best_fid: {state_dict['best_fid']}"
        # backup
        saved_models(model_dict=model_dict,
                     info_msg=info_msg,
                     G=generator, G_ema=G_ema, G_kwargs=global_cfg.G_kwargs.to_dict(),
                     cam_param=cam_param,
                     fixed_z=fixed_z, img_size=global_cfg.img_size,
                     device=device)
        # resume
        saved_models(model_dict=model_dict,
                     info_msg=info_msg,
                     G=generator, G_ema=G_ema, G_kwargs=global_cfg.G_kwargs.to_dict(),
                     cam_param=cam_param,
                     fixed_z=fixed_z, img_size=global_cfg.img_size,
                     saved_dir=f"{global_cfg.tl_ckptdir}/resume",
                     device=device)

        moxing_utils.modelarts_sync_results_dir(cfg=global_cfg, join=False)
        # end rank == 0
      ddp_utils.d2_synchronize()

  cleanup()
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_epochs", type=int, default=60000, help="number of epochs of training")
  parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
  parser.add_argument('--output_dir', type=str, default='results/')
  parser.add_argument('--load_dir', type=str, default='')
  parser.add_argument('--curriculum', type=str, default='CelebA')
  parser.add_argument('--eval_freq', type=int, default=5000)
  parser.add_argument('--port', type=str, default='12355')
  parser.add_argument('--set_step', type=int, default=None)
  parser.add_argument('--model_save_interval', type=int, default=5000)

  argparser_utils.add_argument_bool(parser, 'modelarts', default=False)

  update_parser_defaults_from_yaml(parser)

  opt, _ = parser.parse_known_args()
  argparser_utils.print_args(opt)

  moxing_utils.setup_tl_outdir_obs(global_cfg, unzip_code=False)
  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

  num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  if num_gpus > 1:
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
  else:
    train(rank=0, world_size=num_gpus, opt=opt)

  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
