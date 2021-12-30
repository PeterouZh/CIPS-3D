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
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2 import tl2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.argparser import argparser_utils

# from generators import generators
# from discriminators import discriminators
# from siren import siren
from exp.dev.nerf_inr import fid_evaluation
from exp.dev.nerf_inr import curriculums
from exp.pigan import datasets
from exp.comm import comm_model_utils
from exp.comm import diff_aug


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


def load_images(images, curriculum, device):
  return_images = []
  head = 0
  for stage in curriculum['stages']:
    stage_images = images[head:head + stage['batch_size']]
    stage_images = F.interpolate(stage_images, size=stage['img_size'], mode='bilinear', align_corners=True)
    return_images.append(stage_images)
    head += stage['batch_size']
  return return_images


def z_sampler(shape, device, dist):
  if dist == 'gaussian':
    z = torch.randn(shape, device=device)
  elif dist == 'uniform':
    z = torch.rand(shape, device=device) * 2 - 1
  return z


def saved_models(G_ema,
                 generator_ddp,
                 discriminator_ddp,
                 optimizer_G,
                 optimizer_D,
                 scaler_G,
                 scaler_D,
                 state_dict,
                 step,
                 epoch,
                 saved_dir=None,
                 metadata=None,
                 ):
  if saved_dir is None:
    ckpt_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep(
      name='ckpt', max_to_keep=4, use_circle_number=True)
    saved_dir = ckpt_max2keep.step_and_ret_circle_dir(
      global_cfg.tl_ckptdir, info_msg=step)
  os.makedirs(saved_dir, exist_ok=True)

  if metadata is not None:
    tl2_utils.json_dump(metadata, f"{saved_dir}/metadata.json")
  global_cfg.dump_to_file_with_command(
    f"{saved_dir}/config_command.yaml", global_cfg.tl_command)

  # G_ema = copy.deepcopy(generator_ddp.module)
  # ema.copy_to(G_ema.parameters())

  state_dict['epoch'] = epoch
  state_dict['step'] = step
  model_dict = {
    # 'ema': ema,
    # 'ema2': ema2,
    'generator': generator_ddp.module,
    'G_ema': G_ema,
    'discriminator': discriminator_ddp.module,
    'optimizer_G': optimizer_G,
    'optimizer_D': optimizer_D,
    'scaler_G': scaler_G,
    'scaler_D': scaler_D,
    'state_dict': state_dict,
  }

  torch_utils.save_models(save_dir=saved_dir,
                          model_dict=model_dict,
                          info_msg=f"epoch: {epoch}\n"
                                   f"step: {step}")

  # del G_ema
  torch.cuda.empty_cache()
  # torch.save(ema, os.path.join(saved_dir, 'ema.pth'))
  # torch.save(ema2, os.path.join(saved_dir, 'ema2.pth'))
  # torch.save(generator_ddp.module, os.path.join(saved_dir, 'generator.pth'))
  # torch.save(discriminator_ddp.module, os.path.join(saved_dir, 'discriminator.pth'))
  # torch.save(optimizer_G.state_dict(), os.path.join(saved_dir, 'optimizer_G.pth'))
  # torch.save(optimizer_D.state_dict(), os.path.join(saved_dir, 'optimizer_D.pth'))
  # torch.save(scaler.state_dict(), os.path.join(saved_dir, 'scaler.pth'))
  # torch.save(generator_losses, os.path.join(saved_dir, 'generator.losses'))
  # torch.save(discriminator_losses, os.path.join(saved_dir, 'discriminator.losses'))
  # torch.save(state_dict, f"{saved_dir}/state_dict.pth")

  pass


def save_images_horizontal_flip(saved_imgdir,
                                metadata,
                                G_ema,
                                fixed_z,
                                device,
                                step):
  # fixed_z = fixed_z[:20]
  sub_fixed_z = {}
  for name, z_ in fixed_z.items():
    sub_fixed_z[name] = z_[:20]
  fixed_z = sub_fixed_z

  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      copied_metadata['h_mean'] = 1.44
      # copied_metadata['img_size'] = 128
      if copied_metadata['img_size'] > 128:
        # fixed_z = fixed_z[0:10]
        sub_fixed_z = {}
        for name, z_ in fixed_z.items():
          sub_fixed_z[name] = z_[:10]
        fixed_z = sub_fixed_z
      gen_imgs1 = G_ema(fixed_z,
                        return_aux_img=True,
                        forward_points=128 ** 2,
                        **copied_metadata)[0]

      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      copied_metadata['h_mean'] = 1.70
      # copied_metadata['img_size'] = 128
      gen_imgs2 = G_ema(fixed_z,
                        return_aux_img=True,
                        forward_points=128 ** 2,
                        **copied_metadata)[0]

  gen_imgs = torch.cat([gen_imgs1, gen_imgs2])
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_flip_ema.jpg"),
             nrow=len(list(fixed_z.values())[0]) // 2,
             normalize=True)
  pass


def save_images(generator_ddp,
                G_ema,
                discriminator,
                metadata,
                fixed_z,
                device,
                ):
  generator_ddp.eval()
  fixed_z_ori = fixed_z
  bs = len(list(fixed_z.values())[0])

  img_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep('img')
  saved_imgdir = img_max2keep.step_and_ret_circle_dir(root_dir=global_cfg.tl_imgdir,
                                                      info_msg=discriminator.step)
  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
      if copied_metadata['img_size'] > 128:
        # fixed_z = fixed_z[0:9]
        sub_fixed_z = {}
        for name, z_ in fixed_z_ori.items():
          sub_fixed_z[name] = z_[0:9]
        fixed_z = sub_fixed_z
        bs = 9
      gen_imgs = generator_ddp.module(fixed_z,
                                      return_aux_img=True,
                                      forward_points=128 ** 2,
                                      **copied_metadata)[0]
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_fixed.jpg"),
             nrow=int(math.sqrt(bs)),
             normalize=True)

  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      copied_metadata['h_mean'] += 0.5
      # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
      gen_imgs = generator_ddp.module(fixed_z,
                                      return_aux_img=True,
                                      forward_points=128 ** 2,
                                      **copied_metadata)[0]
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_tilted.jpg"),
             nrow=int(math.sqrt(bs)),
             normalize=True)

  # ema.store(generator_ddp.parameters())
  # ema.copy_to(generator_ddp.parameters())
  # generator_ddp.eval()
  G_ema.eval()

  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
      gen_imgs = G_ema(fixed_z,
                       return_aux_img=True,
                       forward_points=128 ** 2,
                       **copied_metadata)[0]
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_fixed_ema.jpg"),
             nrow=int(math.sqrt(bs)),
             normalize=True)

  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      copied_metadata['h_mean'] += 0.5
      # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
      gen_imgs = G_ema(fixed_z,
                       return_aux_img=True,
                       forward_points=128 ** 2,
                       **copied_metadata)[0]
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_tilted_ema.jpg"),
             nrow=int(math.sqrt(bs)),
             normalize=True)

  save_images_horizontal_flip(saved_imgdir=saved_imgdir,
                              metadata=metadata,
                              G_ema=G_ema,
                              fixed_z=fixed_z_ori,
                              device=device,
                              step=discriminator.step)

  with torch.no_grad():
    with torch.cuda.amp.autocast(global_cfg.use_amp_G):
      copied_metadata = copy.deepcopy(metadata)
      copied_metadata['num_steps'] = max(12, copied_metadata['num_steps'])
      copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
      # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
      copied_metadata['psi'] = 0.7
      gen_imgs = G_ema(fixed_z,
                       return_aux_img=True,
                       forward_points=128 ** 2,
                       **copied_metadata)[0]
  save_image(gen_imgs,
             os.path.join(saved_imgdir, f"gen_random.jpg"),
             nrow=int(math.sqrt(bs)),
             normalize=True)

  # ema.restore(generator_ddp.parameters())
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


def output_images(generator,
                  input_metadata,
                  rank,
                  world_size,
                  fake_dir,
                  num_imgs=2048,
                  img_size=128, ):
  os.makedirs(fake_dir, exist_ok=True)
  metadata = copy.deepcopy(input_metadata)
  metadata['num_steps'] = max(12, metadata['num_steps'])
  metadata['img_size'] = img_size
  metadata['batch_size'] = 1

  metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
  metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
  metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
  metadata['psi'] = 1

  generator.eval()

  img_counter = rank
  if rank == 0: pbar = tqdm(desc=f"generating images at {img_size}x{img_size}", total=num_imgs)
  with torch.no_grad():
    while img_counter < num_imgs:
      # z = torch.randn((metadata['batch_size'], generator.z_dim), device=generator.device)
      zs = generator.get_zs(metadata['batch_size'])
      # generated_imgs, _ = generator.module.staged_forward(z, **metadata)
      generated_imgs = generator(zs,
                                 forward_points=256 ** 2,
                                 **metadata)[0]

      for img in generated_imgs:
        save_image(img,
                   os.path.join(fake_dir, f'{img_counter:0>5}.jpg'),
                   normalize=True,
                   value_range=(-1, 1))
        img_counter += world_size
        if rank == 0: pbar.update(world_size)
  if rank == 0: pbar.close()
  pass


def lambda_foo(itr,
               start_itrs,
               warmup_itrs,
               mul_lr=1):
  """

  :param itr: itr >= start_itrs
  :param start_itrs:
  :param warmup_itrs:
  :param mul_lr:
  :return:
  """
  if itr < start_itrs:
    return mul_lr
  else:
    if itr < start_itrs + warmup_itrs:
      return (itr - start_itrs) / warmup_itrs * mul_lr
    else:
      return mul_lr


def build_optimizer(generator_ddp,
                    discriminator_ddp):

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

  return optimizer_G, optimizer_D

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

  curriculum = get_curriculum(curriculum_name=opt.curriculum)

  metadata = curriculums.extract_metadata(curriculum, 0)

  scaler_G = torch.cuda.amp.GradScaler(enabled=global_cfg.use_amp_G)
  scaler_D = torch.cuda.amp.GradScaler(enabled=global_cfg.use_amp_D)

  state_dict = {
    'best_fid': np.inf,
    'worst_fid': 0,
    'step': 0,
    'epoch': 0,
  }

  generator = build_model(cfg=global_cfg.G_cfg).to(device)
  discriminator = build_model(cfg=global_cfg.D_cfg).to(device)
  G_ema = copy.deepcopy(generator)
  ema_model = comm_model_utils.EMA(source=generator, target=G_ema, decay=0.999, start_itr=1000)

  resume_dir = f"{global_cfg.tl_resumedir}/ckptdir/resume"

  if global_cfg.tl_resume:

    model_dict = {
      'generator': generator,
      'discriminator': discriminator,
      'state_dict': state_dict,
    }
    if global_cfg.load_G_ema:
      model_dict['G_ema'] = G_ema
    torch_utils.load_models(save_dir=resume_dir, model_dict=model_dict, strict=False, rank=rank)
    if global_cfg.load_G_ema:
      ema_model.update_target_dict(G_ema.state_dict())
    else:
      ema_model.update_target_dict(generator.state_dict())

    if global_cfg.reset_best_fid:
      state_dict['best_fid'] = np.inf
    logger.info(pprint.pformat(state_dict))

  # ddp
  generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
  discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
  generator = generator_ddp.module
  discriminator = discriminator_ddp.module

  optimizer_G, optimizer_D = build_optimizer(generator_ddp=generator_ddp, discriminator_ddp=discriminator_ddp)

  start_itr = discriminator.step

  generator.set_device(device)

  fixed_z = generator.get_zs(global_cfg.fixed_z_bs)
  use_diffaug = global_cfg.get('use_diffaug', False)
  # ----------
  #  Training
  # ----------

  dummy_tensor = torch.tensor([0], device=device)

  dataloader = None
  total_progress_bar = tqdm(total=opt.n_epochs, desc="Epoch", dynamic_ncols=True)
  total_progress_bar.update(discriminator.epoch)
  interior_step_bar = tqdm(dynamic_ncols=True)
  for _ in range(discriminator.epoch, opt.n_epochs):
    total_progress_bar.update(1)

    metadata = curriculums.extract_metadata(curriculum, discriminator.step)

    if not dataloader or dataloader.batch_size != metadata['batch_size']:
      dataloader, CHANNELS = datasets.get_dataset_distributed(
        name=metadata['dataset'], world_size=world_size, rank=rank, num_workers=global_cfg.num_workers,
        **metadata)

      step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
      step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)

      interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
      interior_step_bar.set_description(f"Stage {step_last_upsample}-{step_next_upsample}")
      interior_step_bar.update((discriminator.step - step_last_upsample))

    for i, (imgs, _) in enumerate(dataloader):
      metadata = curriculums.extract_metadata(curriculum, discriminator.step)

      if dataloader.batch_size != metadata['batch_size']: break

      generator_ddp.train()
      discriminator_ddp.train()

      # alpha = min(0.9, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))
      max_alpha = 1.
      if global_cfg.D_first_layer_warmup:
        alpha = min(max_alpha, (discriminator.step - start_itr) / (metadata['fade_steps']))
      else:
        alpha = max_alpha

      real_imgs = imgs.to(device, non_blocking=True)

      metadata['nerf_noise'] = max(0, 1. - discriminator.step / 5000.)

      # TRAIN DISCRIMINATOR
      torch_utils.requires_grad(generator_ddp, False)
      torch_utils.requires_grad(discriminator_ddp, True)

      aux_reg = global_cfg.train_aux_img and (i % global_cfg.update_aux_every == 0)

      with torch.cuda.amp.autocast(global_cfg.use_amp_D):
        # Generate images for discriminator training
        with torch.no_grad():
          # z = z_sampler((real_imgs.shape[0], metadata['latent_dim']),
          #               device=device,
          #               dist=metadata['z_dist'])
          zs = generator.get_zs(real_imgs.shape[0])
          split_batch_size = real_imgs.shape[0] // metadata['batch_split']
          gen_imgs = []
          gen_positions = []
          if metadata['img_size'] >= 256 and global_cfg.forward_points is not None:
            forward_points = global_cfg.forward_points ** 2
          else:
            forward_points = None
          for split in range(metadata['batch_split']):
            # subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
            subset_z = {}
            for name, z_ in zs.items():
              subset_z[name] = z_[split * split_batch_size:(split + 1) * split_batch_size]
            g_imgs, g_pos = generator_ddp(subset_z,
                                          return_aux_img=aux_reg,
                                          forward_points=forward_points,
                                          grad_points=None,
                                          **metadata)

            gen_imgs.append(g_imgs)
            gen_positions.append(g_pos)

          gen_imgs = torch.cat(gen_imgs, axis=0)
          if use_diffaug:
            gen_imgs = diff_aug.DiffAugment(gen_imgs)
          gen_positions = torch.cat(gen_positions, axis=0)

        # real_imgs.requires_grad = True
        if aux_reg:
          real_imgs = torch.cat([real_imgs, real_imgs], dim=0)
        if use_diffaug:
          real_imgs = diff_aug.DiffAugment(real_imgs)
        real_imgs.requires_grad_()
        r_preds, _, _ = discriminator_ddp(real_imgs, alpha, use_aux_disc=aux_reg, **metadata)

      d_regularize = i % global_cfg.d_reg_every == 0

      if metadata['r1_lambda'] > 0 and d_regularize:
        # Gradient penalty
        grad_real = torch.autograd.grad(
          outputs=scaler_D.scale(r_preds.sum()),
          inputs=real_imgs,
          create_graph=True)
        inv_scale = 1. / scaler_D.get_scale()
        grad_real = [p * inv_scale for p in grad_real][0]
      with torch.cuda.amp.autocast(global_cfg.use_amp_D):
        if metadata['r1_lambda'] > 0 and d_regularize:
          grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
          grad_penalty = 0.5 * metadata['r1_lambda'] * global_cfg.d_reg_every * grad_penalty + 0 * r_preds[0]
        else:
          grad_penalty = dummy_tensor

        g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
          gen_imgs, alpha, use_aux_disc=aux_reg, **metadata)
        if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
          if metadata['z_lambda'] > 0:
            latent_penalty = torch.nn.functional.mse_loss(
              g_pred_latent, zs) * metadata['z_lambda']
          else:
            latent_penalty = 0
          position_penalty = torch.nn.functional.mse_loss(
            g_pred_position, gen_positions) * metadata['pos_lambda']
          identity_penalty = latent_penalty + position_penalty
        else:
          identity_penalty = 0

        d_loss = torch.nn.functional.softplus(g_preds).mean() + \
                 torch.nn.functional.softplus(-r_preds).mean() + \
                 grad_penalty + \
                 identity_penalty
        # discriminator_losses.append(d_loss.item())

      optimizer_D.zero_grad()
      scaler_D.scale(d_loss).backward()
      scaler_D.unscale_(optimizer_D)
      try:
        D_total_norm = torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(),
                                                      metadata['grad_clip'],
                                                      # error_if_nonfinite=True, # torch >= 1.9
                                                      )
      except:
        logger.info(traceback.format_exc())
        saved_models(G_ema=G_ema,
                     generator_ddp=generator_ddp,
                     discriminator_ddp=discriminator_ddp,
                     optimizer_G=optimizer_G,
                     optimizer_D=optimizer_D,
                     scaler_G=scaler_G,
                     scaler_D=scaler_D,
                     state_dict=state_dict,
                     step=discriminator.step,
                     epoch=discriminator.epoch,
                     saved_dir=f"{global_cfg.tl_ckptdir}/D_crupted")
        # exit(0)
        optimizer_D.zero_grad()

      scaler_D.step(optimizer_D)
      scaler_D.update()

      # TRAIN GENERATOR
      if global_cfg.gen_lr > 0.:
        torch_utils.requires_grad(generator_ddp, True)
        torch_utils.requires_grad(discriminator_ddp, False)

        # z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
        zs = generator.get_zs(imgs.shape[0])

        split_batch_size = imgs.shape[0] // metadata['batch_split']

        if global_cfg.grad_points is not None:
          grad_points = global_cfg.grad_points ** 2
        else:
          grad_points = None
        for split in range(metadata['batch_split']):
          with torch.cuda.amp.autocast(global_cfg.use_amp_G):
            # subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
            subset_z = {}
            for name, z_ in zs.items():
              subset_z[name] = z_[split * split_batch_size:(split + 1) * split_batch_size]
            gen_imgs, gen_positions = generator_ddp(subset_z,
                                                    return_aux_img=aux_reg,
                                                    grad_points=grad_points,
                                                    forward_points=None,
                                                    **metadata)
            if use_diffaug:
              gen_imgs = diff_aug.DiffAugment(gen_imgs)
            with torch.cuda.amp.autocast(global_cfg.use_amp_D):
              g_preds, g_pred_latent, g_pred_position = discriminator_ddp(
                gen_imgs.to(torch.float32), alpha, use_aux_disc=aux_reg, **metadata)

            if g_preds.shape[0] == imgs.shape[0] * 2:
              aux_mask = torch.tensor(global_cfg.main_aux_mask, device=g_preds.device)
              aux_mask = repeat(aux_mask, "b -> b r", r=imgs.shape[0])
              aux_mask = rearrange(aux_mask, "b r -> (b r) 1")
              g_preds = g_preds * aux_mask

            if metadata['topk_v'] > 0:
              topk_percentage = \
                max(
                  0.99 ** (discriminator.step / metadata['topk_interval']),
                  metadata['topk_v']
                ) if 'topk_interval' in metadata and 'topk_v' in metadata \
                  else 1
              topk_num = math.ceil(topk_percentage * g_preds.shape[0])
              g_preds = torch.topk(g_preds, topk_num, dim=0).values
            else:
              topk_num = g_preds.shape[0]

            if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
              if metadata['z_lambda'] > 0:
                latent_penalty = torch.nn.functional.mse_loss(
                  g_pred_latent, subset_z) * metadata['z_lambda']
              else:
                latent_penalty = 0
              position_penalty = torch.nn.functional.mse_loss(
                g_pred_position, gen_positions) * metadata['pos_lambda']
              identity_penalty = latent_penalty + position_penalty
            else:
              identity_penalty = 0

            g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
            # generator_losses.append(g_loss.item())

          scaler_G.scale(g_loss).backward()

        scaler_G.unscale_(optimizer_G)
        try:
          G_total_norm = torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(),
                                                        metadata.get('grad_clip', 0.3),
                                                        # error_if_nonfinite=True, # torch >= 1.9
                                                        )
        except:
          logger.info(traceback.format_exc())
          saved_models(G_ema=G_ema,
                       generator_ddp=generator_ddp,
                       discriminator_ddp=discriminator_ddp,
                       optimizer_G=optimizer_G,
                       optimizer_D=optimizer_D,
                       scaler_G=scaler_G,
                       scaler_D=scaler_D,
                       state_dict=state_dict,
                       step=discriminator.step,
                       epoch=discriminator.epoch,
                       saved_dir=f"{global_cfg.tl_ckptdir}/G_crupted",
                       metadata=metadata)
          # exit(0)
          optimizer_G.zero_grad()
        scaler_G.step(optimizer_G)
        scaler_G.update()
        optimizer_G.zero_grad()
        # ema.update(generator_ddp.parameters())
        # ema2.update(generator_ddp.parameters())
        ema_model.update(itr=discriminator.step, source_dict=generator_ddp.module.state_dict())
      else:
        grad_points = None
        G_total_norm = dummy_tensor
        g_loss = dummy_tensor
        topk_num = 0

      if rank == 0:
        interior_step_bar.update(1)
        if i % 10 == 0:
          tqdm.write(f"[Experiment: {opt.output_dir}] "
                     f"[GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] "
                     f"[Epoch: {discriminator.epoch}/{opt.n_epochs}] "
                     f"[D loss: {d_loss.item():.3f}] "
                     f"[G loss: {g_loss.item():.3f}] "
                     f"[grad_penalty: {grad_penalty.item():.3f}] "
                     f"[Step: {discriminator.step}] "
                     f"[G_lr: {optimizer_G.param_groups[0]['lr']:.8f}] "
                     f"[D_lr: {optimizer_D.param_groups[0]['lr']:.8f}] "
                     f"[Img Size: {metadata['img_size']}] "
                     f"[grad_points: {grad_points}]"
                     f"[Batch Size: {metadata['batch_size']}] "
                     f"[TopK: {topk_num}] "
                     f"[scaler_G: {scaler_G.get_scale()}] "
                     f"[scaler_D: {scaler_D.get_scale()}] "
                     f"[r1_lambda: {metadata['r1_lambda']}] "
                     f"[alpha({metadata['fade_steps']}): {alpha}] ")

        if global_cfg.tl_debug or \
              (discriminator.step % global_cfg.get('log_every', 100) == 0) and discriminator.step > 1000:
          summary_dict = {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'grad_penalty': grad_penalty.item(),
            'G_lr': optimizer_G.param_groups[0]['lr'],
            'D_lr': optimizer_D.param_groups[0]['lr'],
            'img_size': metadata['img_size'],
            'grad_points': grad_points if grad_points is not None else metadata['img_size'] ** 2,
            'batch_size': metadata['batch_size'],
            'topk_num': topk_num,
            'scaler_G': scaler_G.get_scale(),
            'scaler_D': scaler_D.get_scale(),
            'r1_lambda': metadata['r1_lambda'],
            'alpha': alpha,
            'grad_clip': metadata['grad_clip'],
            'D_total_norm': D_total_norm.item(),
            'G_total_norm': G_total_norm.item(),
          }
          summary_dict2txtfig(summary_dict, prefix="train", step=discriminator.step,
                              textlogger=global_textlogger)

        if discriminator.step % global_cfg.get('log_img_every', 500) == 0:
          save_images(generator_ddp=generator_ddp,
                      G_ema=G_ema,
                      discriminator=discriminator,
                      metadata=metadata,
                      fixed_z=fixed_z,
                      device=device)

      if global_cfg.tl_debug or (discriminator.step) % global_cfg.get('eval_every', 5000) == 0:
        if rank == 0:
          saved_models(G_ema=G_ema,
                       generator_ddp=generator_ddp,
                       discriminator_ddp=discriminator_ddp,
                       optimizer_G=optimizer_G,
                       optimizer_D=optimizer_D,
                       scaler_G=scaler_G,
                       scaler_D=scaler_D,
                       state_dict=state_dict,
                       step=discriminator.step,
                       epoch=discriminator.epoch,
                       metadata=metadata)
          saved_models(G_ema=G_ema,
                       generator_ddp=generator_ddp,
                       discriminator_ddp=discriminator_ddp,
                       optimizer_G=optimizer_G,
                       optimizer_D=optimizer_D,
                       scaler_G=scaler_G,
                       scaler_D=scaler_D,
                       state_dict=state_dict,
                       step=discriminator.step,
                       epoch=discriminator.epoch,
                       saved_dir=f"{global_cfg.tl_ckptdir}/resume",
                       metadata=metadata)
          # output real images
          if global_cfg.get('del_fid_real_images', False):
            if os.path.exists(f"{opt.output_dir}/fid/real"):
              shutil.rmtree(f"{opt.output_dir}/fid/real")
          fid_evaluation.setup_evaluation(dataset_name=metadata['dataset'],
                                          image_list_file=metadata['image_list_file'],
                                          real_dir=f"{opt.output_dir}/fid/real",
                                          target_size=min(metadata['img_size'], 128),
                                          num_imgs=global_cfg.get('num_images_real_eval', 8000),
                                          horizontal_flip=False)
        dist.barrier()
        # ema.store(generator_ddp.parameters())
        # ema.copy_to(generator_ddp.parameters())
        generator_ddp.eval()
        # output fake images
        output_images(generator=G_ema,
                      input_metadata=metadata,
                      rank=rank,
                      world_size=world_size,
                      fake_dir=f"{opt.output_dir}/fid/fake",
                      num_imgs=global_cfg.get('num_images_gen_eval', 2048),
                      img_size=min(metadata['img_size'], 128))
        # ema.restore(generator_ddp.parameters())
        dist.barrier()
        if rank == 0:
          fid = fid_evaluation.calculate_fid(real_dir=f"{opt.output_dir}/fid/real",
                                             fake_dir=f"{opt.output_dir}/fid/fake",
                                             batch_size=128)
          logger.info(f"\nstep: {discriminator.step}, fid: {fid}\n")
          summary_dict = {
            'fid': fid
          }
          summary_dict2txtfig(summary_dict, prefix='eval', step=discriminator.step,
                              textlogger=global_textlogger)

          best_dir = f"{global_cfg.tl_ckptdir}/best_fid"
          if not os.path.exists(best_dir):
            state_dict = {'best_fid': np.inf}
          if state_dict['best_fid'] > fid:
            state_dict['best_fid'] = fid
            saved_models(G_ema=G_ema,
                         generator_ddp=generator_ddp,
                         discriminator_ddp=discriminator_ddp,
                         optimizer_G=optimizer_G,
                         optimizer_D=optimizer_D,
                         scaler_G=scaler_G,
                         scaler_D=scaler_D,
                         state_dict=state_dict,
                         step=discriminator.step,
                         epoch=discriminator.epoch,
                         saved_dir=best_dir,
                         metadata=metadata)
          if opt.modelarts:
            modelarts_utils.modelarts_sync_results_dir(cfg=global_cfg, join=False)
        dist.barrier()
        torch.cuda.empty_cache()

      discriminator.step += 1
      generator.step += 1
    discriminator.epoch += 1
    generator.epoch += 1

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
  opt.output_dir = f"{global_cfg.tl_outdir}/exp"
  argparser_utils.print_args(opt)

  moxing_utils.setup_tl_outdir_obs(global_cfg, unzip_code=False)
  # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
  # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

  os.makedirs(opt.output_dir, exist_ok=True)
  num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  if num_gpus > 1:
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
  else:
    train(rank=0, world_size=num_gpus, opt=opt)

  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
