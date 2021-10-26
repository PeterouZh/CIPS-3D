import pprint
import logging
import argparse
import os
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
import copy
from collections import deque
from einops import repeat, rearrange

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image, make_grid

from torch_ema import ExponentialMovingAverage

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.proj.fvcore import set_global_cfg
from tl2.modelarts import modelarts_utils
from tl2.proj.fvcore import build_model
from tl2.proj.stylegan2_ada import ada_utils
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2 import tl2_utils
from tl2.proj.pytorch import torch_utils

from generators import generators
from discriminators import discriminators
from siren import siren
# import fid_evaluation
from exp.dev.nerf_inr import fid_evaluation
# import curriculums
from exp.dev.nerf_inr import curriculums
# import datasets
from exp.pigan import datasets


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def load_images(images, curriculum, device):
    return_images = []
    head = 0
    for stage in curriculum['stages']:
        stage_images = images[head:head + stage['batch_size']]
        stage_images = F.interpolate(stage_images, size=stage['img_size'],  mode='bilinear', align_corners=True)
        return_images.append(stage_images)
        head += stage['batch_size']
    return return_images


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def saved_models(ema,
                 ema2,
                 generator_ddp,
                 discriminator_ddp,
                 optimizer_G,
                 optimizer_D,
                 scaler,
                 state_dict,
                 step,
                 epoch,
                 saved_dir=None,
                 ):
    if saved_dir is None:
        ckpt_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep(
            name='ckpt', max_to_keep=4, use_circle_number=True)
        saved_dir = ckpt_max2keep.step_and_ret_circle_dir(
            global_cfg.tl_ckptdir, info_msg=step)
    os.makedirs(saved_dir, exist_ok=True)

    G_ema = copy.deepcopy(generator_ddp.module)
    ema.copy_to(G_ema.parameters())

    state_dict['epoch'] = epoch
    state_dict['step'] = step
    model_dict = {
        'ema': ema,
        'ema2': ema2,
        'generator': generator_ddp.module,
        'G_ema': G_ema,
        'discriminator': discriminator_ddp.module,
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'scaler': scaler,
        'state_dict': state_dict,
    }

    torch_utils.save_models(save_dir=saved_dir,
                            model_dict=model_dict,
                            info_msg=f"epoch: {epoch}\n"
                                     f"step: {step}")

    del G_ema
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
                                generator_ddp,
                                fixed_z,
                                device,
                                step):
    fixed_z = fixed_z[:20]
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['h_mean'] = 1.44
            # copied_metadata['img_size'] = 128
            if copied_metadata['img_size'] > 128:
                fixed_z = fixed_z[0:10]
            gen_imgs1 = generator_ddp.module(fixed_z.to(device),
                                             return_aux_img=True,
                                             forward_points=128 ** 2,
                                             **copied_metadata)[0]

            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['h_mean'] = 1.70
            # copied_metadata['img_size'] = 128
            gen_imgs2 = generator_ddp.module(fixed_z.to(device),
                                             return_aux_img=True,
                                             forward_points=128 ** 2,
                                             **copied_metadata)[0]

    gen_imgs = torch.cat([gen_imgs1, gen_imgs2])
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_flip_ema.jpg"),
               nrow=len(fixed_z) // 2,
               normalize=True)
    pass

def save_images(generator_ddp,
                ema,
                discriminator,
                metadata,
                fixed_z,
                device,
                ):
    generator_ddp.eval()
    fixed_z_ori = fixed_z

    img_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep('img')
    saved_imgdir = img_max2keep.step_and_ret_circle_dir(root_dir=global_cfg.tl_imgdir,
                                                        info_msg=discriminator.step)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
            if copied_metadata['img_size'] > 128:
                fixed_z = fixed_z[0:9]
            gen_imgs = generator_ddp.module(fixed_z.to(device),
                                            return_aux_img=True,
                                            forward_points=128**2,
                                            **copied_metadata)[0]
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_fixed.jpg"),
               nrow=int(math.sqrt(len(fixed_z))),
               normalize=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['h_mean'] += 0.5
            # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
            gen_imgs = generator_ddp.module(fixed_z.to(device),
                                            return_aux_img=True,
                                            forward_points=128 ** 2,
                                            **copied_metadata)[0]
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_tilted.jpg"),
               nrow=int(math.sqrt(len(fixed_z))),
               normalize=True)

    ema.store(generator_ddp.parameters())
    ema.copy_to(generator_ddp.parameters())
    generator_ddp.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
            gen_imgs = generator_ddp.module(fixed_z.to(device),
                                            return_aux_img=True,
                                            forward_points=128 ** 2,
                                            **copied_metadata)[0]
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_fixed_ema.jpg"),
               nrow=int(math.sqrt(len(fixed_z))),
               normalize=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            copied_metadata['h_mean'] += 0.5
            # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
            gen_imgs = generator_ddp.module(fixed_z.to(device),
                                            return_aux_img=True,
                                            forward_points=128 ** 2,
                                            **copied_metadata)[0]
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_tilted_ema.jpg"),
               nrow=int(math.sqrt(len(fixed_z))),
               normalize=True)

    save_images_horizontal_flip(saved_imgdir=saved_imgdir,
                                metadata=metadata,
                                generator_ddp=generator_ddp,
                                fixed_z=fixed_z_ori,
                                device=device,
                                step=discriminator.step)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            copied_metadata = copy.deepcopy(metadata)
            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
            # copied_metadata['img_size'] = max(128, copied_metadata['img_size'])
            copied_metadata['psi'] = 0.7
            gen_imgs = generator_ddp.module(fixed_z.to(device),
                                            return_aux_img=True,
                                            forward_points=128 ** 2,
                                            **copied_metadata)[0]
    save_image(gen_imgs,
               os.path.join(saved_imgdir, f"gen_random.jpg"),
               nrow=int(math.sqrt(len(fixed_z))),
               normalize=True)

    ema.restore(generator_ddp.parameters())
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


def train(rank,
          world_size,
          opt,
          cfg):
    # set_global_cfg(cfg)
    update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
    if rank == 0:
        modelarts_utils.setup_tl_outdir_obs(global_cfg)
        modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
        modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

    logger = logging.getLogger('tl')
    torch.manual_seed(global_cfg.seed + rank)
    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    curriculum = get_curriculum(curriculum_name=opt.curriculum)

    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((global_cfg.get('fixed_z_bs', 25), metadata['latent_dim']),
                        device='cpu', dist=metadata['z_dist'])

    SIREN = getattr(siren, metadata['model'])
    CHANNELS = 3

    scaler = torch.cuda.amp.GradScaler()
    state_dict = {
        'best_fid': np.inf,
    }

    # if opt.load_dir != '':
    #     generator = torch.load(os.path.join(opt.load_dir, 'generator.pth'), map_location=device)
    #     discriminator = torch.load(os.path.join(opt.load_dir, 'discriminator.pth'), map_location=device)
    #     ema = torch.load(os.path.join(opt.load_dir, 'ema.pth'), map_location=device)
    #     ema2 = torch.load(os.path.join(opt.load_dir, 'ema2.pth'), map_location=device)
    # else:
    if 'G_cfg' in global_cfg:
        generator = build_model(cfg=global_cfg.G_cfg).to(device)
        # ada_utils.save_model(generator, snapshot_pkl=f"/home/z50017127/.keras/G.pkl")
    else:
        generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim']).to(device)
    if 'D_cfg' in global_cfg:
        discriminator = build_model(cfg=global_cfg.D_cfg).to(device)
    else:
        discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)

    if global_cfg.tl_resume:
        resume_dir = f"{global_cfg.tl_resumedir}/ckptdir/resume"
        model_dict = {
            'ema': ema,
            'ema2': ema2,
            'generator': generator,
            'discriminator': discriminator,
            'state_dict': state_dict,
        }
        torch_utils.load_models(save_dir=resume_dir, model_dict=model_dict, strict=False, rank=rank)
        logger.info(pprint.pformat(state_dict))
        discriminator.step = state_dict['step']
        generator.step = state_dict['step']
        discriminator.epoch = state_dict['epoch']
        generator.epoch = state_dict['epoch']

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
        optimizer_G = torch.optim.Adam(
            [
                {'params': generator_parameters,
                 'name': 'generator'},
                {'params': mapping_network_parameters,
                 'name': 'mapping_network',
                 'lr': metadata['gen_lr']*5e-2}
            ],
            lr=metadata['gen_lr'],
            betas=metadata['betas'],
            weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(
            generator_ddp.parameters(),
            lr=metadata['gen_lr'],
            betas=metadata['betas'],
            weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(
        discriminator_ddp.parameters(),
        lr=metadata['disc_lr'],
        betas=metadata['betas'],
        weight_decay=metadata['weight_decay'])

    if global_cfg.tl_resume:
        # optimizer_G.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_G.pth')))
        # optimizer_D.load_state_dict(torch.load(os.path.join(opt.load_dir, 'optimizer_D.pth')))
        # if not metadata.get('disable_scaler', False):
        #     scaler.load_state_dict(torch.load(os.path.join(opt.load_dir, 'scaler.pth')))
        model_dict = {
            'optimizer_G': optimizer_G,
            'optimizer_D': optimizer_D,
            'scaler': scaler,
        }
        torch_utils.load_models(save_dir=resume_dir, model_dict=model_dict, strict=False, rank=rank)

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Epoch", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    for _ in range (discriminator.epoch, opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        # Set learning rates
        for param_group in optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = metadata['gen_lr'] * 5e-2
            else:
                param_group['lr'] = metadata['gen_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = metadata['disc_lr']
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']

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

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']),
                                  device=device,
                                  dist=metadata['z_dist'])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    if metadata['img_size'] >= 256 and global_cfg.forward_points is not None:
                        forward_points = global_cfg.forward_points ** 2
                    else:
                        forward_points = None
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        g_imgs, g_pos = generator_ddp(subset_z,
                                                      return_aux_img=global_cfg.train_aux_img,
                                                      forward_points=forward_points,
                                                      grad_points=None,
                                                      **metadata)

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()),
                                                inputs=real_imgs,
                                                create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    if metadata['z_lambda'] > 0:
                        latent_penalty = torch.nn.functional.mse_loss(
                            g_pred_latent, z) * metadata['z_lambda']
                    else:
                        latent_penalty = 0
                    position_penalty = torch.nn.functional.mse_loss(
                        g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + \
                         torch.nn.functional.softplus(-r_preds).mean() + \
                         grad_penalty + \
                         identity_penalty
                # discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)

            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])

            split_batch_size = z.shape[0] // metadata['batch_split']

            if global_cfg.grad_points is not None:
                grad_points = global_cfg.grad_points ** 2
            else:
                grad_points = None
            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    gen_imgs, gen_positions = generator_ddp(subset_z,
                                                            return_aux_img=global_cfg.train_aux_img,
                                                            grad_points=grad_points,
                                                            forward_points=None,
                                                            **metadata)
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    if g_preds.shape[0] == z.shape[0] * 2:
                        aux_mask = torch.tensor(global_cfg.main_aux_mask, device=g_preds.device)
                        aux_mask = repeat(aux_mask, "b -> b r", r=z.shape[0])
                        aux_mask = rearrange(aux_mask, "b r -> (b r) 1")
                        g_preds = g_preds * aux_mask

                    if metadata['topk_v'] > 0:
                        topk_percentage = \
                            max(
                                0.99 ** (discriminator.step/metadata['topk_interval']),
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

                scaler.scale(g_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())

            if rank == 0:
                interior_step_bar.update(1)
                if i % 10 == 0:
                    tqdm.write(f"[Experiment: {opt.output_dir}] "
                               f"[GPU: {os.environ['CUDA_VISIBLE_DEVICES']}] "
                               f"[Epoch: {discriminator.epoch}/{opt.n_epochs}] "
                               f"[D loss: {d_loss.item()}] "
                               f"[G loss: {g_loss.item()}] "
                               f"[Step: {discriminator.step}] "
                               f"[Alpha: {alpha:.2f}] "
                               f"[Img Size: {metadata['img_size']}] "
                               f"[grad_points: {grad_points}]"
                               f"[Batch Size: {metadata['batch_size']}] "
                               f"[TopK: {topk_num}] "
                               f"[Scale: {scaler.get_scale()}]"
                               f"[grad_lambda: {global_cfg.G_cfg.grad_lambda}]")

                if global_cfg.tl_debug or \
                      (i % global_cfg.get('log_every', 100) == 0) and i > 50:
                    summary_dict = {
                        'd_loss': d_loss.item(),
                        'g_loss': g_loss.item(),
                        'alpha': alpha,
                        'img_size': metadata['img_size'],
                        'grad_points': grad_points if grad_points is not None else metadata['img_size'] ** 2,
                        'batch_size': metadata['batch_size'],
                        'topk_num': topk_num,
                        'scale': scaler.get_scale(),
                        'grad_lambda': global_cfg.G_cfg.grad_lambda,
                    }
                    summary_dict2txtfig(summary_dict, prefix="train", step=discriminator.step,
                                        textlogger=global_textlogger)

                if discriminator.step % global_cfg.get('log_img_every', 500) == 0:
                    save_images(generator_ddp=generator_ddp,
                                ema=ema,
                                discriminator=discriminator,
                                metadata=metadata,
                                fixed_z=fixed_z,
                                device=device)

            if global_cfg.tl_debug or (discriminator.step) % global_cfg.get('eval_every', 5000) == 0:
                if rank == 0:
                    saved_models(ema=ema,
                                 ema2=ema2,
                                 generator_ddp=generator_ddp,
                                 discriminator_ddp=discriminator_ddp,
                                 optimizer_G=optimizer_G,
                                 optimizer_D=optimizer_D,
                                 scaler=scaler,
                                 state_dict=state_dict,
                                 step=discriminator.step,
                                 epoch=discriminator.epoch)
                    saved_models(ema=ema,
                                 ema2=ema2,
                                 generator_ddp=generator_ddp,
                                 discriminator_ddp=discriminator_ddp,
                                 optimizer_G=optimizer_G,
                                 optimizer_D=optimizer_D,
                                 scaler=scaler,
                                 state_dict=state_dict,
                                 step=discriminator.step,
                                 epoch=discriminator.epoch,
                                 saved_dir=f"{global_cfg.tl_ckptdir}/resume")
                    # output real images
                    fid_evaluation.setup_evaluation(dataset_name=metadata['dataset'],
                                                    real_dir=f"{opt.output_dir}/fid/real",
                                                    target_size=min(metadata['img_size'], 256),
                                                    num_imgs=global_cfg.get('num_images_real_eval', 8000),
                                                    horizontal_flip=False)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                # output fake images
                fid_evaluation.output_images(generator=generator_ddp,
                                             input_metadata=metadata,
                                             rank=rank,
                                             world_size=world_size,
                                             fake_dir=f"{opt.output_dir}/fid/fake",
                                             num_imgs=global_cfg.get('num_images_gen_eval', 2048),
                                             img_size=min(metadata['img_size'], 256),)
                ema.restore(generator_ddp.parameters())
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

                    if state_dict['best_fid'] > fid:
                        state_dict['best_fid'] = fid
                        saved_models(ema=ema,
                                     ema2=ema2,
                                     generator_ddp=generator_ddp,
                                     discriminator_ddp=discriminator_ddp,
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     scaler=scaler,
                                     state_dict=state_dict,
                                     step=discriminator.step,
                                     epoch=discriminator.epoch,
                                     saved_dir=f"{global_cfg.tl_ckptdir}/best_fid")

                    modelarts_utils.modelarts_sync_results_dir(cfg=global_cfg, join=False)
                dist.barrier()
                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    update_parser_defaults_from_yaml(parser)

    modelarts_utils.setup_tl_outdir_obs(global_cfg)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt, global_cfg), nprocs=num_gpus, join=True)

    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
