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
import torchvision.transforms.functional as trans_f
import torchvision.transforms as tv_transforms

from torch_ema import ExponentialMovingAverage

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.proj.fvcore import set_global_cfg
from tl2.modelarts import modelarts_utils
from tl2.proj.fvcore import build_model
from tl2.proj.stylegan2_ada import ada_utils
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2 import tl2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pil import pil_utils
from tl2.proj.GAN import eval_FID_given_imgdir

from exp.dev.nerf_inr import fid_evaluation
from exp.dev.nerf_inr import curriculums
from exp.pigan import datasets
from exp.comm import comm_model_utils
from exp.comm import diff_aug
import torch_fidelity
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity import calculate_metrics
from exp.comm import comm_utils

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
                                            forward_points=128**2,
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
                  img_size=128,):
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
    if rank == 0: pbar = tqdm(desc=f"generating images at {img_size}x{img_size}", total = num_imgs)
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


def train(rank,
          world_size,
          opt,
          cfg):
    # set_global_cfg(cfg)
    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
    if rank == 0:
        modelarts_utils.setup_tl_outdir_obs(global_cfg, unzip_code=False)
        modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
        modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

    logger = logging.getLogger('tl')
    torch.manual_seed(global_cfg.seed + rank)
    torch_utils.init_seeds(seed=global_cfg.seed + rank)


    # generator = build_model(cfg=global_cfg.G_cfg).to(device)
    # Checkpointer(generator).load_state_dict_from_file(global_cfg.model_pkl, rank=rank)

    # generator = torch.load(global_cfg.model_pkl)
    # generator = generator.to(device)
    generator = torch_utils.torch_load(global_cfg.model_pkl, rank=rank)
    generator.eval()

    fake_dir = f"{global_cfg.tl_outdir}/fake"
    if rank == 0:
      if os.path.exists(fake_dir):
        logger.info(f"Del fake dir: {fake_dir}")
        shutil.rmtree(fake_dir)
      os.makedirs(fake_dir, exist_ok=True)

    dist.barrier()

    metafile = f"{os.path.dirname(global_cfg.model_pkl)}/metadata.json"
    metadata = comm_utils.get_metadata_from_json(metafile, return_raw=True)
    metadata.update(global_cfg.metadata)

    pbar = range(global_cfg.num_eval_images // (world_size * metadata.batch_size))
    if rank == 0:
      pbar = tqdm(pbar, desc=f"Generating fake images [{world_size * metadata.batch_size * len(pbar)}]")
    for b in pbar:
        with torch.no_grad():
            zs = generator.get_zs(metadata.batch_size)
            g_imgs, g_pos = generator(zs=zs,
                                      return_aux_img=False,
                                      # grad_points=128**2,
                                      **{**metadata, 'forward_points': cfg.forward_points ** 2})
            g_imgs = (g_imgs + 1.) * 0.5
            for idx, img in enumerate(g_imgs):
                img_pil = trans_f.to_pil_image(img)
                save_path = f"{fake_dir}/rank{rank}_b{b}_idx{idx}.{global_cfg.get('ext', 'png')}"
                img_pil.save(save_path)
            if rank == 0 and b == 0:
              tqdm.write(f"image shape: {g_imgs.shape}")
            if global_cfg.tl_debug:
                pil_utils.imshow_pil(img_pil)
                break

    dist.barrier()
    if rank == 0:
      outdir = f"{cfg.tl_outdir}/exp"
      os.makedirs(outdir, exist_ok=True)
      eval_FID_given_imgdir.main(outdir=outdir, fake_dir=fake_dir, dataset_root=cfg.dataset_root,
                                 debug=global_cfg.tl_debug)

    if rank == 0:
      logger.info(f"Del fake images {fake_dir}")
      shutil.rmtree(fake_dir)
    dist.barrier()
    cleanup()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=60000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    update_parser_defaults_from_yaml(parser)

    modelarts_utils.setup_tl_outdir_obs(global_cfg, unzip_code=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt, global_cfg), nprocs=num_gpus, join=True)

    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
