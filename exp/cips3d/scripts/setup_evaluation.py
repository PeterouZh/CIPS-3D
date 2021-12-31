import tqdm
import shutil
from PIL import Image
import logging
import argparse
import os

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
import torchvision.transforms as tv_trans
from torchvision.utils import save_image

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import modelarts_utils, moxing_utils
from tl2.proj.pil import pil_utils
from tl2.proj.skimage import skimage_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.tl2_utils import AverageMeter
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model
from tl2.proj.pytorch.examples.multi_process_main.dataset import ImageListDataset
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils
from tl2.proj.fvcore import build_model
from tl2.proj.pytorch.examples.dataset_stylegan3.dataset import get_training_dataloader, to_norm_tensor


def setup_evaluation(rank,
                     world_size,
                     data_cfg,
                     real_dir,
                     img_size,
                     num_imgs,
                     del_fid_real_images):
  """
  Output real images.

  :param rank:
  :param world_size:
  :param data_cfg:
  :param real_dir:
  :param img_size:
  :param num_imgs:
  :param del_fid_real_images:
  :return:
  """
  logger = logging.getLogger('tl')

  if rank == 0 and del_fid_real_images:
    shutil.rmtree(real_dir, ignore_errors=True)
  ddp_utils.d2_synchronize()

  if os.path.exists(real_dir):
    logger.info("Real images exist.")
    return
  else:
    if rank == 0: os.makedirs(real_dir, exist_ok=True)
    ddp_utils.d2_synchronize()

    dataset = build_model(data_cfg, kwargs_priority=True, xflip=False, resize_resolution=img_size)
    num_imgs = min(num_imgs, len(dataset))

    batch_size = 64
    num_workers = 0
    shuffle = False
    data_loader = get_training_dataloader(dataset=dataset, rank=rank, num_gpus=world_size,
                                          batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    data_loader_iter = iter(data_loader)

    if rank == 0:
      pbar = tqdm.tqdm(desc=f"Output real images at {img_size}x{img_size}", total=num_imgs)
    for idx_b in range((num_imgs + batch_size - 1) // batch_size):
      if rank == 0:
        pbar.update(batch_size)
      real_imgs, _, imgs_idx = next(data_loader_iter)
      # imgs_norm = to_norm_tensor(imgs, device=device)
      for img, idx in zip(real_imgs, imgs_idx):
        saved_path = f"{real_dir}/{idx:0>5}.jpg"
        img_pil = pil_utils.np_to_pil(img.numpy(), channel_first=True)
        img_pil.save(saved_path)

      if global_cfg.tl_debug:
        break
  pass



def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser(description='Training configurations.')

  argparser_utils.add_argument_int(parser, name='local_rank', default=0)
  argparser_utils.add_argument_int(parser, name='seed', default=0)
  argparser_utils.add_argument_int(parser, name='num_workers', default=0)

  return parser


def main():

  parser = build_parser()
  args, _ = parser.parse_known_args()

  rank, world_size = ddp_utils.ddp_init(seed=args.seed)
  torch_utils.init_seeds(seed=args.seed, rank=rank)
  device = torch.device('cuda')

  is_main_process = (rank == 0)

  update_parser_defaults_from_yaml(parser, is_main_process=is_main_process)
  logger = logging.getLogger('tl')

  setup_evaluation(rank=rank,
                   world_size=world_size,
                   data_cfg=global_cfg.data_cfg,
                   real_dir=f"{global_cfg.tl_outdir}/exp/fid/real",
                   img_size=global_cfg.img_size,
                   num_imgs=global_cfg.num_images_real_eval,
                   del_fid_real_images=global_cfg.del_fid_real_images)
  global_cfg.del_fid_real_images = False
  pass


if __name__ == '__main__':
  main()
