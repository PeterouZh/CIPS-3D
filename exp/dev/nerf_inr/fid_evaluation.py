"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""
import logging
import os
import shutil
import torch
import copy
import argparse
from tqdm import tqdm
from PIL import Image

from torchvision.utils import save_image
from torch_fidelity import calculate_metrics
from pytorch_fid import fid_score

from tl2 import tl2_utils

# import curriculums
from exp.dev.nerf_inr import curriculums
from exp.pigan import datasets


def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    pbar = tqdm(desc=f"Output real images", total=num_imgs)
    for i in range(num_imgs//batch_size):
        try:
          real_imgs, _ = next(dataloader)
        except StopIteration:
          break

        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.jpg'), normalize=True, value_range=(-1, 1))
            img_counter += 1
        pbar.update(batch_size)

def setup_evaluation(dataset_name,
                     real_dir,
                     target_size=128,
                     num_imgs=8000,
                     **kwargs):
    # Only make real images if they haven't been made yet
    logger = logging.getLogger('tl')
    if os.path.exists(real_dir):
      img_list = tl2_utils.get_filelist_recursive(real_dir, ext=['*.jpg', '*.png'])
      if len(img_list) > 0:
        img_path = img_list[0]
        img_pil = Image.open(img_path)
        if img_pil.size[0] != target_size:
          shutil.rmtree(real_dir)
          print(f"delete real images at {img_pil.size}")

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataloader, CHANNELS = datasets.get_dataset(dataset_name,
                                                    img_size=target_size,
                                                    shuffle=False,
                                                    **kwargs)
        logger.info('Outputting real images...')
        output_real_images(dataloader, num_imgs, real_dir)
        logger.info('...done')
    else:
        logger.info("Real images exist.")

    return real_dir

def output_images(generator,
                  input_metadata,
                  rank,
                  world_size,
                  fake_dir,
                  num_imgs=2048,
                  img_size=128,):
    os.makedirs(fake_dir, exist_ok=True)
    metadata = copy.deepcopy(input_metadata)
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
            z = torch.randn((metadata['batch_size'], generator.module.z_dim), device=generator.module.device)
            # generated_imgs, _ = generator.module.staged_forward(z, **metadata)
            generated_imgs = generator.module(z,
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

def calculate_fid(real_dir,
                  fake_dir,
                  batch_size=128):

    metrics_dict = calculate_metrics(input1=real_dir,
                                     input2=fake_dir,
                                     cuda=True,
                                     isc=False,
                                     fid=True,
                                     kid=False,
                                     verbose=False)
    fid = metrics_dict['frechet_inception_distance']

    # fid = fid_score.calculate_fid_given_paths(paths=[real_dir, fake_dir],
    #                                           batch_size=batch_size,
    #                                           device='cuda',
    #                                           dims=2048)
    torch.cuda.empty_cache()

    return fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_imgs', type=int, default=8000)

    opt = parser.parse_args()

    real_images_dir = setup_evaluation(opt.dataset, None, target_size=opt.img_size, num_imgs=opt.num_imgs)