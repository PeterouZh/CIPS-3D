import logging
import os
import glob
import PIL
import random
import math
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from tl2.tl2_utils import read_image_list_from_files
from tl2.proj.pytorch.datasets.dataset_danbooru2019_portraits import Danbooru2019_Portraits as Danbooru2019_Portraits_base

class Danbooru2019_Portraits(Danbooru2019_Portraits_base):
    def __getitem__(self, index):
        X = super(Danbooru2019_Portraits, self).__getitem__(index)

        return X, 0


class FFHQ(Dataset):
  """
  python3 -m tl2.tools.get_data_list     \
    --source_dir datasets/ffhq/downsample_ffhq_256x256/  \
    --outfile datasets/ffhq/ffhq_256.txt  \
    --ext *.png
  """

  def __init__(self,
               img_size,
               image_list_file="datasets/ffhq/ffhq_256.txt",
               verbose=False,
               horizontal_flip=True,
               **kwargs):
    super().__init__()

    self.verbose = verbose
    self.image_list = read_image_list_from_files(image_list_file, compress=True)

    assert len(self.image_list) > 0, "Can't find data; make sure you specify the path to your dataset"
    if horizontal_flip:
      self.transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5]),
          transforms.RandomHorizontalFlip(p=0.5),
          # transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
          transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        ])
    else:
      self.transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5]),
          # transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
          transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        ])

    logger = logging.getLogger('tl')
    logger.info(f"\nNum of images ({image_list_file}):\n {len(self)}")
    pass

  def __len__(self):
    return len(self.image_list)

  def __getitem__(self, index):
    image_path = self.image_list[index]
    X = PIL.Image.open(image_path)
    X = self.transform(X)

    if self.verbose:
      return X, image_path
    else:
      return X, 0


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self,
                 img_size,
                 image_list_file="datasets/img_align_celeba.txt",
                 **kwargs):
        super().__init__()

        # self.data = glob.glob('/home/ericryanchan/data/celeba/img_align_celeba/*.jpg')
        self.data = read_image_list_from_files(image_list_file)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize(320),
                     transforms.CenterCrop(256),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5]),
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.Resize((img_size, img_size), interpolation=0)])

        logger = logging.getLogger('tl')
        logger.info(f"\nNum of images (CelebA): {len(self)}")
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index][0])
        X = self.transform(X)

        return X, 0

class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob('/home/marcorm/S-GAN/data/cats_bigger_than_128x128/*.jpg')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0

class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, img_size, **kwargs):
        super().__init__()
        
        self.data = glob.glob('/home/ericryanchan/graf-beta/data/carla/carla/*.png')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=0), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        
        return X, 0


def get_dataset(name, subsample=None, batch_size=1, shuffle=True, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed(
      name,
      world_size,
      rank,
      batch_size,
      num_workers=4,
      **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return dataloader, 3
