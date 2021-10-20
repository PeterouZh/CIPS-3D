## CIPS-3D

This repository will contain the code of the paper, </br > 
[CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2110.09788).


We are planning to publish the training code here in December. But if the github star reaches two hundred, I will advance the date. Stay tuned :clock10:.


## Demo videos

https://user-images.githubusercontent.com/26176709/137924071-26f700b1-46dc-4c1d-bb2d-189e6cc09116.mp4

https://user-images.githubusercontent.com/26176709/137924277-751c342b-87c0-4539-8ab9-96b795d257ab.mp4

https://user-images.githubusercontent.com/26176709/137924346-dd628c97-64e5-4cf7-9e34-ac6c42a98d3d.mp4

https://user-images.githubusercontent.com/26176709/137924529-c38fa07c-9673-42ab-8a27-8e510d4c65ca.mp4

https://user-images.githubusercontent.com/26176709/137924557-1aa23be9-d079-472e-8a9f-0e08f78fdce8.mp4

https://user-images.githubusercontent.com/26176709/137924581-f5dbf759-1c8c-4dc3-9b85-26f215f0fde0.mp4


## Mirror symmetry problem

<img src="./.github/mirror_symm.png" width="800">

The problem of mirror symmetry refers to the sudden change of the direction of the bangs near the yaw angle of pi/2. We propose to use an auxiliary discriminator to solve this problem (please see the paper).

Note that in the initial stage of training, the auxiliary discriminator must dominate the generator more than the main discriminator does. Otherwise, if the main discriminator dominates the generator, the mirror symmetry problem will still occur. In practice, progressive training is able to guarantee this. We have trained many times from scratch. Adding an auxiliary discriminator stably solves the mirror symmetry problem. If you find any problems with this idea, please open an issue. 

## Envs

```bash


```


## Training

```bash

```


## Citation

If you find our work useful in your research, please cite:
```

@article{zhou2021CIPS3D,
  title = {{{CIPS}}-{{3D}}: A {{3D}}-{{Aware Generator}} of {{GANs Based}} on {{Conditionally}}-{{Independent Pixel Synthesis}}},
  shorttitle = {{{CIPS}}-{{3D}}},
  author = {Zhou, Peng and Xie, Lingxi and Ni, Bingbing and Tian, Qi},
  year = {2021},
  eprint = {2110.09788},
  eprinttype = {arxiv},
  primaryclass = {cs, eess},
  archiveprefix = {arXiv}
}

```

## Acknowledgments

- pi-GAN from [https://github.com/marcoamonteiro/pi-GAN](https://github.com/marcoamonteiro/pi-GAN)
- CIPS from [https://github.com/saic-mdal/CIPS](https://github.com/saic-mdal/CIPS)
- StyleGAN2 from [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
- torch-fidelity from [https://github.com/toshas/torch-fidelity](https://github.com/toshas/torch-fidelity)
- StudioGAN from [https://github.com/POSTECH-CVLab/PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)

