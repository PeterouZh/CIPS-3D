# π-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis
### [Project Page](https://marcoamonteiro.github.io/pi-GAN-website/) | [Paper](https://arxiv.org/pdf/2012.00926.pdf) | [Data]()
[Eric Ryan Chan](https://ericryanchan.github.io/about.html)\*,
[Marco Monteiro](https://twitter.com/MonteiroAMarco)\*,
[Petr Kellnhofer](https://kellnhofer.xyz/),
[Jiajun Wu](https://jiajunwu.com/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<br>
\*denotes equal contribution

This is the official implementation of the paper "π-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis".

π-GAN is a novel generative model for high-quality 3D aware image synthesis.

https://user-images.githubusercontent.com/9628319/122865841-e2d1c080-d2db-11eb-9621-1e176db59352.mp4

## Training a Model

The main training script can be found in train.py. Majority of hyperparameters for training and evaluation are set in the curriculums.py file. (see file for more details) We provide recommended curriculums for CelebA, Cats, and CARLA.

##### Relevant Flags:

Set the output directory:
`--output_dir=[output directory]`

Set the model loading directory:
`--load_dir=[load directory]`

Set the current training curriculum:
`--curriculum=[curriculum]`

Set the port for distributed training:
`--port=[port]`


##### To start training:

Update the `dataset_path` field in the curriculum to point to your images.

On one GPU for CelebA:
`CUDA_VISIBLE_DEVICES=0 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

On multiple GPUs, simply list cuda visible devices in a comma-separated list:
`CUDA_VISIBLE_DEVICES=1,3 python3 train.py --curriculum CelebA --output_dir celebAOutputDir`

To continue training from another run specify the `--load_dir=path/to/directory` flag. 

## Model Results and Evaluation

#### Evaluation Metrics
To generate real images for evaluation run
`python fid_evaluation --dataset CelebA --img_size 128 --num_imgs 8000`.
To calculate fid/kid/inception scores run
`python eval_metrics.py path/to/generator.pth --real_image_dir path/to/real_images/directory --curriculum CelebA --num_images 8000`.


#### Rendering Images
`python render_multiview_images.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

For best visual results, load the EMA parameters, use truncation, increase the resolution (e.g. to 512 x 512) and increase the number of depth samples (e.g. to 24 or 36).

#### Rendering Videos
`python render_video.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

You can pass the flag `--lock_view_dependence` to remove view dependent effects. This can help mitigate distracting visual artifacts such as shifting eyebrows. However, locking view dependence may lower the visual quality of images (edges may be blurrier etc.)

#### Rendering Videos Interpolating between faces
`python render_video_interpolation.py path/to/generator.pth --curriculum CelebA --seeds 0 1 2 3`

#### Extracting 3D Shapes

`python extract_shapes.py path/to/generator.pth --curriculum CelebA --seed 0`

## Pretrained Models
We provide pretrained models for CelebA, Cats, and CARLA.

CelebA: https://drive.google.com/file/d/1bRB4-KxQplJryJvqyEa8Ixkf_BVm4Nn6/view?usp=sharing

Cats: https://drive.google.com/file/d/1WBA-WI8DA7FqXn7__0TdBO0eO08C_EhG/view?usp=sharing

CARLA: https://drive.google.com/file/d/1n4eXijbSD48oJVAbAV4hgdcTbT3Yv4xO/view?usp=sharing

All zipped model files contain a generator.pth, ema.pth, and ema2.pth files. ema.pth used a decay of 0.999 and ema2.pth used a decay of 0.9999. All evaluation scripts will by default load the EMA from the file named `ema.pth` in the same directory as the generator.pth file.

## Training Tips

If you have the resources, increasing the number of samples (steps) per ray will dramatically increase the quality of your 3D shapes. If you're looking for good shapes, e.g. for CelebA, try increasing num_steps and moving the back plane (ray_end) to allow the model to move the background back and capture the full head.

Training has been tested to work well on either two RTX 6000's or one RTX 8000. Training with smaller GPU's and batch sizes generally works fine, but it's also possible you'll encounter instability, especially at higher resolutions. Bubbles and artifacts that suddenly appear, or blurring in the tilted angles, are signs that training destabilized. This can usually be mitigated by training with a larger batch size or by reducing the learning rate.

Since the original implementation we added a pose identity component to the loss. Controlled by pos_lambda in the curriculum, the pose idedntity component helps ensure generated scenes share the same canonical pose. Empirically, it seems to improve 3D models, but may introduce a minor decrease in image quality scores.

## Citation

If you find our work useful in your research, please cite:
```
@inproceedings{piGAN2021,
  title={pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis},
  author={Eric Chan and Marco Monteiro and Petr Kellnhofer and Jiajun Wu and Gordon Wetzstein},
  year={2021},
  booktitle={Proc. CVPR},
}
```