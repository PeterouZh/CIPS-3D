import os
import sys
import unittest
import argparse

import torch


class Testing_ffhq_diffcam_exp(unittest.TestCase):

  def test__build_generator(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export PORT=12345
        export TIME_STR=1
        export PYTHONPATH=.
        python -c "from tl2.launch.tests.test_launch import Testing_Launch_v1;\
          Testing_Launch_v1().test_launch_ddp(debug=False)" \
          --tl_opts root_obs s3://$bucket/ZhouPeng/ \
          --tl_outdir results/train_ffhq_256/train_ffhq_256-20210726_202423_412
          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    G = Generator_Diffcam(**cfg.G_cfg).to(device)
    Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs

    # metadata['h_stddev'] = 0
    # metadata['v_stddev'] = 0
    # metadata['img_size'] = 128
    # metadata['batch_size'] = 4
    # metadata['psi'] = 1

    num_imgs = 4
    H = W = 128
    # N_rays = 1024
    N_rays = -1

    cam_param = cam_params.CamParams.from_config(num_imgs=num_imgs,
                                                 H0=H,
                                                 W0=W).cuda()

    idx = list(range(num_imgs))
    R, t, fx, fy = cam_param(idx)

    rays_o, rays_d, select_inds = G.get_rays_axis_angle(R=R, t=t, fx=fx, fy=fy, H=H, W=W, N_rays=-1)
    # G.eval()

    with torch.set_grad_enabled(True):
      zs = G.get_zs(num_imgs)
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=None, # disable gradients
                         return_aux_img=True,
                         **metadata)

    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, imgs.shape)

    pass

