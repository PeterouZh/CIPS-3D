import importlib
import os
import sys
import unittest
import argparse

import torch

from tl2.proj.pytorch import torch_utils



class Testing_ffhq_diffcam_exp(unittest.TestCase):
  """
  NeRF: siren;

  """
  def test__plot_fid(self):
    """
    export PYTHONPATH=.:./tl2_lib
    python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
      Testing_ffhq_exp().test__plot_fid()"

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                  --tl_config_file none
                  --tl_command none
                  --tl_outdir {outdir}
                  """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from tl2.proj.matplot.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    bucket_root = "/home/ma-user/work/ZhouPeng/bucket_3690/"

    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220109_174243_622'] = \
      {'20220109_174243_622-cips_v1': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220123_203119_425'] = \
      {'20220123_203119_425-ffhq_r64-gpu.4x8-finetune.F': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_151831_125'] = \
    #   {'20220124_151831_125-ffhq_r64-gpu.4x8-finetune.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_160131_073'] = \
      {'20220124_160131_073-ffhq_r64-gpu.4x8-finetune.F-far.1_12-gridwraper.T': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_163418_396'] = \
    #   {'20220124_163418_396-ffhq_r64-gpu.4x8-finetune.F-far.1-gridwraper.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_164043_043'] = \
      {'20220124_164043_043-ffhq_r64-gpu.4x8-finetune.F-far.1_5-gridwraper.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_171134_132'] = \
      {'20220124_171134_132-ffhq_r64-gpu.4x8-finetune.F-far.1_7-gridwraper.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_172907_111'] = \
      {'20220124_172907_111-ffhq_r64-gpu.4x8-finetune.F-far.1_12-gridwraper.F': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r64_debug = collections.defaultdict(dict)
    title = 'FID_r64_debug'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_160131_073'] = \
      {'20220124_160131_073-ffhq_r64-gpu.4x8-finetune.F-far.1_12-gridwraper.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_164043_043'] = \
      {'20220124_164043_043-ffhq_r64-gpu.4x8-finetune.F-far.1_5-gridwraper.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220217_061738_838'] = \
      {'20220217_061738_838-ffhq_r64-gpu.4x8-finetune.F-far.1_5-grad_points.64': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220217_093119_701'] = \
      {'20220217_093119_701-ffhq_r64-gpu.4x8-finetune.F-far.1_5-grad_points.32': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r128 = collections.defaultdict(dict)
    title = 'FID_r128'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220216_110534_197'] = \
      {'20220216_110534_197-ffhq_r128-gpu.4x8-finetune.T-freeze_intr.F-grad_points.96': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220216_140026_050'] = \
      {'20220216_140026_050-ffhq_r128-gpu.4x8-finetune.T-freeze_intr.T-grad_points.96': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220216_141517_754'] = \
      {'20220216_141517_754-ffhq_r128-gpu.4x8-finetune.T-freeze_intr.T-grad_points.128': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220217_174134_576'] = \
      {'20220217_174134_576-ffhq_r128-gpu.4x8-finetune.T-freeze_intr.T-grad_points.128': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220217_235524_842'] = \
      {'20220217_235524_842-ffhq_r128-gpu.4x8-finetune.T-freeze_intr.T-grad_points.128': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

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
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 4
    H = W = 128
    # N_rays = 1024
    N_rays = -1

    cam_param = cam_params.CamParams.from_config(num_imgs=1,
                                                 H0=H,
                                                 W0=W).cuda()

    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_164043_043/ckptdir/resume"
    model_dict = {
      'G_ema': G,
      'cam_param': cam_param
    }
    torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    # idx = list(range(num_imgs))
    # R, t, fx, fy = cam_param(idx)
    # rays_o, rays_d, select_inds = G.get_rays_axis_angle(R=R, t=t, fx=fx, fy=fy, H=H, W=W, N_rays=-1)

    # rays_o, rays_d = cam_param.get_rays_of_pose_avg(H=H, W=W, bs=num_imgs)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(True):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         grad_points=32**2,
                         forward_points=None,  # disable gradients
                         return_aux_img=True,
                         **metadata)
    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"grad_points {imgs.shape}")

    with torch.set_grad_enabled(False):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=32 ** 2,  # disable gradients
                         return_aux_img=True,
                         **metadata)
    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"forward_points {imgs.shape}")


    with torch.set_grad_enabled(False):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=None, # disable gradients
                         return_aux_img=True,
                         **metadata)
    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test__gen_images(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp;\
          Testing_ffhq_diffcam_exp().test__gen_images(debug=False)"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python -m torch.distributed.launch --nproc_per_node={n_gpus} --master_port={PORT} 
        exp/cips3d_inversion/scripts/gen_images.py
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts 
                  """
    else:
      cmd_str += f"""
                  --num_workers {n_gpus}
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import modelarts_utils
    # update_parser_defaults_from_yaml(parser)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp;\
          Testing_ffhq_diffcam_exp().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 load_finetune True \
          --tl_outdir results/ffhq_exp/train_ffhq


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
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

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass


class Testing_ffhq_diffcam_exp_v1(unittest.TestCase):
  """
  NeRF: cips_net + cips_net;
  geometry problem;

  """

  def test__plot_fid(self):
    """
    export PYTHONPATH=.:./tl2_lib
    python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
      Testing_ffhq_exp().test__plot_fid()"

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                  --tl_config_file none
                  --tl_command none
                  --tl_outdir {outdir}
                  """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from tl2.proj.matplot.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    bucket_root = "/home/ma-user/work/ZhouPeng/bucket_3690/"

    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp/train_ffhq-20220124_160131_073'] = \
      {'20220124_160131_073-ffhq_r64-gpu.4x8-finetune.F-far.1_12-gridwraper.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_012019_645'] = \
      {'20220218_012019_645-ffhq_r128-gpu.4x4-': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_023549_830'] = \
      {'20220218_023549_830-ffhq_r128-gpu.4x8-': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_093950_696'] = \
      {'20220218_093950_696-ffhq_r128-gpu.4x8-gridscale.F': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_104824_186'] = \
    #   {'20220218_104824_186-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_113228_416'] = \
      {'20220218_113228_416-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_141818_940'] = \
      {'20220218_141818_940-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F-xyz_affine.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_225857_033'] = \
      {'20220218_225857_033-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F-xyz_affine.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r128 = collections.defaultdict(dict)
    title = 'FID_r128'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220219_104511_901'] = \
      {'20220219_104511_901-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F-xyz_affine.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220219_115100_194'] = \
      {'20220219_115100_194-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F-xyz_affine.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

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
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v1.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts} 
                --tl_debug True
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator_v1 import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    G = Generator_Diffcam(**cfg.G_cfg).to(device)
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 4
    H = W = 128
    # N_rays = 1024
    N_rays = -1

    cam_param = cam_params.CamParams.from_config(num_imgs=1,
                                                 H0=H,
                                                 W0=W).cuda()

    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_113228_416/ckptdir/resume"
    # ckpt_dir = "results/ffhq_diffcam_exp_v1/train_ffhq/ckptdir/resume"
    model_dict = {
      'G_ema': G,
      'cam_param': cam_param
    }
    # torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(False):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=256**2,  # disable gradients
                         return_aux_img=True,
                         **metadata)
    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v1;\
          Testing_ffhq_diffcam_exp_v1().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v1.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_interpolate_z_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v1;\
          Testing_ffhq_diffcam_exp_v1().test_interpolate_z_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v1.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test_sampling_yaw_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v1;\
          Testing_ffhq_diffcam_exp_v1().test_sampling_yaw_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v1.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass


class Testing_ffhq_diffcam_exp_v2(unittest.TestCase):
  """
  NeRF: siren_skip_net + cips_net;
  issue: large G gradients;

  """

  def test__plot_fid(self):
    """
    export PYTHONPATH=.:./tl2_lib
    python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
      Testing_ffhq_exp().test__plot_fid()"

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                  --tl_config_file none
                  --tl_command none
                  --tl_outdir {outdir}
                  """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from tl2.proj.matplot.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    bucket_root = "/home/ma-user/work/ZhouPeng/bucket_3690/"


    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v1/train_ffhq-20220218_225857_033'] = \
      {'20220218_225857_033-ffhq_r128-gpu.4x8-gridscale.F-appendxyz.F-xyz_affine.T': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_175046_931'] = \
    #   {'20220219_175046_931-ffhq_r128-gpu.4x8-shape_block_end_index.2': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_182559_307'] = \
      {'20220219_182559_307-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.1': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_194936_950'] = \
    #   {'20220219_194936_950-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.2': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_211756_715'] = \
      {'20220219_211756_715-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.3': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220221_091857_103'] = \
      {'20220221_091857_103-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.1-norm_raysd.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220221_092814_741'] = \
      {'20220221_092814_741-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.1-norm_raysd.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

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
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts} 
                --tl_debug True
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator_v2 import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 2
    H = W = 64
    # N_rays = 1024
    N_rays = -1

    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_182559_307/ckptdir/resume"
    # ckpt_dir = None
    # ckpt_dir = "results/ffhq_diffcam_exp_v1/train_ffhq/ckptdir/resume"
    if ckpt_dir is not None:
      load_G_cfg = TLCfgNode.load_yaml_file(cfg_filename=f"{os.path.abspath(ckpt_dir)}/config_command.yaml")
      load_G_cfg = list(load_G_cfg.values())[0]
      D = torch.load(f"{os.path.abspath(ckpt_dir)}/discriminator_model.pth")
    else:
      load_G_cfg = cfg
      D = None
    G = Generator_Diffcam(**load_G_cfg.G_cfg).to(device)

    cam_param = cam_params.CamParams.from_config(H0=H, W0=W, **load_G_cfg.get('cam_cfg', {})).cuda()

    # D = build_model(cfg=load_G_cfg.D_cfg, kwargs_priority=True, diffaug=False).to(device)

    model_dict = {
      # 'G_ema': G,
      'generator': G,
      'cam_param': cam_param,
      # 'D': D,
    }
    torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(True):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=None,  # disable gradients
                         return_aux_img=True,
                         **metadata)

      if D is not None:
        g_imgs_aux = ret_imgs['aux_img']
        gen_imgs = torch.cat([imgs, g_imgs_aux], dim=0)

        g_preds, _, _ = D(gen_imgs.to(torch.float32), alpha=1, use_aux_disc=True)
        g_loss = torch.nn.functional.softplus(-g_preds).mean()
        g_loss.backward()
        grad_str = torch_utils.get_grad_norm_string(G.named_parameters(), )
        print(grad_str)
        # G_total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), 10.)
        grad_dict = G.get_subnet_grad_norm()

    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v2;\
          Testing_ffhq_diffcam_exp_v2().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_interpolate_z_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v2;\
          Testing_ffhq_diffcam_exp_v2().test_interpolate_z_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test_sampling_yaw_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v2;\
          Testing_ffhq_diffcam_exp_v2().test_sampling_yaw_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v2.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass


class Testing_ffhq_diffcam_exp_v3(unittest.TestCase):
  """
  shape app inr: mod_fc + siren, cips_net, cips_net;
  grad_clip for nerf_net;

  """

  def test__plot_fid(self):
    """
    export PYTHONPATH=.:./tl2_lib
    python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
      Testing_ffhq_exp().test__plot_fid()"

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                  --tl_config_file none
                  --tl_command none
                  --tl_outdir {outdir}
                  """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from tl2.proj.matplot.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    bucket_root = "/home/ma-user/work/ZhouPeng/bucket_3690/"

    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220221_212808_342'] = \
      {'20220221_091857_103-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.1': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220221_213111_543'] = \
      {'20220221_213111_543-ffhq_r128-gpu.4x8-shape_block.3-app_blocks.1': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220222_064527_486'] = \
      {'20220222_064527_486-ffhq_r128-gpu.4x8-shape_block.3-app_blocks.1-sigma_mul.20': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r64_xyz_mod_film = collections.defaultdict(dict)
    title = 'FID_r64_xyz_mod_film'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v2/train_ffhq-20220219_182559_307'] = \
      {'20220219_182559_307-ffhq_r128-gpu.4x8-shape_block.1-app_blocks.1-siren': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220221_213111_543'] = \
      {'20220221_213111_543-ffhq_r128-gpu.4x8-shape_block.3-app_blocks.1': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220222_160912_420'] = \
      {'20220222_160912_420-ffhq_r128-gpu.4x8-shape_block.1-skip.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220222_161353_390'] = \
      {'20220222_161353_390-ffhq_r128-gpu.4x8-shape_block.3-skip.F': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 100000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

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
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v3.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts} 
                --tl_debug True
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator_v3 import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 4
    H = W = 64
    # N_rays = 1024
    N_rays = -1


    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v3/train_ffhq-20220222_160912_420/ckptdir/resume"
    # ckpt_dir = None
    # ckpt_dir = "results/ffhq_diffcam_exp_v1/train_ffhq/ckptdir/resume"
    if ckpt_dir is not None:
      load_G_cfg = TLCfgNode.load_yaml_file(cfg_filename=f"{os.path.abspath(ckpt_dir)}/config_command.yaml")
      load_G_cfg = list(load_G_cfg.values())[0]
      D = torch.load(f"{os.path.abspath(ckpt_dir)}/discriminator_model.pth")
    else:
      load_G_cfg = cfg
      D = None
    G = Generator_Diffcam(**load_G_cfg.G_cfg).to(device)

    cam_param = cam_params.CamParams.from_config(H0=H, W0=W, **load_G_cfg.get('cam_cfg', {})).cuda()

    model_dict = {
      # 'G_ema': G,
      'generator': G,
      'cam_param': cam_param
    }
    if ckpt_dir is not None:
      torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(True):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=256 ** 2,  # disable gradients
                         return_aux_img=True,
                         **metadata)

      if D is not None:
        g_imgs_aux = ret_imgs['aux_img']
        gen_imgs = torch.cat([imgs, g_imgs_aux], dim=0)

        g_preds, _, _ = D(gen_imgs.to(torch.float32), alpha=1, use_aux_disc=True)
        g_loss = torch.nn.functional.softplus(-g_preds).mean()
        g_loss.backward()
        grad_str = torch_utils.get_grad_norm_string(G.named_parameters(), )
        print(grad_str)

        grad_dict = G.get_subnet_grad_norm()
        # G_total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), 10.)

    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v3;\
          Testing_ffhq_diffcam_exp_v3().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v3.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_sampling_yaw_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v3;\
          Testing_ffhq_diffcam_exp_v3().test_sampling_yaw_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v3.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass


class Testing_ffhq_diffcam_exp_v4(unittest.TestCase):
  """
  shape app inr: mod_film, cips_net, cips_net;

  """

  def test__plot_fid(self):
    """
    export PYTHONPATH=.:./tl2_lib
    python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
      Testing_ffhq_exp().test__plot_fid()"

    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                  --tl_config_file none
                  --tl_command none
                  --tl_outdir {outdir}
                  """
    args = setup_outdir_and_yaml(argv_str)
    outdir = args.tl_outdir

    from tl2.proj.matplot.plot_results import PlotResults
    import collections
    import pickle

    outfigure = os.path.join(outdir, 'FID.jpg')
    default_dicts = collections.OrderedDict()
    show_max = []

    bucket_root = "/home/ma-user/work/ZhouPeng/bucket_3690/"

    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220222_224654_660'] = \
    #   {'20220222_224654_660-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220222_225039_342'] = \
      {'20220222_225039_342-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_134902_190'] = \
      {'20220223_134902_190-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T-freq_shift.4.86': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_150208_893'] = \
    #   {'20220223_150208_893-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T-freq_shift.0': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_233549_972'] = \
    #   {'20220223_233549_972-ffhq_r128-gpu.4x8-shape_block.8-pos_enc.T-freq_shift.4.86': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_235312_246'] = \
    #   {'20220223_235312_246-ffhq_r128-gpu.4x8-shape_block.8-pos_enc.T-freq_shift.30-gs.15': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220225_164209_334'] = \
      {'20220225_164209_334-ffhq_r128-gpu.4x8-block_idx.2_1_4-inr_detach.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220225_223444_066'] = \
      {'20220225_223444_066-ffhq_r128-gpu.4x8-block_idx.2_1_9-inr_detach.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220225_225313_381'] = \
      {'20220225_225313_381-ffhq_r128-gpu.4x8-block_idx.2_1_9-inr_detach.T-app.c128': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 100]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r128 = collections.defaultdict(dict)
    title = 'FID_r128'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_093152_188'] = \
      {'20220223_093152_188-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T-inr_block.9': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_104711_687'] = \
      {'20220223_104711_687-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T-inr_block.1': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220224_091527_603'] = \
    #   {'20220224_091527_603-ffhq_r128-gpu.4x8-shape_block.2-freq_shift.4.86': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220224_092021_015'] = \
      {'20220224_092021_015-ffhq_r128-gpu.4x8-shape_block.2-freq_shift.4.86': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220227_232054_529'] = \
      {'20220227_232054_529-ffhq_r128-gpu.4x8-block_idx.2_1_9-inr_detach.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220227_232853_341'] = \
      {'20220227_232853_341-ffhq_r128-gpu.4x8-block_idx.2_1_9-inr_detach.T-2_1_4': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220227_233438_211'] = \
      {'20220227_233438_211-ffhq_r128-gpu.4x8-block_idx.2_1_9-inr_detach.T-app.c128': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 100]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r256 = collections.defaultdict(dict)
    title = 'FID_r256'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq_freeze_nerf-20220224_115722_103'] = \
      {'20220224_115722_103-ffhq_r128-gpu.4x8-shape_block.2-pos_enc.T-inr_block.9': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 100]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    plotobs = PlotResults()
    label2datas_list = plotobs.plot_defaultdicts(
      outfigure=outfigure, default_dicts=default_dicts, show_max=show_max, figsize_wh=(16, 7.2))
    print(f'Save to {outfigure}.')

    saved_data = '__'.join(outdir.split('/')[-2:])
    saved_data = f"{outdir}/{saved_data}.pkl"
    with open(saved_data, 'wb') as f:
      pickle.dump(label2datas_list, f)
    print(f"Save data to {saved_data}")
    pass

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
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v4.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts} 
                --tl_debug True
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator_v4 import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 2
    H = W = 64
    # N_rays = 1024
    N_rays = -1

    # ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220222_224654_660/ckptdir/resume"
    # ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220222_225039_342/ckptdir/resume"
    # ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220223_134902_190/ckptdir/resume"
    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220225_164209_334/ckptdir/resume"
    # ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq_freeze_nerf-20220224_115722_103/ckptdir/resume"
    # ckpt_dir = None

    if ckpt_dir is not None:
      load_G_cfg = TLCfgNode.load_yaml_file(cfg_filename=f"{os.path.abspath(ckpt_dir)}/config_command.yaml")
      load_G_cfg = list(load_G_cfg.values())[0]
      # load_G_cfg.G_cfg.shape_block_end_index = 8
      # load_G_cfg.G_cfg.nerf_cfg.shape_net_cfg.gradient_scale = 1/15
      D = torch.load(f"{os.path.abspath(ckpt_dir)}/discriminator_model.pth")
    else:
      load_G_cfg = cfg
      D = None
    G = Generator_Diffcam(**load_G_cfg.G_cfg).to(device)

    cam_param = cam_params.CamParams.from_config(H0=H, W0=W, **load_G_cfg.get('cam_cfg', {})).cuda()

    model_dict = {
      # 'G_ema': G,
      'generator': G,
      'cam_param': cam_param
    }
    if ckpt_dir is not None:
      torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(True):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=256 ** 2,  # disable gradients
                         return_aux_img=True,
                         **metadata)

      g_imgs_aux = ret_imgs['aux_img']
      gen_imgs = torch.cat([imgs, g_imgs_aux], dim=0)

      if D is not None:

        g_preds, _, _ = D(gen_imgs.to(torch.float32), alpha=1, use_aux_disc=True)
        g_loss = torch.nn.functional.softplus(-g_preds).mean()
        g_loss.backward()
        grad_str = torch_utils.get_grad_norm_string(G.named_parameters(), )
        print(grad_str)

        grad_dict = G.get_subnet_grad_norm()
        # G_total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), 10.)

    img = make_grid(gen_imgs, nrow=num_imgs, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v4;\
          Testing_ffhq_diffcam_exp_v4().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v4.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_train_ffhq_freeze_nerf(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v4;\
          Testing_ffhq_diffcam_exp_v4().test_train_ffhq_freeze_nerf(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v4.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_interpolate_z_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v4;\
          Testing_ffhq_diffcam_exp_v4().test_interpolate_z_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v4.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test_sampling_yaw_web(self, debug=True):
    """
    Usage:
        source activate PyTorch-1.8

        export CUDA_VISIBLE_DEVICES=1
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v4;\
          Testing_ffhq_diffcam_exp_v4().test_sampling_yaw_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v4.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass


class Testing_ffhq_diffcam_exp_v5(unittest.TestCase):
  """
  shape app inr: pigan (mod_film), pigan (mod_film), cips_net;
  gradient detach;

  """

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
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v5.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts} 
                --tl_debug True
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pytorch.examples.nerf import cam_params
    from exp.cips3d_inversion.models.generator_v5 import Generator_Diffcam

    torch_utils.init_seeds(seed=0)

    device = 'cuda'

    # G = build_model(cfg.G_cfg).to(device)
    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    metadata = cfg.G_kwargs
    metadata['nerf_kwargs']['h_stddev'] = 0.
    metadata['nerf_kwargs']['v_stddev'] = 0.

    num_imgs = 2
    H = W = 64
    # N_rays = 1024
    N_rays = -1

    ckpt_dir = "../bucket_3690/results/CIPS-3D/ffhq_diffcam_exp_v5/train_ffhq-20220225_144057_120/ckptdir/resume"
    # ckpt_dir = None

    if ckpt_dir is not None:
      load_G_cfg = TLCfgNode.load_yaml_file(cfg_filename=f"{os.path.abspath(ckpt_dir)}/config_command.yaml")
      load_G_cfg = list(load_G_cfg.values())[0]

      # load_G_cfg.G_cfg.shape_block_end_index = 8
      load_G_cfg.G_cfg.nerf_cfg.shape_net_cfg.grad_norm_layer = True

      D = torch.load(f"{os.path.abspath(ckpt_dir)}/discriminator_model.pth")
    else:
      load_G_cfg = cfg
      D = None
    G = Generator_Diffcam(**load_G_cfg.G_cfg).to(device)

    cam_param = cam_params.CamParams.from_config(H0=H, W0=W, **load_G_cfg.get('cam_cfg', {})).cuda()

    model_dict = {
      # 'G_ema': G,
      'generator': G,
      'cam_param': cam_param
    }
    if ckpt_dir is not None:
      torch_utils.load_models(ckpt_dir, model_dict=model_dict)

    intr = cam_param(mode='get_intrinsic')
    rays_o, rays_d, select_inds = cam_param.get_rays_random_pose(
      device=device, bs=num_imgs, intr=intr, **metadata.nerf_kwargs)

    # G.eval()
    zs = G.get_zs(num_imgs)

    with torch.set_grad_enabled(True):
      imgs, ret_imgs = G(zs=zs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         forward_points=256 ** 2,  # disable gradients
                         return_aux_img=True,
                         **metadata)

      if D is not None:
        g_imgs_aux = ret_imgs['aux_img']
        gen_imgs = torch.cat([imgs, g_imgs_aux], dim=0)

        g_preds, _, _ = D(gen_imgs.to(torch.float32), alpha=1, use_aux_disc=True)
        g_loss = torch.nn.functional.softplus(-g_preds).mean()
        g_loss.backward()
        grad_str = torch_utils.get_grad_norm_string(G.named_parameters(), )
        print(grad_str)

        grad_dict = G.get_subnet_grad_norm()
        # G_total_norm = torch.nn.utils.clip_grad_norm_(G.parameters(), 10.)

    img = make_grid(imgs, nrow=2, normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(img)
    pil_utils.imshow_pil(img_pil, f"whole {imgs.shape}")

    pass

  def test_train_ffhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v5;\
          Testing_ffhq_diffcam_exp_v5().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32 \
            G_cfg.nerf_cfg.scale_factor None G_cfg.inr_block_end_index 1 \
            load_finetune False \
          --tl_outdir results/ffhq_exp/train_ffhq


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
      # sys.argv.extend(['--tl_outdir', 'results/ffhq_exp/train_ffhq'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d_inversion/configs/ffhq_diffcam_exp_v5.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    cmd_str = f"""
        python 
        exp/cips3d_inversion/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --tl_opts num_workers {n_gpus} 
                    {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass
