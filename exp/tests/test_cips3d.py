import os
import sys
import unittest
import argparse


class Testing_snippet(unittest.TestCase):

  def test_copy_files(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_snippet;\
          Testing_snippet().test_copy_files(debug=False)"

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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from tl2.modelarts import moxing_utils

    # moxing_utils.moxing_copy_parallel(src_url="s3://bucket-3690/ZhouPeng/keras/Faces/AFHQv2/",
    #                                   dst_url="/home/ma-user/work/ZhouPeng/.keras/AFHQv2")

    moxing_utils.moxing_copy_parallel(src_url="datasets/photo2cartoon",
                                      dst_url="s3://bucket-3690/ZhouPeng/keras/")
    pass


class Testing_PrepareDatasets(unittest.TestCase):

  def test_afhq_r256(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_PrepareDatasets;\
          Testing_PrepareDatasets().test_afhq_r256(debug=False)"

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
                --tl_config_file none
                --tl_command none
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
        scripts/dataset_tool.py \
        --source=datasets/AFHQv2/AFHQv2 \
        --dest=datasets/AFHQv2/AFHQv2_stylegan2.zip \
        --width=256 --height=256
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  """
    else:
      cmd_str += f"""
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

  def test_photo2cartoon(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_PrepareDatasets;\
          Testing_PrepareDatasets().test_afhq_r256(debug=False)"

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
                --tl_config_file none
                --tl_command none
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
        scripts/dataset_tool.py \
        --source=datasets/photo2cartoon/photo2cartoon \
        --dest=datasets/photo2cartoon/photo2cartoon_stylegan2.zip 
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  """
    else:
      cmd_str += f"""
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


class Testing_ffhq_exp(unittest.TestCase):

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

    FID_r32 = collections.defaultdict(dict)
    title = 'FID_r32'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20211231_221348_328'] = \
      {'20211231_221348_328-ffhq_r32-gpu.8x8': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20211231_221845_770'] = \
      {'20211231_221845_770-ffhq_r32-gpu.4x8': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 100]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r64 = collections.defaultdict(dict)
    title = 'FID_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220101_130834_616'] = \
      {'20220101_130834_616-ffhq_r64-gpu.8x4': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220101_123421_189'] = \
      {'20220101_123421_189-ffhq_r64-gpu.4x8': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220101_141337_543'] = \
      {'20220101_141337_543-ffhq_r64-gpu.4x8-finetune.F': f"{log_file}", }

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
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220102_120440_740'] = \
      {'20220102_120440_740-ffhq_r128-gpu.8x4-aux.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220102_121420_779'] = \
      {'20220102_121420_779-ffhq_r128-gpu.8x4-aux.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq-20220106_090056_403'] = \
      {'20220106_090056_403-ffhq_r128-gpu.8x4-aux.F-warmupD.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 100]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r256 = collections.defaultdict(dict)
    title = 'FID_r256'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220104_161101_025'] = \
      {'20220104_161101_025-ffhq_r256-gpu.8x4-aux.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220104_195958_205'] = \
      {'20220104_195958_205-ffhq_r256-gpu.8x4-aux.F-diffaug.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220105_183636_413'] = \
      {'20220105_183636_413-ffhq_r256-gpu.8x4-aux.F-diffaug.T-warmupD.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220104_182014_278'] = \
      {'20220104_182014_278-ffhq_r256-gpu.8x4-aux.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220104_223240_621'] = \
      {'20220104_223240_621-ffhq_r256-gpu.8x4-aux.T-diffaug.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp/train_ffhq_high-20220105_143314_190'] = \
      {'20220105_143314_190-ffhq_r256-gpu.8x4-aux.T-diffaug.T-warmupD.T': f"{log_file}", }

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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer

    device = 'cuda'

    cfg = TLCfgNode.load_yaml_with_command('exp/cips3d/configs/ffhq_exp.yaml', command='_build_generator')
    G = build_model(cfg.G_cfg).to(device)
    Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    pass

  def test__build_discriminator(self, debug=True):
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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                --tl_debug
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['MAX_JOBS '] = "8"

    import collections
    import torch
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer

    device = 'cuda'

    D = build_model(cfg.D_cfg, kwargs_priority=True, diffaug=True).to(device)

    x = torch.randn(8, 3, 64, 64, device=device, requires_grad=True)
    summary_ddict = collections.defaultdict(dict)

    out = D(x, summary_ddict=summary_ddict)

    # Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)



    pass

  def test__setup_evaluation(self, debug=True):
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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
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
        exp/cips3d/scripts/setup_evaluation.py
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

  def test__gen_images(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
          Testing_ffhq_exp().test__gen_images(debug=False)"

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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
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
        exp/cips3d/scripts/gen_images.py
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

  def test__save_images(self, debug=True):
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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.scripts.train import save_images

    device = 'cuda'

    G = build_model(cfg.G_cfg).to(device)
    Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)
    G.requires_grad_(True)

    fixed_z = G.get_zs(cfg.fixed_z_bs)
    for z in fixed_z.values():
      z.requires_grad_()

    save_images(saved_dir=cfg.tl_outdir,
                G=G,
                G_ema=G,
                G_kwargs=cfg.G_kwargs,
                fixed_z=fixed_z,
                img_size=cfg.img_size)
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
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
          Testing_ffhq_exp().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 8 img_size 32  \
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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
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
        exp/cips3d/scripts/train.py
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

  def test_train_ffhq_high(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
          Testing_ffhq_exp().test_train_ffhq_high(debug=False)" \
          --tl_opts \
            batch_size 2 img_size 256 num_images_gen_eval 512 load_nerf_ema False  \
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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
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
        exp/cips3d/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0 batch_size 2
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

  def test_eval_fid(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp;\
          Testing_ffhq_exp().test_eval_fid(debug=False)" \
          --tl_opts \
            img_size 64 num_images_real_eval 1000 num_images_gen_eval 1000 kid True

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
                --tl_config_file exp/cips3d/configs/ffhq_exp.yaml
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
        exp/cips3d/scripts/eval_fid.py
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


class Testing_ffhq_exp_v1(unittest.TestCase):

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
      {'20220109_174243_622-ffhq_r64-gpu.8x4-num_steps.12': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220114_093931_594'] = \
    #   {'20220114_093931_594-ffhq_r64-gpu.8x4-num_steps.48': f"{log_file}", }


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
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220111_034327_756'] = \
      {'20220111_034327_756-ffhq_r128-gpu.8x4-aux.F-num_steps.12': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220114_115800_280'] = \
      {'20220114_115800_280-ffhq_r128-gpu.8x4-aux.F-num_steps.36': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_r256 = collections.defaultdict(dict)
    title = 'FID_r256'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/ffhq_exp_v1/train_ffhq_high-20220118_144003_691'] = \
      {'20220118_144003_691-ffhq_r256-gpu.8x4-aux.F-num_steps.12': f"{log_file}", }

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
                --tl_config_file exp/cips3d/configs/ffhq_exp_v1.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                --tl_debug
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    import numpy as np
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as tv_f
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pil import pil_utils
    from exp.comm import comm_utils

    device = 'cuda'

    # cfg = TLCfgNode.load_yaml_with_command('exp/cips3d/configs/ffhq_exp.yaml', command='_build_generator')
    G = build_model(cfg.G_cfg).to(device)
    Checkpointer(G).load_state_dict_from_file(cfg.network_pkl)

    bs = 4
    img_size = 128

    zs = G.get_zs(bs)
    imgs = G(zs, img_size=img_size, **cfg.G_kwargs)[0]

    imgs = make_grid(imgs, nrow=int(np.sqrt(bs)), normalize=True, scale_each=True)
    img_pil = tv_f.to_pil_image(imgs)
    pil_utils.imshow_pil(img_pil)

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
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp_v1;\
          Testing_ffhq_exp_v1().test_train_ffhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 64  warmup_D True \
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
                --tl_config_file exp/cips3d/configs/ffhq_exp_v1.yaml
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
        exp/cips3d/scripts/train.py
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

  def test_train_ffhq_high(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_ffhq_exp_v1;\
          Testing_ffhq_exp_v1().test_train_ffhq_high(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 64  warmup_D True \
          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/ffhq_exp_v1.yaml
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
        exp/cips3d/scripts/train.py
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

class Testing_finetuning_exp(unittest.TestCase):

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

    FID_r256 = collections.defaultdict(dict)
    title = 'FID_r256'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/finetuning_exp/finetune_photo2cartoon-20220107_172255_454'] = \
      {'20220107_172255_454-photo2cartoon_r256-gpu.8x4-aux.F-num_steps.12': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/finetuning_exp/finetune_afhq-20220124_193407_473'] = \
      {'20220124_193407_473-afhq_r256-gpu.8x4-aux.F-num_steps.12': f"{log_file}", }

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

  def test_finetune_photo2cartoon(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_finetuning_exp;\
          Testing_finetuning_exp().test_finetune_photo2cartoon(debug=False)" \
          --tl_opts \
            batch_size 2 img_size 256 \
            load_finetune True finetune_dir results/CIPS-3D/ffhq_exp/train_ffhq_high-20220105_143314_190/ckptdir/best_fid

          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/finetuning_exp.yaml
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
        exp/cips3d/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0 batch_size 2
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

  def test_finetune_afhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_finetuning_exp;\
          Testing_finetuning_exp().test_finetune_afhq(debug=False)" \
          --tl_opts \
            batch_size 2 img_size 256 \
            load_finetune True finetune_dir results/CIPS-3D/ffhq_exp/train_ffhq_high-20220105_143314_190/ckptdir/best_fid

          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/finetune_afhq.yaml
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
        exp/cips3d/scripts/train.py
        --port {PORT}

        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts num_workers 0 batch_size 2
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


class Testing_afhq_exp(unittest.TestCase):

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

    FID_afhq_r64 = collections.defaultdict(dict)
    title = 'FID_afhq_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220120_122125_662'] = \
      {'20220120_122125_662-afhq_r64-gpu.8x4-finetune.F-aux.T-num_steps.48': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_afhq_r128 = collections.defaultdict(dict)
    title = 'FID_afhq_r128'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220121_090048_760'] = \
      {'20220121_090048_760-afhq_r128-gpu.8x4-finetune.T-aux.F-num_steps.24': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220123_134221_229'] = \
      {'20220123_134221_229-afhq_r128-gpu.8x4-finetune.T-aux.T-num_steps.24': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    # FID_r128 = collections.defaultdict(dict)
    # title = 'FID_r128'
    # log_file = 'textdir/eval.ma0.FID.log'
    # dd = eval(title)
    # # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220111_145246_883'] = \
    # #   {'20220111_145246_883-afhq_r64-gpu.8x4': f"{log_file}", }
    # # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220111_164049_316'] = \
    # #   {'20220111_164049_316-afhq_r64-gpu.8x4': f"{log_file}", }
    # # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220111_204332_565'] = \
    # #   {'20220111_204332_565-afhq_r64-gpu.8x4': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220112_090440_418'] = \
    #   {'20220112_090440_418-afhq_r64-gpu.8x4-finetune.T-aux.T': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq-20220112_091215_231'] = \
    #   {'20220112_091215_231-afhq_r64-gpu.8x4-finetune.T-aux.F': f"{log_file}", }
    #
    # dd['properties'] = {'title': title,
    #                     # 'xlim': [0, 3000000],
    #                     # 'ylim': [0, 50]
    #                     }
    # default_dicts[title] = dd
    # show_max.append(False)

    FID_cat_r64 = collections.defaultdict(dict)
    title = 'FID_cat_r64'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220112_173835_862'] = \
    #   {'20220112_173835_862-afhq_cat_r64-gpu.8x4-finetune.F-aux.T': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220112_194152_153'] = \
    #   {'20220112_194152_153-afhq_cat_r64-gpu.8x4-finetune.F-aux.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220113_090159_613'] = \
      {'20220113_090159_613-afhq_cat_r64-gpu.8x4-finetune.F-aux.T-num_steps.48': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_cat_r128 = collections.defaultdict(dict)
    title = 'FID_cat_r128'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220112_151835_162'] = \
    #   {'20220112_151835_162-afhq_cat_r128-gpu.8x4-finetune.T-aux.F': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220113_210353_416'] = \
      {'20220113_210353_416-afhq_cat_r128-gpu.8x4-finetune.T-aux.F-num_steps.36': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220114_185938_751'] = \
      {'20220114_185938_751-afhq_cat_r128-aux.F-num_steps.24-nerf_noise_disable.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat-20220114_191837_314'] = \
      {'20220114_191837_314-afhq_cat_r128-aux.T-num_steps.12-nerf_noise_disable.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        # 'xlim': [0, 3000000],
                        # 'ylim': [0, 50]
                        }
    default_dicts[title] = dd
    show_max.append(False)

    FID_cat_r256 = collections.defaultdict(dict)
    title = 'FID_cat_r256'
    log_file = 'textdir/eval.ma0.FID.log'
    dd = eval(title)
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220114_112900_020'] = \
    #   {'20220114_112900_020-afhq_cat_r256-gpu.8x4-finetune.T-aux.F-num_steps.12': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220114_114448_735'] = \
    #   {'20220114_114448_735-afhq_cat_r256-num_steps.24-R1.10-nerf_noise_disable.F': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220114_143143_586'] = \
    #   {'20220114_143143_586-afhq_cat_r256-num_steps.24-R1.20-nerf_noise_disable.F': f"{log_file}", }
    # dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220114_144612_347'] = \
    #   {'20220114_144612_347-afhq_cat_r256-num_steps.24-R1.20-nerf_noise_disable.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220114_165819_026'] = \
      {'20220114_165819_026-afhq_cat_r256-num_steps.24-R1.10-nerf_noise_disable.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220118_090720_512'] = \
      {'20220118_090720_512-afhq_cat_r256-aux.F-num_steps.24-R1.10-nerf_noise_disable.T': f"{log_file}", }
    dd[f'{bucket_root}/results/CIPS-3D/afhq_exp/train_afhq_cat_high-20220118_091548_726'] = \
      {'20220118_091548_726-afhq_cat_r256-aux.T-num_steps.12-R1.10-nerf_noise_disable.T': f"{log_file}", }

    dd['properties'] = {'title': title,
                        'xlim': [0, 300000],
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

  def test_train_afhq(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_afhq_exp;\
          Testing_afhq_exp().test_train_afhq(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32  \
          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/afhq_exp.yaml
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
        exp/cips3d/scripts/train.py
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

  def test_train_afhq_cat(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_afhq_exp;\
          Testing_afhq_exp().test_train_afhq_cat(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32  \
            load_finetune True \
            finetune_dir results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220111_034327_756/ckptdir/best_fid \
          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/afhq_exp.yaml
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
        exp/cips3d/scripts/train.py
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

  def test_train_afhq_cat_high(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib
        python -c "from exp.tests.test_cips3d import Testing_afhq_exp;\
          Testing_afhq_exp().test_train_afhq_cat_high(debug=False)" \
          --tl_opts \
            batch_size 4 img_size 32  \
          --tl_outdir


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
                --tl_config_file exp/cips3d/configs/afhq_exp.yaml
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
        exp/cips3d/scripts/train.py
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
