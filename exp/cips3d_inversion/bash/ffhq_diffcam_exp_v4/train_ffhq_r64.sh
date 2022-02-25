set -x

# v2

# Env vars e.g.
PROJ_NAME=CIPS-3D

run_num=${1:-0}
bucket=${2:-bucket-3690}
#cuda_devices=${3:-0,1,2,3,4,5,6,7}
cuda_devices=`python -c "import torch;print(','.join([str(i) for i in range(torch.cuda.device_count())]), end='')"`


#curdir: /home/ma-user/modelarts/user-job-dir
pwd && ls -la

proj_root=$PROJ_NAME

############ copy code
cd $proj_root

## modelarts code
# copy tool
pip install tl2

python -m tl2.modelarts.scripts.copy_tool \
  -s s3://$bucket/ZhouPeng/codes/$proj_root \
  -d ../$proj_root \
  -t copytree -b ../$proj_root/code.zip
## cache code
python -m tl2.modelarts.scripts.copy_tool \
  -s s3://$bucket/ZhouPeng/codes/$proj_root \
  -d /cache/$proj_root \
  -t copytree -b /cache/$proj_root/code.zip

cd /cache/$proj_root
pwd

############ copy results
#resume_dir=encoder_inr_train/train_ffhq_r256_softplus-20211219_144749_467
#python -m tl2.modelarts.scripts.copy_tool \
#  -s s3://$bucket/ZhouPeng/results/$proj_root/$resume_dir \
#  -d /cache/$proj_root/results/$resume_dir -t copytree

#finetune_pkl=encoder_inr_train/train_ffhq_r256_softplus-20211217_175316_465/ckptdir/resume/snapshot_data.pkl
#python -m tl2.modelarts.scripts.copy_tool \
#  -s s3://$bucket/ZhouPeng/results/$proj_root/$finetune_pkl \
#  -d /cache/$proj_root/results/$finetune_pkl -t copy

############ Prepare envs
bash exp/tests/setup_env.sh
#pip uninstall -y tl2

export ANSI_COLORS_DISABLED=1

export CUDA_VISIBLE_DEVICES=$cuda_devices
export RUN_NUM=$run_num

export TIME_STR=1
export PORT=12345
#
export PYTHONPATH=.:./tl2_lib

# bash = bash CIPS-3D/exp/cips3d_inversion/bash/ffhq_diffcam_exp_v4/train_ffhq_r64.sh 0 bucket-3690

python -c "from exp.tests.test_cips3d_inversion import Testing_ffhq_diffcam_exp_v4;\
  Testing_ffhq_diffcam_exp_v4().test_train_ffhq(debug=False)" \
  --tl_opts \
    batch_size 4 img_size 64 total_iters 200000 \
    warmup_D True fade_steps 10000 \
    train_aux_img True G_kwargs.nerf_kwargs.N_samples 12 G_kwargs.nerf_kwargs.N_importance 12 \
    grad_points 64 \
    G_cfg.shape_block_end_index 2 G_cfg.app_block_end_index 1 G_cfg.inr_block_end_index 4 \
    G_cfg.nerf_cfg.shape_net_cfg.use_pos_enc True G_cfg.nerf_cfg.shape_net_cfg.gradient_scale None \
    G_cfg.nerf_cfg.shape_net_cfg.freq_scale 1. G_cfg.nerf_cfg.shape_net_cfg.freq_shift 4.86 \
    G_cfg.inr_detach True \
    cam_cfg.normalize_rays_d True \
    load_finetune False
#    load_finetune True finetune_dir results/CIPS-3D/ffhq_diffcam_exp_v4/train_ffhq-20220222_225039_342/ckptdir/resume

#  --tl_outdir results/ffhq_exp/train_ffhq









