set -x

# v2
# bash = bash CIPS-3D/exp/cips3d/bash/afhq_exp/train_afhq_cat_r128.sh 0 bucket-3690

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

python -c "from exp.tests.test_cips3d import Testing_afhq_exp;\
  Testing_afhq_exp().test_train_afhq_cat(debug=False)" \
  --tl_opts \
    batch_size 4 img_size 128 total_iters 50000 warmup_D True fade_steps 10000 \
    train_aux_img False \
    load_finetune True finetune_dir results/CIPS-3D/ffhq_exp_v1/train_ffhq-20220111_034327_756/ckptdir/best_fid












