set -x

# v2
# "bash CIPS-3D/bash/modelarts/run_train_ffhq.sh 0 bucket-3690"

# Env vars e.g.
PROJ_NAME=CIPS-3D

run_num=${1:-0}
bucket=${2:-bucket-3690}
cuda_devices=${3:-0,1,2,3,4,5,6,7}

#curdir: /home/ma-user/modelarts/user-job-dir
pwd
ls -la

proj_root=$PROJ_NAME

############ copy code
cd $proj_root
pip install tl2==0.0.6
## modelarts code
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
############ Prepare envs
        python -m tl2.modelarts.scripts.copy_tool \
          -s s3://$bucket/ZhouPeng/pypi/torch182_cu101_py36 -d /cache/pypi -t copytree
        for filename in /cache/pypi/*.whl; do
            pip install $filename
        done
pip install --no-cache-dir -r requirements.txt
pip install -e torch_fidelity_lib
pip install -e pytorch_ema_lib
# copy results
resume_dir=train_ffhq
python -m tl2.modelarts.scripts.copy_tool \
  -s s3://bucket-3690/ZhouPeng/results/$proj_root/$resume_dir \
  -d /cache/$proj_root/results/$resume_dir -t copytree

export CUDA_VISIBLE_DEVICES=$cuda_devices
export CUDA_HOME=/usr/local/cuda-10.2/
export PYTHONPATH=.


## train 128x128
python exp/dev/nerf_inr/scripts/train_v16.py \
    --port 8888 \
    --tl_config_file configs/train_ffhq.yaml \
    --tl_command train_ffhq_r128 \
    --modelarts True \
    --tl_resume \
    --tl_resumedir results/train_ffhq \
    --tl_opts curriculum.new_attrs.image_list_file datasets/ffhq/ffhq_256.txt \
      D_first_layer_warmup True reset_best_fid True update_aux_every 16 d_reg_every 8 \
    --tl_outdir results/$resume_dir


## train 64x64
#python exp/dev/nerf_inr/scripts/train_v16.py \
#    --port 8888 \
#    --tl_config_file configs/train_ffhq.yaml \
#    --tl_command train_ffhq \
#    --modelarts True \
#    --tl_opts curriculum.new_attrs.image_list_file datasets/ffhq/ffhq_256.txt \
#      D_first_layer_warmup True \
#    --tl_outdir results/$resume_dir
#



