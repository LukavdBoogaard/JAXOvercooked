#!/bin/bash
source /data/vscail/anaconda3/bin/activate jaxovercooked
export MUJOCO_GL=egl
#export OMP_NUM_THREADS=6

cd ${HOME}/overcook/JAXOvercooked


CUDA_VISIBLE_DEVICES=$1 python -m baselines.IPPO_MLP_CBP \
    --log_interval 10 \
    --wandb_mode "disabled" \



