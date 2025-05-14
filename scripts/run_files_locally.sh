#!/bin/bash
cd /home/luka/repo/JAXOvercooked

models=(IPPO_CNN
        IPPO_multihead_L2 
        IPPO_shared_MLP_AGEM 
        IPPO_shared_MLP_EWC 
        IPPO_shared_MLP_MAS 
        IPPO_MLP_CBP
        )

for model in "${models[@]}"; do
  echo "Running $model"
  # Run the model with the specified parameters
   python -m baselines.${model} --seq_length=3 --anneal_lr --evaluation --seed=0
done
