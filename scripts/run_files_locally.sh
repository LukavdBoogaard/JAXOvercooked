#!/bin/bash
cd /home/luka/repo/JAXOvercooked

models=(IPPO_shared_MLP_L2
        IPPO_shared_MLP_EWC 
        IPPO_shared_MLP_MAS
        )

for model in "${models[@]}"; do
  echo "Running $model"
  # Run the model with the specified parameters
   python -m baselines.${model} --seq_length=2 --anneal_lr --evaluation --seed=0
done
