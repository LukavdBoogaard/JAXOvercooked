#!/bin/bash
cd /home/luka/repo/JAXOvercooked

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  echo "Running $seed"
  # Run the model with the specified parameters
   python -m baselines.vdn_cnn --seq_length=2  \
                                --seed=$seed \
                                --group="test vdn" \
                                --tags $seed \
                                --layouts "easy_levels" \
                                --test_interval 0.05 \
                                --num_envs 64 
done
