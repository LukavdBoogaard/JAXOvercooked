#!/bin/bash

seeds=(0)
architectures=(IPPO_MLP IPPO_CNN IPPO_decoupled_MLP IPPO_shared_MLP)
tags=(
  "MLP baseline"
  "CNN baseline"
  "decoupled MLP baseline"
  "shared MLP baseline"
)
layouts=(
  "easy_levels"
  "medium_levels"
  "hard_levels"
)

for idx in "${!architectures[@]}"; do
  architecture="${architectures[$idx]}"
  tag="${tags[$idx]}"
  for layout in "${layouts[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Submitting $architecture with layout=$layout and seed=$seed"

      cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 12:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=overcooked
#SBATCH -o /home/lvdenboogaard/slurm/%j.out

module load 2022
module load 2023
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source /etc/bashrc
source ~/.bashrc
source ~/venv/bin/activate

PYTHONPATH=\$HOME/JAXOvercooked python \$HOME/JAXOvercooked/baselines/${architecture}.py \
  --seq_length 5 \
  --seed ${seed} \
  --anneal_lr \
  --tags "${tag}" "seed ${seed}" "${layout}" \
  --group "experiment 2" \
  --layouts "${layout}" \
  --evaluation
EOF
    sleep 1
    done
  done
done