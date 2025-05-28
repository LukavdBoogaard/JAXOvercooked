#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --time 12:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=overcooked
#SBATCH -o /home/lvdenboogaard/slurm/%A_%a.out
#SBATCH --array=0-23%5

module load 2022
module load 2023
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source /etc/bashrc
source ~/.bashrc
source ~/venv/bin/activate

seeds=(0 1) # 2 3 4)
architectures=(IPPO_MLP IPPO_CNN IPPO_decoupled_MLP)
tags=(
  "MLP baseline"
  "CNN baseline"
  "decoupled MLP baseline"
)
layouts=(
  "easy_levels"
  "medium_levels"
  "hard_levels"
)

tid=$SLURM_ARRAY_TASK_ID  
n_seeds=${#seeds[@]}
n_architectures=${#architectures[@]}
n_tags=${#tags[@]}
n_layouts=${#layouts[@]}

index() {                            # index tid base₀ base₁ …
    local val=$1; shift
    local bases=("$@")
    local i
    for (( i=0; i<${#bases[@]}; i++ )); do
        local stride=1
        for (( j=i+1; j<${#bases[@]}; j++ )); do
            stride=$(( stride * bases[j] ))
        done
        printf '%s ' $(( val / stride ))
        val=$(( val % stride ))
    done
}

read i_seed i_architecture i_tag <<<"$(index "$tid" "$n_seeds" "$n_architectures" "$n_layouts")"


seed=${seeds[$i_seed]}
architecture=${architectures[$i_architecture]}
tag=${tags[$i_architecture]}
layout=${layouts[$i_tag]}

echo "Running experiment with seed $seed, architecture $architecture, tag $tag, layout $layout"

PYTHONPATH=$HOME/JAXOvercooked python $HOME/JAXOvercooked/baselines/${architecture}.py \
  --seq_length 5 \
  --seed ${seed} \
  --anneal_lr \
  --tags "${tag}" "seed ${seed}" "${layout}" \
  --group "experiment 2" \
  --layouts "${layout}" \
  --log_interval 50 \
  --eval_num_episodes 10 \
  --evaluation

