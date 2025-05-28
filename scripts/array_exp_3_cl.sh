#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --time 12:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=overcooked
#SBATCH -o /home/lvdenboogaard/slurm/%A_%a.out
#SBATCH --array=0-47%5

module load 2022
module load 2023
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source /etc/bashrc
source ~/.bashrc
source ~/venv/bin/activate

seeds=(0 1 2) # 2 3 4)
methods=(L2 EWC MAS AGEM)
seq_lengths=(5 25)


tid=$SLURM_ARRAY_TASK_ID  
n_seeds=${#seeds[@]}
n_task_ids=${#task_ids[@]}
n_multiheads=${#multiheads[@]}
# n_layer_norms=${#layer_norms[@]}
n_shared_backbone=${#shared_backbone[@]}
n_layouts=${#layouts[@]}
n_cnn=${#cnn[@]}

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

read i_seed i_task_id i_multihead i_shared_backbone i_layout i_cnn \
<<<"$(index "$tid" "$n_seeds" "$n_task_ids" "$n_multiheads" "$n_shared_backbone" "$n_layouts" "$n_cnn")"


seed=${seeds[$i_seed]}
task_id=${task_ids[$i_task_id]}
multihead=${multiheads[$i_multihead]}
shared_backbone=${shared_backbone[$i_shared_backbone]}
layout=${layouts[$i_layout]}
cnn=${cnn[$i_cnn]}

echo "Running experiment with seed $seed, architecture \
$task_id $multihead $shared_backbone $cnn, layout $layout"

PYTHONPATH=$HOME/JAXOvercooked python $HOME/JAXOvercooked/baselines/IPPO_CL.py \
  --seq_length  \
  --seed ${seed} \
  --anneal_lr \
  --tags $seed $layout $task_id $multihead $shared_backbone $cnn "EWC" " \
  --group "experiment 2" \
  --layouts "easy_levels" \
  --log_interval 100 \
  --eval_num_episodes 10 \
  --use_layer_norm \
  --$task_id \
  --$multihead \
  --$shared_backbone \
  --$cnn \
  --cl_method  \